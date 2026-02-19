"""
Train Decision Transformer (ART) with structured logging and periodic plotting.
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import argparse

root_folder = Path(__file__).resolve().parent.parent.parent  # /art_lang/
sys.path.append(str(root_folder))

# Fix for RTX 4000 series GPU compatibility with Accelerate
# Disable P2P and IB communication which are not supported on RTX 4000 series
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import DecisionTransformerConfig, get_scheduler
import torch.nn as nn

# ---------- In-project imports ----------
from rpod.decision_transformer.art import AutonomousRendezvousTransformerLang
import rpod.decision_transformer.manage as ART_manager
from rpod.decision_transformer.manage import device
from rpod.decision_transformer.adapter import FrozenTextAdapter

# ============================ Utilities ============================
@dataclass
class TrainArgs:
    dataset_name: str = "v08"
    model_name : str = "v08_w3"
    command_file_train: str = "commands_summary_w3_train.jsonl"
    command_file_eval : str = "commands_summary_w3_val.jsonl"
    ctg_condition: bool = True
    tailored_command : bool = True  
    equal_datasize: bool = False   # if True, data-size is ensured to be equal across behaviors by downsampling larger ones
    max_tokens: int = 50
    
    epochs: int = 1
    max_steps: int = 2.5e6
    lr: float = 3e-5
    weight_decay: float = 0.0
    warmup_steps: int = 10
    grad_accum: int = 8
    mixed_precision: str = "no"  

    # intervals (in optimizer steps)
    eval_every: int = 500
    save_every: int = 5_000
    plot_every: int = 500

    # logging/outputs
    out_dir: str = "rpod/decision_transformer/saved_files/checkpoints"
    log_csv: bool = False

    # model width
    hidden_size: int = 384
    n_layer: int = 6
    n_head: int = 6

    # evaluation
    eval_iters: int = 100  

    def display(self):
        args_dict = asdict(self)
        print("\n" + "=" * 60)
        print(" Training Configuration")
        print("=" * 60)
        key_width = max(len(k) for k in args_dict.keys()) + 2
        for k, v in args_dict.items():
            print(f"{k:<{key_width}} : {v}")
        print("=" * 60 + "\n")

class PlotManager:
    def __init__(self, out_dir: Path, version: str):
        self.plots_dir = out_dir / version / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def _plot_dual_series(
        self,
        xs_train: List[int],
        ys_train: List[float],
        xs_eval: List[int],
        ys_eval: List[float],
        title: str,
        ylabel: str,
        filename: Path,
        log_scale: bool = True,
    ):
        # Align by lengths in case one side is shorter
        n_train = min(len(xs_train), len(ys_train))
        n_eval = min(len(xs_eval), len(ys_eval))
        if n_train == 0 and n_eval == 0:
            return

        plt.figure()
        if n_train > 0:
            plt.plot(xs_train[:n_train], ys_train[:n_train], label="Train", linewidth=1.6)
        if n_eval > 0:
            plt.plot(xs_eval[:n_eval], ys_eval[:n_eval], label="Eval", linewidth=1.6)
        plt.title(title)
        plt.xlabel("Global Step")
        plt.ylabel(ylabel)
        if log_scale:
            plt.yscale("log")
            plt.ylabel(f"{ylabel} (log scale)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_all(self, log: Dict[str, List[float]], steps: List[int]):
        """
        Save only three figures:
          - action_loss.png   (Train + Eval)
          - state_loss.png    (Train + Eval)
          - combined_loss.png (Train + Eval)
        Notes:
          * We assume `steps` are the x-axis for *eval events*. We plot train snapshots
            taken at the same eval events against the same `steps`.
        """
        if len(steps) == 0:
            return

        # Guard for keys (in case of partial logs early in training)
        for k in ["train_loss", "train_loss_state", "train_loss_action",
                  "eval_loss", "eval_loss_state", "eval_loss_action"]:
            if k not in log:
                log[k] = []

        # 1) ACTION: Train vs Eval
        self._plot_dual_series(
            xs_train=steps, ys_train=log["train_loss_action"],
            xs_eval=steps, ys_eval=log["eval_loss_action"],
            title="Action Loss (Train vs Eval)",
            ylabel="MSE",
            filename=self.plots_dir / "action_loss.png",
        )

        # 2) STATE: Train vs Eval
        self._plot_dual_series(
            xs_train=steps, ys_train=log["train_loss_state"],
            xs_eval=steps, ys_eval=log["eval_loss_state"],
            title="State Loss (Train vs Eval)",
            ylabel="MSE",
            filename=self.plots_dir / "state_loss.png",
        )

        # 3) COMBINED: Train vs Eval
        self._plot_dual_series(
            xs_train=steps, ys_train=log["train_loss"],
            xs_eval=steps, ys_eval=log["eval_loss"],
            title="Combined Loss (Train vs Eval)",
            ylabel="MSE (state + action)",
            filename=self.plots_dir / "combined_loss.png",
        )


def build_config_and_data(dataset_name: str, ctg_condition: bool, timestep_norm: bool = False, equal_datasize: bool = False, tailored_data: bool = False):
    
    datasets, dataloaders = ART_manager.get_train_val_test_data(
        ctg_condition=ctg_condition, 
        dataset_name=dataset_name, 
        timestep_norm=timestep_norm, 
        equal_datasize=equal_datasize, 
        tailored_data=tailored_data
    )
    train_loader, eval_loader, test_loader = dataloaders
    n_state = train_loader.dataset.n_state
    n_action = train_loader.dataset.n_action
    n_time = train_loader.dataset.max_len

    return (train_loader, eval_loader, test_loader), (n_state, n_action, n_time)


def build_model(n_state: int, n_action: int, n_time: int, hidden_size: int, n_layer: int, n_head: int):
    config = DecisionTransformerConfig(
        state_dim=n_state,
        act_dim=n_action,
        hidden_size=hidden_size,
        max_ep_len=n_time,
        vocab_size=1,
        action_tanh=False,
        n_positions=1024,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=None,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = AutonomousRendezvousTransformerLang(config)
    model.to(device)
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    return model

# -----------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    eval_dataloader,
    text_encoder,
    command_mapping_eval,
    eval_iters: int,
    tailored_command: bool,
) -> Tuple[float, float, float]:
    
    model.eval()

    total_losses = []
    state_losses = []
    action_losses = []

    data_iter = iter(eval_dataloader)
    for _ in range(eval_iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(eval_dataloader)
            batch = next(data_iter)

        if not tailored_command:
            states_i, actions_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix, behavior_i, command_id_i = batch

            # text -> embeddings; randomly sampling from the command_mapping_eval for evaluation
            command_txt_i = []
            for j in range(len(behavior_i)):
                behav_ij = behavior_i[j].item()
                len_command = len(command_mapping_eval[behav_ij]['description'])
                idx_ij = np.random.randint(0, len_command)
                command_txt_i.append(command_mapping_eval[behav_ij]['description'][idx_ij])

        else:
            states_i, actions_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix, behavior_i, command_id_i, command_txt_i, wyp_i, wyp_times_i = batch
            
            # we can generate a spontenaous command here for evaluation if needed
            # command_txt_i = annotate_eval_date()
            

        commands_emb_i = text_encoder(command_txt_i)

        state_preds, action_preds = model(
            states=states_i,
            actions=actions_i,
            constraints=ctgs_i,
            commands_emb=commands_emb_i,
            timesteps=timesteps_i,
            attention_mask=attention_mask_i,
            return_dict=False,
        )

        loss_action = torch.mean((action_preds - actions_i) ** 2)
        loss_state = torch.mean((state_preds[:, :-1, :] - states_i[:, 1:, :]) ** 2)
        loss = loss_action + loss_state

        # gather to main process numerics
        total_losses.append(accelerator.gather(loss.detach()).mean().item())
        state_losses.append(accelerator.gather(loss_state.detach()).mean().item())
        action_losses.append(accelerator.gather(loss_action.detach()).mean().item())

    model.train()
    return float(np.mean(total_losses)), float(np.mean(state_losses)), float(np.mean(action_losses))

def maybe_save_checkpoint(accelerator: Accelerator, text_encoder: FrozenTextAdapter, out_path: Path, model_state_only: bool = False):
    out_path.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(out_path))
    text_encoder.save_adapter(str(out_path / "text_adapter.pth"))

def dump_log_npz_and_csv(out_dir: Path, version: str, log: Dict[str, List[float]], steps: List[int], write_csv: bool):
    run_dir = out_dir / version
    run_dir.mkdir(parents=True, exist_ok=True)

    # NPZ
    np.savez_compressed(run_dir / "log.npz", steps=np.array(steps), **{k: np.array(v) for k, v in log.items()})

    # CSV (optional)
    if write_csv:
        import csv
        csv_path = run_dir / "log.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["step"] + list(log.keys())
            writer.writerow(header)
            for i in range(len(steps)):
                row = [steps[i]] + [log[k][i] if i < len(log[k]) else "" for k in log.keys()]
                writer.writerow(row)

# ============================ Main ============================

def main(cli: Optional[TrainArgs] = None):
    # --------- Parse CLI ---------
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_name", type=str)
    p.add_argument("--model_name", type=str)
    p.add_argument("--epochs", type=int)
    p.add_argument("--max_steps", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--warmup_steps", type=int)
    p.add_argument("--grad_accum", type=int)
    p.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"])

    p.add_argument("--eval_every", type=int)
    p.add_argument("--save_every", type=int)
    p.add_argument("--plot_every", type=int)

    p.add_argument("--out_dir", type=str)
    p.add_argument("--log_csv", action="store_true")
    
    p.add_argument("--command_file", type=str)

    p.add_argument("--hidden_size", type=int)
    p.add_argument("--n_layer", type=int)
    p.add_argument("--n_head", type=int)

    p.add_argument("--eval_iters", type=int)
    p.add_argument("--ctg_condition", type=bool, default=True)
    p.add_argument("--tailored_command", type=bool, default=True)
    p.add_argument("--equal_datasize", type=bool, default=False)
    p.add_argument("--max_tokens", type=int, default=30)

    if len(sys.argv) > 1 and cli is None:
        args_ns = p.parse_args()
        args = TrainArgs(**{**asdict(TrainArgs()), **vars(args_ns)})
    else:
        # no CLI args use TrainArgs() or provided cli
        args = cli if cli is not None else TrainArgs()

    args.display()

    # --------- Accelerator ---------
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum,
    )

    # --------- Data & Model ---------
    (train_loader, eval_loader, _), (n_state, n_action, n_time) = build_config_and_data(args.dataset_name, args.ctg_condition, equal_datasize=args.equal_datasize, tailored_data=args.tailored_command)
    model = build_model(n_state, n_action, n_time, args.hidden_size, args.n_layer, args.n_head)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )
    
    # --------- LR Scheduler ---------
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # load a previous checkpoint if one exists (both ranks) 
    model_dir = root_folder / args.out_dir / args.model_name
    if os.path.isdir(model_dir) and any(fname.endswith(".bin") or fname.startswith("pytorch_model") for fname in os.listdir(model_dir)):
        accelerator.load_state(model_dir)
        accelerator.print(f"Resumed from {model_dir}")
    else:
        accelerator.print("No previous state found, training from scratch.")    

    # --------- Text encoder & commands ---------
    MODEL = os.getenv("FTA_MODEL", "distilbert-base-uncased")
    text_encoder = FrozenTextAdapter(model_name=MODEL, out_dim=args.hidden_size, output_mode="tokens", max_tokens=args.max_tokens).to(device).eval()
    command_mapping = []
    with open(root_folder / "rpod/dataset" / args.command_file_train , "r") as f:
        for line in f:
            command_mapping.append(json.loads(line))
            
    command_mapping_eval = []
    with open(root_folder / "rpod/dataset" / args.command_file_eval , "r") as f:
        for line in f:
            command_mapping_eval.append(json.loads(line))
    
    
    # --------- Output & plotting ---------
    out_dir = (root_folder / args.out_dir).resolve()
    plotter = PlotManager(out_dir=out_dir, version=args.model_name)

    # --------- Initial eval ---------
    eval_total, eval_state, eval_action = evaluate(
        accelerator, model, eval_dataloader, text_encoder, command_mapping_eval, args.eval_iters, args.tailored_command
    )
    accelerator.print(
        {"loss/eval": eval_total, "loss/state": eval_state, "loss/action": eval_action}
    )

    # --------- Logs ---------
    # We store one entry per "eval event" (to keep steps aligned across train/eval)
    log_dir = model_dir / "log.npz"
    
    if log_dir.is_file():
        with np.load(log_dir) as data:  # ensures the file is closed promptly
            # steps stored separately; keep losses in `log`
            eval_steps_axis = data["steps"].tolist()
            log = {k: data[k].tolist() for k in data.files if k != "steps"}
        # optional: sanity checks
        for k, v in log.items():
            if len(v) != len(eval_steps_axis):
                raise ValueError(f"Length mismatch: {k} has {len(v)} entries but steps has {len(eval_steps_axis)}.")
        accelerator.print(f"Found a previous log, loaded from {log_dir}")
    else:
        log = {
            "eval_loss": [],
            "eval_loss_state": [],
            "eval_loss_action": [],
            "train_loss": [],
            "train_loss_state": [],
            "train_loss_action": [],
        }
        eval_steps_axis: list[int] = []

    print("\n" + "=" * 60)
    
    # Seed (optional): uncomment if we want deterministic runs
    # torch.manual_seed(4); np.random.seed(4)

    # --------- Train loop ---------
    samples_per_step = accelerator.state.num_processes * train_loader.batch_size
    last_logged_step = eval_steps_axis[-1] if len(eval_steps_axis) > 0 else 0
    global_step = last_logged_step
    completed_steps = last_logged_step

    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dataloader, start=0):
            if completed_steps >= args.max_steps:
                break

            with accelerator.accumulate(model):
                
                if not args.tailored_command:
                    states_i, actions_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix, behavior_i, command_id_i = batch
                    # text -> embeddings
                    command_txt_i = []
                    for j in range(len(behavior_i)):
                        command_txt_i.append(command_mapping[behavior_i[j]]['description'][command_id_i[j]]) 
                else:
                    states_i, actions_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix, behavior_i, command_id_i, command_txt_i, wyp_i, wyp_times_i = batch
                
                commands_emb_i = text_encoder(command_txt_i)

                # forward
                state_preds, action_preds = model(
                    states=states_i,
                    actions=actions_i,
                    constraints=ctgs_i,
                    commands_emb=commands_emb_i,
                    timesteps=timesteps_i,
                    attention_mask=attention_mask_i,
                    return_dict=False,
                )

                loss_action = torch.mean((action_preds - actions_i) ** 2)
                loss_state = torch.mean((state_preds[:, :-1, :] - states_i[:, 1:, :]) ** 2)
                loss = loss_action + loss_state

                if step % 100 == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    accelerator.print(
                        {
                            "lr": cur_lr,
                            "samples_seen": global_step * samples_per_step,
                            "global_step": global_step,
                            "loss/train": float(loss.detach().item()),
                        }
                    )

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            completed_steps += 1

            # ------ Periodic eval/log ------
            if (global_step % args.eval_every) == 0 or global_step == 1:
                eval_total, eval_state, eval_action = evaluate(
                    accelerator, model, eval_dataloader, text_encoder, command_mapping_eval, args.eval_iters, args.tailored_command
                )
                # Store eval + latest train snapshot (both at same x-axis step)
                log["eval_loss"].append(eval_total)
                log["eval_loss_state"].append(eval_state)
                log["eval_loss_action"].append(eval_action)
                log["train_loss"].append(float(loss.detach().item()))
                log["train_loss_state"].append(float(loss_state.detach().item()))
                log["train_loss_action"].append(float(loss_action.detach().item()))
                eval_steps_axis.append(global_step)

                accelerator.print(
                    {
                        "step": global_step,
                        "loss/eval_total": eval_total,
                        "loss/eval_state": eval_state,
                        "loss/eval_action": eval_action,
                        "loss/train_total": float(loss.detach().item()),
                        "loss/train_state": float(loss_state.detach().item()),
                        "loss/train_action": float(loss_action.detach().item()),
                    }
                )

            # ------ Periodic plotting ------
            if (global_step % args.plot_every) == 0 and accelerator.is_main_process:
                plotter.save_all(log, eval_steps_axis)

            # ------ Periodic checkpoint ------
            if (global_step % args.save_every) == 0 and accelerator.is_main_process:
                run_dir = out_dir / args.model_name
                accelerator.print(f"[{global_step}] Saving checkpoint & logs to: {run_dir}")
                maybe_save_checkpoint(accelerator, text_encoder, run_dir)
                dump_log_npz_and_csv(out_dir, args.model_name, log, eval_steps_axis, write_csv=args.log_csv)

        if completed_steps >= args.max_steps:
            break

    # --------- Final save ---------
    if accelerator.is_main_process:
        run_dir = out_dir / args.model_name
        accelerator.print(f"[final] Saving checkpoint & logs to: {run_dir}")
        maybe_save_checkpoint(accelerator, text_encoder, run_dir)
        dump_log_npz_and_csv(out_dir, args.model_name, log, eval_steps_axis, write_csv=args.log_csv)
        plotter.save_all(log, eval_steps_axis)


if __name__ == "__main__":
    main()
