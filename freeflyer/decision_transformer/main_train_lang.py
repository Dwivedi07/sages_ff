"""
Train Decision Transformer (ART) with structured logging and periodic plotting.

Features added:
- Argparse CLI for schedule knobs (eval/save/plot intervals, epochs, steps, etc.)
- Centralized evaluate()
- Log dict for train/eval: total, state, action losses
- PlotManager: saves PNGs for train/eval (state/action/total) at regular intervals
- Safe main-process-only I/O under Accelerate
- Optional CSV dump of the log for post-analysis
"""


import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import argparse

root_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(root_folder))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import DecisionTransformerConfig, get_scheduler
import torch.nn as nn

# ---------- In-project imports ----------
from decision_transformer.art import AutonomousFreeflyerTransformer_Lang, AutonomousFreeflyerTransformer_Lang_ctg
import decision_transformer.manage as ART_manager
from decision_transformer.manage import device
from decision_transformer.adapter import FrozenTextAdapter
from dataset_generation.dataset_pargen import load_behavior_texts, get_behavior_text_batch


# ============================ Utilities ============================

@dataclass
class TrainArgs:
    model_name: str = "v_05" # v_03: ctg based model, v04: no cttg normal IL model, v_02: with new short text cmds, v_05: model trained on 80k samples data
    dataset_name: str = "v02" # v01: new with cvx soln also having semantic info, v05: old without cvx slon haviing sematic info
    epochs: int = 1
    max_steps: int = 2500000 
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
    out_dir: str = "decision_transformer/saved_files/checkpoints"
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


def build_config_and_data(model_name: str, dataset_name: str):
    import_config = ART_manager.transformer_import_config(model_name)
    ctg_condition = import_config["ctg_condition"]
    timestep_norm = import_config["timestep_norm"]
    dataset_to_use = import_config["dataset_to_use"] 

    datasets, dataloaders = ART_manager.get_train_val_test_data(
        ctg_condition=ctg_condition, timestep_norm=timestep_norm, dataset_to_use=dataset_to_use, dataset_version=dataset_name, max_samples = None
    )
    train_loader, eval_loader, test_loader = dataloaders
    n_state = train_loader.dataset.n_state
    n_action = train_loader.dataset.n_action
    n_time = train_loader.dataset.max_len

    return (train_loader, eval_loader, test_loader), (n_state, n_action, n_time), ctg_condition


def build_model(n_state: int, n_action: int, n_time: int, hidden_size: int, n_layer: int, n_head: int, ctg_condition: bool):
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
    if ctg_condition:
        model = AutonomousFreeflyerTransformer_Lang_ctg(config)
    else:
        model = AutonomousFreeflyerTransformer_Lang(config)
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
    command_mapping,
    eval_iters: int,
    ctg_condition: bool,
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

        states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix, behavior_i, command_id_i = batch

        # text -> embeddings
        command_txt_i = get_behavior_text_batch(command_mapping, behavior_i, command_id_i)
        commands_emb_i = text_encoder(command_txt_i)
        # command_txt_i = [get_behavior_text(command_mapping, behavior_i[j], command_id_i[j]) for j in range(len(behavior_i))]
        # commands_emb_i = text_encoder(command_txt_i)

        # forward
        if ctg_condition:
            state_preds, action_preds = model(
                states=states_i,
                actions=actions_i,
                constraints=ctgs_i,
                commands_emb=commands_emb_i,
                timesteps=timesteps_i,
                attention_mask=attention_mask_i,
                return_dict=False,
            )
        else:
            state_preds, action_preds = model(
                states=states_i,
                actions=actions_i,
                goal=goal_i,
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

    p.add_argument("--model_name", type=str)
    p.add_argument("--dataset_name", type=str)
    p.add_argument("--ctg_condition", type=bool)
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

    p.add_argument("--hidden_size", type=int)
    p.add_argument("--n_layer", type=int)
    p.add_argument("--n_head", type=int)

    p.add_argument("--eval_iters", type=int)


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
    (train_loader, eval_loader, _), (n_state, n_action, n_time), args.ctg_condition = build_config_and_data(args.model_name, args.dataset_name)
    print(f"CTG condition set to: {args.ctg_condition} ")
    model = build_model(n_state, n_action, n_time, args.hidden_size, args.n_layer, args.n_head, args.ctg_condition)
    

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

    # accelerator.register_for_checkpointing(lr_scheduler)
    # load a previous checkpoint if one exists (both ranks) 
    model_dir = root_folder / args.out_dir / args.model_name
    if os.path.isdir(model_dir) and any(fname.endswith(".bin") or fname.startswith("pytorch_model") for fname in os.listdir(model_dir)):
        accelerator.load_state(model_dir)
        accelerator.print(f"Resumed from {model_dir}")
    else:
        accelerator.print("No previous state found, training from scratch.") 

    # --------- Text encoder & commands ---------
    MODEL = os.getenv("FTA_MODEL", "distilbert-base-uncased")
    text_encoder = FrozenTextAdapter(model_name=MODEL, out_dim=args.hidden_size, output_mode="tokens").to(device).eval()
    # master_file.json: keys "0".."26"; get_behavior_text_batch remaps keys via behavior_mode_to_text_key
    # so language matches horizon (slow <-> long k_T). Old checkpoints: pass use_text_key_remap=False.
    command_mapping = load_behavior_texts(root_folder / "dataset" / "master_file_new.json")

    # --------- Output & plotting ---------
    out_dir = (root_folder / args.out_dir).resolve()
    plotter = PlotManager(out_dir=out_dir, version=args.model_name)

    # --------- Initial eval ---------
    eval_total, eval_state, eval_action = evaluate(
        accelerator, model, eval_dataloader, text_encoder, command_mapping, args.eval_iters, args.ctg_condition
    )
    accelerator.print(
        {"loss/eval": eval_total, "loss/state": eval_state, "loss/action": eval_action}
    )

    # --------- Logs ---------
    # We store one entry per "eval event" (to keep steps aligned across train/eval)
    log = {
        "eval_loss": [],
        "eval_loss_state": [],
        "eval_loss_action": [],
        "train_loss": [],
        "train_loss_state": [],
        "train_loss_action": [],
    }
    eval_steps_axis: List[int] = []

    # Seed (optional): uncomment if we want deterministic runs
    # torch.manual_seed(4); np.random.seed(4)

    # --------- Train loop ---------
    samples_per_step = accelerator.state.num_processes * train_loader.batch_size
    global_step = 0
    completed_steps = 0

    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dataloader, start=0):
            if completed_steps >= args.max_steps:
                break

            with accelerator.accumulate(model):
                states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix, behavior_i, command_id_i = batch

                # text -> embeddings
                command_txt_i = get_behavior_text_batch(command_mapping, behavior_i, command_id_i)
                commands_emb_i = text_encoder(command_txt_i)

                # forward
                if args.ctg_condition:
                    state_preds, action_preds = model(
                        states=states_i,
                        actions=actions_i,
                        constraints=ctgs_i,
                        commands_emb=commands_emb_i,
                        timesteps=timesteps_i,
                        attention_mask=attention_mask_i,
                        return_dict=False,
                    )
                else:
                     state_preds, action_preds = model(
                        states=states_i,
                        actions=actions_i,
                        goal=goal_i,
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
                    accelerator, model, eval_dataloader, text_encoder, command_mapping, args.eval_iters, args.ctg_condition
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