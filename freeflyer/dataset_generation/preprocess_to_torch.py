"""
Standalone script to load saved dataset npz files (scp, cvx, param) and run
preprocessing: compute reward-to-go and constraint-to-go, then save torch_*.pth
and permutation.npy under dataset/torch/{dataset_version}/.

Use when dataset generation completed saving npz but preprocessing was interrupted
(e.g. power loss). Run from repo root or with correct PYTHONPATH so that
freeflyer.optimization and freeflyer.dynamics are importable.

Usage:
  python -m freeflyer.dataset_generation.preprocess_to_torch [--version v02]
  # or from freeflyer/dataset_generation:
  python preprocess_to_torch.py --version v02
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

# Path so we can import optimization / dynamics
root_folder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_folder))

import optimization.ff_scenario as ff
from dynamics.freeflyer import compute_constraint_to_go, compute_reward_to_go


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset npz -> torch tensors")
    parser.add_argument("--version", type=str, default="v02", help="Dataset version (e.g. v02)")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Override dataset dir (default: root_folder/dataset)")
    args = parser.parse_args()

    dataset_version = args.version
    dataset_dir = args.dataset_dir or (root_folder / "dataset")
    torch_dir = root_folder / "dataset" / "torch" / dataset_version
    torch_dir.mkdir(parents=True, exist_ok=True)

    base = f"dataset-ff-{dataset_version}"
    scp_path = dataset_dir / f"{base}-scp.npz"
    cvx_path = dataset_dir / f"{base}-cvx.npz"
    param_path = dataset_dir / f"{base}-param.npz"

    for p in (scp_path, cvx_path, param_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    print("Loading npz files...")
    scp = np.load(scp_path, allow_pickle=True)
    cvx = np.load(cvx_path, allow_pickle=True)
    par = np.load(param_path, allow_pickle=True)

    states_scp = scp["states_scp"]
    actions_scp = scp["actions_scp"]
    states_cvx = cvx["states_cvx"]
    actions_cvx = cvx["actions_cvx"]

    # Param keys: support both "behavior_mode" and legacy "behavior"
    behavior_mode = par["behavior_mode"] if "behavior_mode" in par else par["behavior"]
    command_id = par["command_id"]
    region_id_arr = par["region_id"]
    time_id_arr = par["time_id"]
    k_T_arr = par["k_T"]
    goal_xy_arr = par["goal_xy"]

    n = states_scp.shape[0]
    assert states_cvx.shape[0] == n, "scp and cvx sample count mismatch"
    assert behavior_mode.shape[0] == n, "param and scp sample count mismatch"

    print(f"Loaded {n} samples. Computing reward-to-go and constraint-to-go...")
    torch_rtgs_scp = torch.from_numpy(compute_reward_to_go(actions_scp))
    torch_rtgs_cvx = torch.from_numpy(compute_reward_to_go(actions_cvx))

    obs = copy.deepcopy(ff.obs)
    obs["radius"] = (obs["radius"] + ff.robot_radius) * ff.safety_margin
    torch_ctgs_scp = torch.from_numpy(
        compute_constraint_to_go(states_scp, obs["position"], obs["radius"])
    )
    torch_ctgs_cvx = torch.from_numpy(
        compute_constraint_to_go(states_cvx, obs["position"], obs["radius"])
    )

    print("Converting to torch and saving...")
    torch.save(torch.from_numpy(states_scp), torch_dir / "torch_states_scp.pth")
    torch.save(torch.from_numpy(states_cvx), torch_dir / "torch_states_cvx.pth")
    torch.save(torch.from_numpy(actions_scp), torch_dir / "torch_actions_scp.pth")
    torch.save(torch.from_numpy(actions_cvx), torch_dir / "torch_actions_cvx.pth")
    torch.save(torch.from_numpy(behavior_mode), torch_dir / "torch_behavior_mode.pth")
    torch.save(torch.from_numpy(command_id), torch_dir / "torch_command_id.pth")
    torch.save(torch.from_numpy(region_id_arr), torch_dir / "torch_region_id.pth")
    torch.save(torch.from_numpy(time_id_arr), torch_dir / "torch_time_id.pth")
    torch.save(torch.from_numpy(k_T_arr), torch_dir / "torch_k_T.pth")
    torch.save(torch.from_numpy(goal_xy_arr), torch_dir / "torch_goal_xy.pth")
    torch.save(torch_rtgs_scp, torch_dir / "torch_rtgs_scp.pth")
    torch.save(torch_rtgs_cvx, torch_dir / "torch_rtgs_cvx.pth")
    torch.save(torch_ctgs_scp, torch_dir / "torch_ctgs_scp.pth")
    torch.save(torch_ctgs_cvx, torch_dir / "torch_ctgs_cvx.pth")

    perm = np.random.permutation(n * 2)
    np.save(torch_dir / "permutation.npy", perm)

    print(f"Done. Torch files written to {torch_dir}")


if __name__ == "__main__":
    main()
