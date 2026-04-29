import os
import sys
import json
from pathlib import Path
import itertools
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, set_start_method
import copy
from typing import Dict, List

import torch

root_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(root_folder))

import optimization.ff_scenario as ff
from dynamics.freeflyer import (
    FreeflyerModel,
    sample_init_target,
    ocp_no_obstacle_avoidance,
    ocp_obstacle_avoidance,
    compute_constraint_to_go,
    compute_reward_to_go,
)
from optimization.ff_scenario import (
    N_STATE, N_ACTION, N_CLUSTERS,
    n_time_rpod, dt, T
)

# --------------------------------------------------------------------------------
# -------- helpers --------
# --------------------------------------------------------------------------------
def _flatten_to_ints(x) -> List[int]:
    """Flatten x to list[int]. Handles tensors/ndarrays/lists/tuples/scalars."""
    out: List[int] = []
    if torch.is_tensor(x):
        out.extend(int(v) for v in x.detach().cpu().reshape(-1).tolist())
    elif isinstance(x, np.ndarray):
        out.extend(int(v) for v in x.reshape(-1).tolist())
    elif isinstance(x, (list, tuple)):
        for item in x:
            out.extend(_flatten_to_ints(item))
    else:
        out.append(int(x))
    return out


def _to_int_list(x) -> List[int]:
    return _flatten_to_ints(x)


# --------------------------------------------------------------------------------
# -------------------------CASE LABELS (optional; kept for compatibility)
# --------------------------------------------------------------------------------
def load_behavior_texts(json_path=root_folder / "dataset" / "master_file_new.json"):
    """
    Loads the behavior-mode -> command_id -> text mapping
    into an O(1)-lookup dictionary.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    mapping = {}
    for key, items in data.items():
        mapping[int(key)] = {item["command_id"]: item["text"] for item in items}

    return mapping


def behavior_mode_to_text_key(behavior_mode: int) -> int:
    """
    Map stored behavior_mode (physics label) to master_file.json top-level key for language.

    Stored behavior_mode = 9 * time_id + region_id (time_id 0,1,2 -> k_T 60,80,100).

    master_file groups use "slow" / "moderate" / "fast" wording aligned with **long / mid / short**
    planning horizons. The JSON was authored so that keys 0-8 read "slow" but those modes are
    time_id=0 (shortest k_T). This remap swaps time_id 0 <-> 2 for **text lookup only**, so:
      - long horizon (time_id 2, k_T=100) -> slow-vocabulary bucket (original keys 0-8)
      - short horizon (time_id 0, k_T=60) -> fast-vocabulary bucket (original keys 18-26)
      - time_id 1 unchanged (moderate, keys 9-17).

    Physics tensors and behavior_mode in the dataset are unchanged.
    """
    b = int(behavior_mode)
    region_id = b % 9
    time_id = b // 9
    return time_id * 9 + region_id
    # return (2 - time_id) * 9 + region_id


def get_behavior_text(mapping, key, command_id, use_text_key_remap: bool = True):
    """Returns the text corresponding to (key, command_id) pair.

    If use_text_key_remap (default True), key is passed through behavior_mode_to_text_key
    so language matches horizon semantics (slow <-> long k_T).
    """
    k = behavior_mode_to_text_key(int(key)) if use_text_key_remap else int(key)
    try:
        return mapping[k][command_id]
    except KeyError:
        raise KeyError(f"No entry for key={k} (from behavior={key}), command_id={command_id}")


def get_behavior_text_batch(
    mapping: Dict[int, Dict[int, str]], keys, command_ids, use_text_key_remap: bool = True
) -> List[str]:
    ks = _to_int_list(keys)
    cids = _to_int_list(command_ids)
    if len(ks) != len(cids):
        raise ValueError(f"Length mismatch: len(keys)={len(ks)} vs len(command_ids)={len(cids)}")
    out: List[str] = []
    for k, cid in zip(ks, cids):
        tk = behavior_mode_to_text_key(k) if use_text_key_remap else k
        try:
            out.append(mapping[tk][cid])
        except KeyError:
            raise KeyError(f"No entry for key={tk} (from behavior={k}), command_id={cid}.")
    return out


# --------------------------------------------------------------------------------
# -------- NEW: generalized behavior modes via (space region x time horizon)
# --------------------------------------------------------------------------------
#
# BEHAVIOR MODE ENCODING (behavior_mode = 9 * time_id + region_id):
#
#   There are 27 behavior modes: 9 regions × 3 time horizons.
#
#   REGION (region_id 0..8): 3×3 grid of goal regions for x > 1.2 (row-major).
#     Layout:  for r in rows (y), for c in cols (x)  =>  region_id = r*3 + c
#
#     region_id │ grid (x→, y↑)   │ description
#     ---------─┼─────────────────┼────────────────────────────────────────
#        0      │ (c=0, r=0)      │ low-x,  low-y   (left,   bottom)
#        1      │ (c=1, r=0)      │ mid-x,  low-y   (center, bottom)
#        2      │ (c=2, r=0)      │ high-x, low-y   (right,  bottom)
#        3      │ (c=0, r=1)      │ low-x,  mid-y   (left,   middle)
#        4      │ (c=1, r=1)      │ mid-x,  mid-y   (center, center)
#        5      │ (c=2, r=1)      │ high-x, mid-y   (right,  middle)
#        6      │ (c=0, r=2)      │ low-x,  high-y  (left,   top)
#        7      │ (c=1, r=2)      │ mid-x,  high-y  (center, top)
#        8      │ (c=2, r=2)      │ high-x, high-y  (right,  top)
#
#   TIME (time_id 0..2): terminal step k_T (horizon length). 40 removed (infeasible).
#
#     time_id   │  k_T   │ description
#     ---------─┼────────┼────────────────
#        0      │  60    │
#        1      │  80    │
#        2      │ 100    │ full horizon
#
#   Decode:  region_id = behavior_mode % 9;  time_id = behavior_mode // 9
#
#   Language (master_file.json): use behavior_mode_to_text_key(behavior_mode) when
#   resolving text so "slow"/"fast" wording matches long/short k_T (see that helper).
#
# --------------------------------------------------------------------------------

def build_goal_regions_3x3_xgt12(x_min=1.2, ncols=3, nrows=3):
    """
    Build 3x3 grid regions for x > x_min within table bounds.
    Returns list of regions with (xlo, xhi, ylo, yhi), length ncols*nrows (=9).
    """
    table_lo = np.asarray(ff.table["xy_low"], dtype=float)
    table_hi = np.asarray(ff.table["xy_up"], dtype=float)

    xlo = max(float(x_min), float(table_lo[0]))
    xhi = float(table_hi[0])
    ylo = float(table_lo[1])
    yhi = float(table_hi[1])

    if xhi <= xlo:
        raise ValueError(f"Table x upper ({xhi}) <= x_min ({xlo}); cannot discretize x>1.2.")

    xs = np.linspace(xlo, xhi, ncols + 1)
    ys = np.linspace(ylo, yhi, nrows + 1)

    regions = []
    for r in range(nrows):      # y direction
        for c in range(ncols):  # x direction
            regions.append((xs[c], xs[c + 1], ys[r], ys[r + 1]))
    return regions


def _shrink_interval(lo, hi, margin):
    lo2 = lo + margin
    hi2 = hi - margin
    if hi2 <= lo2:
        return None
    return lo2, hi2


def is_goal_clear_of_obstacles(p_xy: np.ndarray) -> bool:
    """
    Reject if goal lies within any inflated obstacle:
      inflated_radius = (obs_radius + robot_radius) * safety_margin
    """
    c_all = np.asarray(ff.obs["position"], dtype=float)  # (3,2)
    r_all = np.asarray(ff.obs["radius"], dtype=float)    # (3,)
    robot_r = float(ff.robot_radius)
    gamma = float(ff.safety_margin)

    inflated = (r_all + robot_r) * gamma
    d = np.linalg.norm(p_xy - c_all, axis=1)
    return bool(np.all(d >= inflated))


def sample_goal_in_region(
    rng,
    region,
    table_boundary_buffer=0.05,
    region_boundary_buffer=0.05,
    max_tries=512,
):
    """
    Sample a goal point uniformly in a region but avoid:
      - table edges (table_boundary_buffer)
      - region grid lines (region_boundary_buffer)
      - obstacles (inflated)
    """
    table_lo = np.asarray(ff.table["xy_low"], dtype=float)
    table_hi = np.asarray(ff.table["xy_up"], dtype=float)

    rx_lo, rx_hi, ry_lo, ry_hi = region

    # Table boundary buffer
    xlo = max(rx_lo, table_lo[0] + table_boundary_buffer)
    xhi = min(rx_hi, table_hi[0] - table_boundary_buffer)
    ylo = max(ry_lo, table_lo[1] + table_boundary_buffer)
    yhi = min(ry_hi, table_hi[1] - table_boundary_buffer)

    # Region boundary buffer (stay away from internal grid lines)
    xshr = _shrink_interval(xlo, xhi, region_boundary_buffer)
    yshr = _shrink_interval(ylo, yhi, region_boundary_buffer)
    if xshr is None or yshr is None:
        raise ValueError("Region too small after buffers; reduce buffers or adjust x_min/grid.")

    xlo, xhi = xshr
    ylo, yhi = yshr

    for _ in range(max_tries):
        p = np.array([rng.uniform(xlo, xhi), rng.uniform(ylo, yhi)], dtype=float)
        if is_goal_clear_of_obstacles(p):
            return p

    raise RuntimeError("Failed to sample a collision-free goal in region after many tries.")


def sample_time_horizon_from_last_4_chunks(rng):
    """
    Choose from 3 terminal horizons {60, 80, 100}. (40 removed: too many infeasibilities.)
    Returns (time_id, k_T).
    """
    k_choices = np.array([60, 80, 100], dtype=int)
    time_id = int(rng.integers(0, len(k_choices)))
    return time_id, int(k_choices[time_id])


def pad_to_full_horizon(states_T, actionsG_T, actionsT_T, k_T, n_time_rpod, hold_state=True):
    """
    Inputs:
      states_T   : (k_T, N_STATE)
      actionsG_T : (k_T, N_ACTION)
      actionsT_T : (k_T, N_CLUSTERS)
    Returns:
      states_full   : (n_time_rpod, N_STATE)
      actionsG_full : (n_time_rpod, N_ACTION)
      actionsT_full : (n_time_rpod, N_CLUSTERS)
    """
    N_S = states_T.shape[1]
    N_A = actionsG_T.shape[1]
    N_C = actionsT_T.shape[1]

    states_full = np.zeros((n_time_rpod, N_S), dtype=states_T.dtype)
    actionsG_full = np.zeros((n_time_rpod, N_A), dtype=actionsG_T.dtype)
    actionsT_full = np.zeros((n_time_rpod, N_C), dtype=actionsT_T.dtype)

    states_full[:k_T, :] = states_T
    actionsG_full[:k_T, :] = actionsG_T
    actionsT_full[:k_T, :] = actionsT_T

    if hold_state and k_T < n_time_rpod:
        states_full[k_T:, :] = states_T[k_T - 1, :][None, :]

    return states_full, actionsG_full, actionsT_full


# --------------------------------------------------------------------------------
# -------------------------COMPUTE
# --------------------------------------------------------------------------------
def for_computation(input):
    # Input unpacking
    current_data_index = input[0]
    other_args = input[1]
    ff_model = other_args["ff_model"]
    regions = other_args["regions"]

    # deterministic seed per index
    seed = 7 + int(current_data_index)
    rng = np.random.default_rng(seed)

    # Random init (use existing sampler, discard its target)
    init_state, _ = sample_init_target()

    # --- NEW: sample space-region and time-horizon ---
    region_id = int(rng.integers(0, len(regions)))  # 0..8
    time_id, k_T = sample_time_horizon_from_last_4_chunks(rng)  # k_T in {60,80,100}

    # --- NEW: sample goal point in chosen region w/ buffers, reject if in obstacles ---
    goal_xy = sample_goal_in_region(
        rng,
        regions[region_id],
        table_boundary_buffer=0.05,
        region_boundary_buffer=0.05,
    )

    # --- NEW: construct target_state from sampled goal position ---
    # Assumes first two state dims are x,y. Others set to 0.
    target_state = np.zeros((N_STATE,), dtype=float)
    target_state[0:2] = goal_xy

    # --- NEW: behavior mode = (time_id, region_id) combined into one label ---
    # 9 regions x 3 time bins = 27 behavior modes.
    behavior_mode = 9 * time_id + region_id

    # Keep command_id (optional) for your language mapping pipeline
    c_id = int(rng.integers(0, 100))

    # No waypoint for this generalized regime
    wp = None

    out = {
        "feasible": True,
        "states_cvx": [],
        "actions_cvx": [],
        "actions_t_cvx": [],
        "states_scp": [],
        "actions_scp": [],
        "actions_t_scp": [],
        "target_state": [],
        "dtime": [],
        "time": [],
        "behavior_mode": behavior_mode,
        "command_id": c_id,
        "waypoint": wp,
        # New labels/metadata
        "region_id": region_id,
        "time_id": time_id,
        "k_T": k_T,
        "goal_xy": goal_xy,
    }

    # Solve simplified problem (no obstacle avoidance) with chosen horizon
    traj_cvx_i, J_cvx_i, iter_cvx_i, feas_cvx_i = ocp_no_obstacle_avoidance(
        ff_model, init_state, target_state, n_time_override=k_T, waypoint=wp
    )

    if np.char.equal(feas_cvx_i, "optimal"):
        try:
            # Solve SCP with obstacles; same horizon
            traj_scp_i, J_scp_i, iter_scp_i, feas_scp_i = ocp_obstacle_avoidance(
                ff_model,
                traj_cvx_i["states"][:, : k_T + 1],
                traj_cvx_i["actions_G"][:, :k_T],
                init_state,
                target_state,
                n_time_override=k_T,
                waypoint=wp,
            )

            if np.char.equal(feas_scp_i, "optimal"):
                # Convert to time-major arrays (k_T, dim)
                states_cvx_T = np.transpose(traj_cvx_i["states"][:, :k_T])
                actions_cvx_T = np.transpose(traj_cvx_i["actions_G"][:, :k_T])
                actions_t_cvx_T = np.transpose(traj_cvx_i["actions_t"][:, :k_T])

                states_scp_T = np.transpose(traj_scp_i["states"][:, :k_T])
                actions_scp_T = np.transpose(traj_scp_i["actions_G"][:, :k_T])
                actions_t_scp_T = np.transpose(traj_scp_i["actions_t"][:, :k_T])

                # Pad to fixed horizon length n_time_rpod (100)
                states_cvx_full, actions_cvx_full, actions_t_cvx_full = pad_to_full_horizon(
                    states_cvx_T, actions_cvx_T, actions_t_cvx_T, k_T, n_time_rpod, hold_state=True
                )
                states_scp_full, actions_scp_full, actions_t_scp_full = pad_to_full_horizon(
                    states_scp_T, actions_scp_T, actions_t_scp_T, k_T, n_time_rpod, hold_state=True
                )

                out["states_cvx"] = states_cvx_full
                out["actions_cvx"] = actions_cvx_full
                out["actions_t_cvx"] = actions_t_cvx_full

                out["states_scp"] = states_scp_full
                out["actions_scp"] = actions_scp_full
                out["actions_t_scp"] = actions_t_scp_full

                out["target_state"] = target_state
                out["dtime"] = dt
                out["time"] = np.linspace(0, T, n_time_rpod, endpoint=False)
            else:
                out["feasible"] = False
        except Exception:
            out["feasible"] = False
    else:
        out["feasible"] = False

    return out


if __name__ == "__main__":
    dataset_version = "v02"
    (root_folder / f"dataset/torch/{dataset_version}").mkdir(parents=True, exist_ok=True)

    N_data = 600000
    set_start_method("spawn", force=True)

    n_S = N_STATE
    n_A = N_ACTION
    n_C = N_CLUSTERS

    # Model initialization
    ff_model = FreeflyerModel()

    # Build the 9 spatial regions once and pass to workers
    regions = build_goal_regions_3x3_xgt12(x_min=1.2, ncols=3, nrows=3)
    assert len(regions) == 9

    other_args = {
        "ff_model": ff_model,
        "regions": regions,
    }

    # Pre-allocate arrays
    states_cvx = np.empty((N_data, n_time_rpod, n_S), dtype=float)
    actions_cvx = np.empty((N_data, n_time_rpod, n_A), dtype=float)
    actions_t_cvx = np.empty((N_data, n_time_rpod, n_C), dtype=float)

    states_scp = np.empty((N_data, n_time_rpod, n_S), dtype=float)
    actions_scp = np.empty((N_data, n_time_rpod, n_A), dtype=float)
    actions_t_scp = np.empty((N_data, n_time_rpod, n_C), dtype=float)

    target_state = np.empty((N_data, n_S), dtype=float)
    dtime_arr = np.empty((N_data,), dtype=float)
    time_arr = np.empty((N_data, n_time_rpod), dtype=float)

    # Labels
    behavior_mode = np.empty((N_data,), dtype=np.int32)  # 0..26
    command_id = np.empty((N_data,), dtype=np.int32)

    region_id_arr = np.empty((N_data,), dtype=np.int32)  # 0..8
    time_id_arr = np.empty((N_data,), dtype=np.int32)    # 0..3
    k_T_arr = np.empty((N_data,), dtype=np.int32)        # 60/80/100
    goal_xy_arr = np.empty((N_data, 2), dtype=float)

    # Keep waypoint for compatibility (all None here)
    waypoint = []

    i_unfeas = []

    with Pool(processes=24) as p:
        iterator = p.imap(for_computation, zip(np.arange(N_data), itertools.repeat(other_args)))
        for i, res in enumerate(tqdm(iterator, total=N_data)):

            if res["feasible"]:
                # Store full padded trajectories (always length n_time_rpod)
                states_cvx[i] = res["states_cvx"]
                actions_cvx[i] = res["actions_cvx"]
                actions_t_cvx[i] = res["actions_t_cvx"]

                states_scp[i] = res["states_scp"]
                actions_scp[i] = res["actions_scp"]
                actions_t_scp[i] = res["actions_t_scp"]

                target_state[i] = res["target_state"]
                dtime_arr[i] = res["dtime"]
                time_arr[i] = res["time"]

                behavior_mode[i] = res["behavior_mode"]
                command_id[i] = res["command_id"]

                region_id_arr[i] = res["region_id"]
                time_id_arr[i] = res["time_id"]
                k_T_arr[i] = res["k_T"]
                goal_xy_arr[i, :] = res["goal_xy"]

                waypoint.append(res["waypoint"])
            else:
                i_unfeas.append(i)

            # periodic checkpoint
            if i % 50000 == 0 and i > 0:
                np.savez_compressed(
                    root_folder / f"dataset/dataset-ff-{dataset_version}-scp{i}",
                    states_scp=states_scp,
                    actions_scp=actions_scp,
                    actions_t_scp=actions_t_scp,
                    i_unfeas=i_unfeas,
                )
                np.savez_compressed(
                    root_folder / f"dataset/dataset-ff-{dataset_version}-cvx{i}",
                    states_cvx=states_cvx,
                    actions_cvx=actions_cvx,
                    actions_t_cvx=actions_t_cvx,
                    i_unfeas=i_unfeas,
                )
                np.savez_compressed(
                    root_folder / f"dataset/dataset-ff-{dataset_version}-param{i}",
                    target_state=target_state,
                    time=time_arr,
                    dtime=dtime_arr,
                    i_unfeas=i_unfeas,
                    behavior_mode=behavior_mode,
                    command_id=command_id,
                    region_id=region_id_arr,
                    time_id=time_id_arr,
                    k_T=k_T_arr,
                    goal_xy=goal_xy_arr,
                    waypoint=waypoint,
                )

    # Remove unfeasible data points
    if i_unfeas:
        states_cvx = np.delete(states_cvx, i_unfeas, axis=0)
        actions_cvx = np.delete(actions_cvx, i_unfeas, axis=0)
        actions_t_cvx = np.delete(actions_t_cvx, i_unfeas, axis=0)

        states_scp = np.delete(states_scp, i_unfeas, axis=0)
        actions_scp = np.delete(actions_scp, i_unfeas, axis=0)
        actions_t_scp = np.delete(actions_t_scp, i_unfeas, axis=0)

        target_state = np.delete(target_state, i_unfeas, axis=0)
        dtime_arr = np.delete(dtime_arr, i_unfeas, axis=0)
        time_arr = np.delete(time_arr, i_unfeas, axis=0)

        behavior_mode = np.delete(behavior_mode, i_unfeas, axis=0)
        command_id = np.delete(command_id, i_unfeas, axis=0)

        region_id_arr = np.delete(region_id_arr, i_unfeas, axis=0)
        time_id_arr = np.delete(time_id_arr, i_unfeas, axis=0)
        k_T_arr = np.delete(k_T_arr, i_unfeas, axis=0)
        goal_xy_arr = np.delete(goal_xy_arr, i_unfeas, axis=0)

    # Filter waypoint list (kept for compatibility)
    waypoint = [wp for idx, wp in enumerate(waypoint) if idx not in i_unfeas]

    # Save final datasets (npz)
    np.savez_compressed(
        root_folder / f"dataset/dataset-ff-{dataset_version}-scp",
        states_scp=states_scp,
        actions_scp=actions_scp,
        actions_t_scp=actions_t_scp,
    )
    np.savez_compressed(
        root_folder / f"dataset/dataset-ff-{dataset_version}-cvx",
        states_cvx=states_cvx,
        actions_cvx=actions_cvx,
        actions_t_cvx=actions_t_cvx,
    )
    np.savez_compressed(
        root_folder / f"dataset/dataset-ff-{dataset_version}-param",
        target_state=target_state,
        time=time_arr,
        dtime=dtime_arr,
        behavior_mode=behavior_mode,
        command_id=command_id,
        region_id=region_id_arr,
        time_id=time_id_arr,
        k_T=k_T_arr,
        goal_xy=goal_xy_arr,
        waypoint=waypoint,
    )

    # preprocess: save to torch/{dataset_version}/
    torch_states_scp = torch.from_numpy(states_scp)
    torch_states_cvx = torch.from_numpy(states_cvx)
    torch_actions_scp = torch.from_numpy(actions_scp)
    torch_actions_cvx = torch.from_numpy(actions_cvx)
    torch_behavior_mode = torch.from_numpy(behavior_mode)
    torch_command_id = torch.from_numpy(command_id)

    torch_region_id = torch.from_numpy(region_id_arr)
    torch_time_id = torch.from_numpy(time_id_arr)
    torch_k_T = torch.from_numpy(k_T_arr)
    torch_goal_xy = torch.from_numpy(goal_xy_arr)

    torch_rtgs_scp = torch.from_numpy(compute_reward_to_go(actions_scp))
    torch_rtgs_cvx = torch.from_numpy(compute_reward_to_go(actions_cvx))

    obs = copy.deepcopy(ff.obs)
    obs["radius"] = (obs["radius"] + ff.robot_radius) * ff.safety_margin

    torch_ctgs_scp = torch.from_numpy(compute_constraint_to_go(states_scp, obs["position"], obs["radius"]))
    torch_ctgs_cvx = torch.from_numpy(compute_constraint_to_go(states_cvx, obs["position"], obs["radius"]))

    # Save torch tensors
    torch_dir = root_folder / f"dataset/torch/{dataset_version}"
    torch.save(torch_states_scp, torch_dir / "torch_states_scp.pth")
    torch.save(torch_states_cvx, torch_dir / "torch_states_cvx.pth")
    torch.save(torch_actions_scp, torch_dir / "torch_actions_scp.pth")
    torch.save(torch_actions_cvx, torch_dir / "torch_actions_cvx.pth")
    torch.save(torch_behavior_mode, torch_dir / "torch_behavior_mode.pth")
    torch.save(torch_command_id, torch_dir / "torch_command_id.pth")
    torch.save(torch_region_id, torch_dir / "torch_region_id.pth")
    torch.save(torch_time_id, torch_dir / "torch_time_id.pth")
    torch.save(torch_k_T, torch_dir / "torch_k_T.pth")
    torch.save(torch_goal_xy, torch_dir / "torch_goal_xy.pth")
    torch.save(torch_rtgs_scp, torch_dir / "torch_rtgs_scp.pth")
    torch.save(torch_rtgs_cvx, torch_dir / "torch_rtgs_cvx.pth")
    torch.save(torch_ctgs_scp, torch_dir / "torch_ctgs_scp.pth")
    torch.save(torch_ctgs_cvx, torch_dir / "torch_ctgs_cvx.pth")

    # Permutation (kept from your original pattern)
    if states_cvx.shape[0] != states_scp.shape[0]:
        raise RuntimeError("Different dimensions of cvx and scp datasets.")
    perm = np.random.permutation(states_cvx.shape[0] * 2)
    np.save(torch_dir / "permutation.npy", perm)

    print("Completed dataset generation successfully.")
