import os
import sys
import json
from pathlib import Path
import itertools
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, set_start_method
import copy
from typing import Dict, List, Sequence, Union

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW

root_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(root_folder))

import optimization.ff_scenario as ff
from dynamics.freeflyer import FreeflyerModel, sample_init_target, ocp_no_obstacle_avoidance, ocp_obstacle_avoidance, compute_constraint_to_go, compute_reward_to_go
from optimization.ff_scenario import N_STATE, N_ACTION, N_CLUSTERS, n_time_rpod, dt, T, S, WAYPOINT_MARGIN, FAST_TIDX_RANGE, SLOW_TIDX

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
# -------------------------CASE LABELS
# --------------------------------------------------------------------------------

def load_behavior_texts(json_path = root_folder / "dataset" / "master_file.json"):
    """
    Loads the behavior-mode -> command_id -> text mapping
    into an O(1)-lookup dictionary.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Convert list-based mapping to dict-based for O(1) lookup
    mapping = {}
    for key, items in data.items():
        mapping[int(key)] = {item['command_id']: item['text'] for item in items}

    return mapping

def get_behavior_text(mapping, key, command_id):
    """
    Returns the text corresponding to (key, command_id) pair.
    """
    try:
        return mapping[key][command_id]
    except KeyError:
        raise KeyError(f"No entry for key={key}, command_id={command_id}")

# -------- batch API --------
def get_behavior_text_batch(mapping: Dict[int, Dict[int, str]], keys, command_ids) -> List[str]:
    ks   = _to_int_list(keys)
    cids = _to_int_list(command_ids)
    if len(ks) != len(cids):
        raise ValueError(f"Length mismatch: len(keys)={len(ks)} vs len(command_ids)={len(cids)}")
    out: List[str] = []
    for k, cid in zip(ks, cids):
        try:
            out.append(mapping[k][cid])
        except KeyError:
            raise KeyError(f"No entry for key={k}, command_id={cid}.")
    return out


# --------------------------------------------------------------------------------
# -------------------------HELPER FUNCTION
# --------------------------------------------------------------------------------

def build_waypoint_for_obstacle(behavior_mode, rng=None,
                                half_angle_deg=30.0,   # ± sector half-angle
                                near_margin_min=0.04,  # min extra beyond inflated radius
                                near_margin_max=0.10,  # max extra beyond inflated radius
                                max_tries=128):
    """
    Sample a waypoint in the sector BETWEEN the chosen body (left/right) and the middle obstacle:
      1) Use behavior_mode to choose body index: {0,1} -> left (idx=0); else -> right (idx=2).
      2) Sector is centered on the line from BODY -> MIDDLE, with ±30° by default.
      3) Radius is just outside the body's inflated KOZ, keeping the point close to the body.

    Clearance rules:
      - Robot center must stay within table with a robot-radius pad.
      - Must clear ALL obstacles grown by (obs_r + robot_r) * safety_margin.

    Returns: {'pos': np.array([x, y]), 'radius': 0.06, 't_index': None}
    """
    # 1) Which body (left/right) from behavior_mode
    if behavior_mode in {4, 5}:
        return None  # direct transit
    body_idx = 0 if behavior_mode in {0, 1} else 2  # left=0, right=2
    mid_idx  = 1

    rng = np.random.default_rng() if rng is None else rng

    # Shorthands
    c_all   = np.asarray(ff.obs['position'])   # (3,2)
    r_all   = np.asarray(ff.obs['radius'])     # (3,)
    robot_r = float(ff.robot_radius)           # 0.15
    gamma   = float(ff.safety_margin)          # e.g., 1.05–1.2
    table_lo = np.asarray(ff.table['xy_low'])
    table_hi = np.asarray(ff.table['xy_up'])

    c_body = c_all[body_idx]
    r_body = float(r_all[body_idx])
    c_mid  = c_all[mid_idx]

    # 2) Sector centered on ray from BODY -> MIDDLE
    v_bm = c_mid - c_body
    base_theta = np.arctan2(v_bm[1], v_bm[0])  # angle of the line (body -> middle)
    half_angle = np.deg2rad(half_angle_deg)

    # 3) Close to the body: radius just outside inflated KOZ
    R_inflated_body = (r_body + robot_r) * gamma
    dmin = max(0.0, float(near_margin_min))
    dmax = max(dmin, float(near_margin_max))

    # Precompute inflated radii for ALL obstacles for clearance checks
    inflated_all = (r_all + robot_r) * gamma

    def is_clear(p):
        # Table bounds with robot-radius padding
        pad_lo = table_lo + robot_r
        pad_hi = table_hi - robot_r
        if not (pad_lo[0] <= p[0] <= pad_hi[0] and pad_lo[1] <= p[1] <= pad_hi[1]):
            return False
        # Clearance to all inflated obstacles
        d = np.linalg.norm(p - c_all, axis=1)
        return np.all(d >= inflated_all)

    # Rejection sampling in the sector (between body and middle)
    for _ in range(max_tries):
        theta = base_theta + rng.uniform(-half_angle, +half_angle)
        dr    = rng.uniform(dmin, dmax)  # keep it close to body
        R     = R_inflated_body + dr

        pos = c_body + R * np.array([np.cos(theta), np.sin(theta)])

        if is_clear(pos):
            return {'pos': pos, 'radius': 0.06, 't_index': None}

    # Fallback: deterministic point on the centerline (body->middle), at R_inflated + median margin
    theta_fb = base_theta
    dr_fb    = 0.5 * (dmin + dmax)
    R_fb     = R_inflated_body + dr_fb
    pos_fb   = c_body + R_fb * np.array([np.cos(theta_fb), np.sin(theta_fb)])

    # Clip to table with robot-radius pad if needed
    pos_fb = np.minimum(np.maximum(pos_fb, table_lo + robot_r), table_hi - robot_r)

    # Ensure fallback is clear; if not, nudge slightly toward the body while staying ≥ inflated boundary
    if not is_clear(pos_fb):
        # Try a small angular sweep for a clear point
        for delta in np.linspace(-half_angle, half_angle, 9):
            p = c_body + R_fb * np.array([np.cos(base_theta + delta), np.sin(base_theta + delta)])
            p = np.minimum(np.maximum(p, table_lo + robot_r), table_hi - robot_r)
            if is_clear(p):
                pos_fb = p
                break

    return {'pos': pos_fb, 'radius': 0.06, 't_index': None}

def pick_terminal_index_and_wp_tidx(behavior_mode, rng):
    if behavior_mode%2 == 0:
        lo, hi = ff.FAST_TIDX_RANGE
        k_T = int(rng.integers(lo, hi + 1))
    else:
        k_T = int(ff.SLOW_TIDX)  # full horizon, e.g., 100

    k_wp = None
    if behavior_mode in {0,1,2,3}: # left or right traj
        k_wp = max(1, int(0.5 * k_T))
    return k_T, k_wp

def sample_case(rng,behavior_mapping = None):
    behavior_mode = int(rng.integers(0, 6)) # any random behaviou mode
    command_id = np.random.randint(0,100)
    if behavior_mapping is not None:  
        text = get_behavior_text(behavior_mapping, behavior_mode, command_id)
    else:
        text = ""
    return behavior_mode, text

def pad_to_full_horizon(states_T, actionsG_T, actionsT_T, k_T, n_time_rpod, hold_state=True):
    """
    Inputs:
      states_T   : (k_T, N_STATE)     -- up to terminal step (exclusive of the last appended state in your pipeline)
      actionsG_T : (k_T, N_ACTION)
      actionsT_T : (k_T, N_CLUSTERS)
      k_T        : terminal time index for the solved plan (50..70 for fast; 100 for slow)
      n_time_rpod: full horizon length (100)
      hold_state : if True, copy the terminal state to the rest of horizon

    Returns arrays of shape (n_time_rpod, *) with zeros beyond k_T and (optionally) held state.
    """
    N_S = states_T.shape[1]
    N_A = actionsG_T.shape[1]
    N_C = actionsT_T.shape[1]

    states_full   = np.zeros((n_time_rpod, N_S), dtype=states_T.dtype)
    actionsG_full = np.zeros((n_time_rpod, N_A), dtype=actionsG_T.dtype)
    actionsT_full = np.zeros((n_time_rpod, N_C), dtype=actionsT_T.dtype)

    # copy the solved segment [0..k_T-1]
    states_full[:k_T,   :] = states_T
    actionsG_full[:k_T,  :] = actionsG_T
    actionsT_full[:k_T,  :] = actionsT_T

    if hold_state and k_T < n_time_rpod:
        states_full[k_T:, :] = states_T[k_T-1, :][None, :]  # repeat terminal state
        # actions remain zeros

    return states_full, actionsG_full, actionsT_full


# --------------------------------------------------------------------------------
# -------------------------COMPUTE
# --------------------------------------------------------------------------------

def for_computation(input):
    # Input unpacking
    current_data_index = input[0]
    other_args = input[1]
    ff_model = other_args['ff_model']
    seed = 7 + current_data_index
    rng = np.random.default_rng(seed)

    # Randomic sample of initial and final conditions
    init_state, target_state = sample_init_target()

    # --- NEW: decide case / timing / waypoint ---
    behavior_mode, _ = sample_case(rng)
    k_T, k_wp = pick_terminal_index_and_wp_tidx(behavior_mode, rng)
    wp = None
    if behavior_mode in {0, 1}:
        wp = build_waypoint_for_obstacle(behavior_mode)
    elif behavior_mode in {2, 3}:
        wp = build_waypoint_for_obstacle(behavior_mode)
    if wp is not None and k_wp is not None:
        wp['t_index'] = int(k_wp)
    
    c_id = np.random.randint(0,100) # a random text command in same sematics

    # Output dictionary initialization
    out = {'feasible' : True,
           'states_cvx' : [],
           'actions_cvx' : [],
           'actions_t_cvx' : [],
           'states_scp': [],
           'actions_scp' : [],
           'actions_t_scp' : [],
           'target_state' : [],
           'dtime' : [],
           'time' : [],
           'behavior_mode': behavior_mode,
           'command_id': c_id,
           'waypoint': wp,
           }

    # Solve simplified problem -> without obstacle avoidance
    traj_cvx_i, J_cvx_i, iter_cvx_i, feas_cvx_i = ocp_no_obstacle_avoidance(ff_model, init_state, target_state, n_time_override=k_T, waypoint=wp)
    
    if np.char.equal(feas_cvx_i,'optimal'):
        try:
            # Solve SCP with obstacles; same horizon and waypoint
            traj_scp_i, J_scp_i, iter_scp_i, feas_scp_i = ocp_obstacle_avoidance(
                ff_model,
                traj_cvx_i['states'][:, :k_T+1],      # pass same length refs
                traj_cvx_i['actions_G'][:, :k_T],
                init_state, target_state,
                n_time_override=k_T,
                waypoint=wp
            )

            if np.char.equal(feas_scp_i,'optimal'):
                # (time, dim) from your code
                states_cvx_T     = np.transpose(traj_cvx_i['states'][:,:k_T])      # (k_T, N_STATE)
                actions_cvx_T    = np.transpose(traj_cvx_i['actions_G'][:,:k_T])   # (k_T, N_ACTION)
                actions_t_cvx_T  = np.transpose(traj_cvx_i['actions_t'][:,:k_T])   # (k_T, N_CLUSTERS)

                states_scp_T     = np.transpose(traj_scp_i['states'][:,:k_T])      # (k_T, N_STATE)
                actions_scp_T    = np.transpose(traj_scp_i['actions_G'][:,:k_T])   # (k_T, N_ACTION)
                actions_t_scp_T  = np.transpose(traj_scp_i['actions_t'][:,:k_T])   # (k_T, N_CLUSTERS)

    
                states_cvx_full, actions_cvx_full, actions_t_cvx_full = pad_to_full_horizon(
                    states_cvx_T, actions_cvx_T, actions_t_cvx_T, k_T, n_time_rpod, hold_state=True)

                states_scp_full, actions_scp_full, actions_t_scp_full = pad_to_full_horizon(
                    states_scp_T, actions_scp_T, actions_t_scp_T, k_T, n_time_rpod, hold_state=True)

                out['states_cvx']     = states_cvx_full
                out['actions_cvx']    = actions_cvx_full
                out['actions_t_cvx']  = actions_t_cvx_full

                out['states_scp']     = states_scp_full
                out['actions_scp']    = actions_scp_full
                out['actions_t_scp']  = actions_t_scp_full

                out['target_state']   = target_state
                out['dtime']          = dt
                out['time']           = np.linspace(0, T, n_time_rpod, endpoint=False)  # full, fixed horizon
                
            else:
                out['feasible'] = False
        except:
            out['feasible'] = False
    else:
        out['feasible'] = False

    return out

if __name__ == '__main__':

    dataset_version = 'v01'
    (root_folder / f'dataset/torch/{dataset_version}').mkdir(parents=True, exist_ok=True)
    
    N_data = 200000 #20
    set_start_method('spawn')

    n_S = N_STATE # state size
    n_A = N_ACTION # action size
    n_C = N_CLUSTERS # cluster size

    # Model initialization
    ff_model = FreeflyerModel()
    other_args = {
        'ff_model' : ff_model
    }

    states_cvx = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float) # [m,m,m,m/s,m/s,m/s]
    actions_cvx = np.empty(shape=(N_data, n_time_rpod, n_A), dtype=float) # [m/s]
    actions_t_cvx = np.empty(shape=(N_data, n_time_rpod, n_C), dtype=float)

    states_scp = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float) # [m,m,m,m/s,m/s,m/s]
    actions_scp = np.empty(shape=(N_data, n_time_rpod, n_A), dtype=float) # [m/s]
    actions_t_scp = np.empty(shape=(N_data, n_time_rpod, n_C), dtype=float)

    target_state = np.empty(shape=(N_data, n_S), dtype=float)
    dtime = np.empty(shape=(N_data, ), dtype=float)
    time = np.empty(shape=(N_data, n_time_rpod), dtype=float)

    # Create labels for behavior_mode
    behavior_mode   = np.empty(shape=(N_data,), dtype=np.int32)
    command_id = np.empty(shape=(N_data,), dtype=np.int32)

    # added waypoint info as each waypoint is dict {'pos': pos_fb, 'radius': 0.06, 't_index': None}
    waypoint = []

    i_unfeas = []

    # Pool creation --> Should automatically select the maximum number of processes
    # p = Pool(processes=24)
    with Pool(processes=24) as p:
        for i, res in enumerate(tqdm(p.imap(for_computation, zip(np.arange(N_data), itertools.repeat(other_args))), total=N_data)):
            # If the solution is feasible save the optimization output
            if res['feasible']:
                k_T = res['time'].shape[0]  # actual length for this sample

                states_cvx[i, :k_T, :]    = res['states_cvx']
                actions_cvx[i, :k_T, :]   = res['actions_cvx']
                actions_t_cvx[i, :k_T, :] = res['actions_t_cvx']

                states_scp[i, :k_T, :]    = res['states_scp']
                actions_scp[i, :k_T, :]   = res['actions_scp']
                actions_t_scp[i,:k_T, :]  = res['actions_t_scp']

                target_state[i, :] = res['target_state']
                dtime[i]           = res['dtime']
                time[i, :k_T]      = res['time']

                behavior_mode[i]       = res['behavior_mode']
                command_id[i]     = res['command_id']
                waypoint += [ res['waypoint'] ]

            # Else add the index to the list
            else:
                i_unfeas += [ i ]
            
            if i % 50000 == 0 and i > 0:
                
                np.savez_compressed(root_folder / f'dataset/dataset-ff-{dataset_version}-scp{i}', states_scp = states_scp, actions_scp = actions_scp, actions_t_scp = actions_t_scp, i_unfeas = i_unfeas)
                np.savez_compressed(root_folder / f'dataset/dataset-ff-{dataset_version}-cvx{i}' , states_cvx = states_cvx, actions_cvx = actions_cvx, actions_t_cvx = actions_t_cvx, i_unfeas = i_unfeas)
                np.savez_compressed(root_folder / f'dataset/dataset-ff-{dataset_version}-param{i}', target_state = target_state, time = time, dtime = dtime, i_unfeas = i_unfeas, behavior_mode = behavior_mode, command_id = command_id, waypoint = waypoint)

    
    # Remove unfeasible data points
    if i_unfeas:
        states_cvx = np.delete(states_cvx, i_unfeas, axis=0)
        actions_cvx = np.delete(actions_cvx, i_unfeas, axis=0)
        actions_t_cvx = np.delete(actions_t_cvx, i_unfeas, axis=0)

        states_scp = np.delete(states_scp, i_unfeas, axis=0)
        actions_scp = np.delete(actions_scp, i_unfeas, axis=0)
        actions_t_scp = np.delete(actions_t_scp, i_unfeas, axis=0)
        
        target_state = np.delete(target_state, i_unfeas, axis=0)
        dtime = np.delete(dtime, i_unfeas, axis=0)
        time = np.delete(time, i_unfeas, axis=0)
        
        behavior_mode = np.delete(behavior_mode, i_unfeas, axis=0)
        command_id = np.delete(command_id, i_unfeas, axis=0)
    
    waypoint = [wp for idx, wp in enumerate(waypoint) if idx not in i_unfeas]

    #  Save dataset (local folder for the workstation)
    np.savez_compressed(root_folder / f'dataset/dataset-ff-{dataset_version}-scp', states_scp = states_scp, actions_scp = actions_scp, actions_t_scp = actions_t_scp)
    np.savez_compressed(root_folder / f'dataset/dataset-ff-{dataset_version}-cvx', states_cvx = states_cvx, actions_cvx = actions_cvx, actions_t_cvx = actions_t_cvx)
    np.savez_compressed(root_folder / f'dataset/dataset-ff-{dataset_version}-param', target_state = target_state, time = time, dtime = dtime, behavior=behavior_mode, command_id=command_id, waypoint=waypoint)


    # preprocess the data and save it to torch/{dataset_version}/ directory
    torch_states_scp = torch.from_numpy(states_scp)
    torch_states_cvx = torch.from_numpy(states_cvx)
    torch_actions_scp = torch.from_numpy(actions_scp)
    torch_actions_cvx = torch.from_numpy(actions_cvx)
    torch_behavior_mode = torch.from_numpy(behavior_mode)
    torch_command_id = torch.from_numpy(command_id)

    torch_rtgs_scp = torch.from_numpy(compute_reward_to_go(actions_scp))
    torch_rtgs_cvx = torch.from_numpy(compute_reward_to_go(actions_cvx))

    obs = copy.deepcopy(ff.obs)
    obs['radius'] = (obs['radius'] + ff.robot_radius)*ff.safety_margin

    torch_ctgs_scp = torch.from_numpy(compute_constraint_to_go(states_scp, obs['position'], obs['radius']))
    torch_ctgs_cvx = torch.from_numpy(compute_constraint_to_go(states_cvx, obs['position'], obs['radius']))

    # save the torch data
    torch.save(torch_states_scp, root_folder / f'dataset/torch/{dataset_version}/torch_states_scp.pth')
    torch.save(torch_states_cvx, root_folder / f'dataset/torch/{dataset_version}/torch_states_cvx.pth')
    torch.save(torch_actions_scp, root_folder / f'dataset/torch/{dataset_version}/torch_actions_scp.pth')
    torch.save(torch_actions_cvx, root_folder / f'dataset/torch/{dataset_version}/torch_actions_cvx.pth')
    torch.save(torch_behavior_mode, root_folder / f'dataset/torch/{dataset_version}/torch_behavior_mode.pth')
    torch.save(torch_command_id, root_folder / f'dataset/torch/{dataset_version}/torch_command_id.pth')    
    torch.save(torch_rtgs_scp, root_folder / f'dataset/torch/{dataset_version}/torch_rtgs_scp.pth')
    torch.save(torch_rtgs_cvx, root_folder / f'dataset/torch/{dataset_version}/torch_rtgs_cvx.pth')
    torch.save(torch_ctgs_scp, root_folder / f'dataset/torch/{dataset_version}/torch_ctgs_scp.pth')
    torch.save(torch_ctgs_cvx, root_folder / f'dataset/torch/{dataset_version}/torch_ctgs_cvx.pth')

    # Permutation
    if states_cvx.shape[0] != states_scp.shape[0]:
        raise RuntimeError('Different dimensions of cvx and scp datasets.')
    perm = np.random.permutation(states_cvx.shape[0]*2)
    np.save(root_folder / f'dataset/torch/{dataset_version}/permutation.npy', perm)
    print('Completed dataset generation successfully.')
