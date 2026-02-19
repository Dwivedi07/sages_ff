"""
This script will be perofrming pre-scp and post-scp analysis for both ART models for 
purely new samples.
pre-scp compares these three variants:
    - Convex warmstarting (CVX)
    - ART-ctg-text warmstarting (DT_ctg)
    
Criteria of comparison:
    - Success rate of sematic obeying
    - saftey: ctgs evalution for the warmstarted trajectories

Post SCP analysis compares the above three variants across these criterias:

    post-scp compares these three variants:
        - CVX(with added waypoint) - SCP(feasibility problem)
        - ART-ctg-text warmstarting (DT_ctg) - SCP (feasibility problem)
        - ART-ctg-text warmstarting (DT_ctg) - SCP (Fuel Optimality problem)

    - Fuel consumtion: final const
    - Number of iteration to converge
    - Runtime to converge
"""
import os
import sys
from pathlib import Path
root_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(root_folder))

import time
import numpy as np
import numpy.linalg as la
import itertools
import torch

import itertools
from multiprocessing import Pool, set_start_method
from tqdm import tqdm

# /src/
import decision_transformer.manage as DT_manager
from dynamics.freeflyer import FreeflyerModel, ocp_no_obstacle_avoidance, ocp_obstacle_avoidance, ocp_obstacle_avoidance_feasibility, compute_constraint_to_go, sample_init_target
from optimization.ff_scenario import n_time_rpod, N_STATE, N_ACTION, obs, iter_max_SCP, robot_radius, safety_margin
from decision_transformer.adapter import FrozenTextAdapter
from dataset_generation.dataset_pargen import load_behavior_texts, get_behavior_text_batch, get_behavior_text, build_waypoint_for_obstacle, pick_terminal_index_and_wp_tidx, sample_case 

device = DT_manager.device

'''
TODO:
- (1) 
'''

def pad_traj_to_full(states, actions_G, n_time_rpod):
    """
    states:  (6, k_T+1)
    actions: (3, k_T)
    returns: states (6, n_time_rpod+1), actions (3, n_time_rpod)
    """
    cur_S = states.shape[1]   # k_T+1
    cur_A = actions_G.shape[1]  # k_T
    need_S = (n_time_rpod + 1) - cur_S
    need_A = n_time_rpod - cur_A
    if need_S > 0:
        states = np.hstack([states, np.repeat(states[:, -1][:, None], need_S, axis=1)])
    if need_A > 0:
        actions_G = np.hstack([actions_G, np.zeros((actions_G.shape[0], need_A))])
    return states, actions_G


def for_computation(input_iterable):

    # Extract input
    current_idx = input_iterable[0]
    input_dict = input_iterable[1]
    model_ctg = input_dict['model_ctg']
    model_text = input_dict['model_text']
    text_encoder_ctg = input_dict['text_encoder_ctg']
    text_encoder_text = input_dict['text_encoder_text']
    sample_init_final = input_dict['sample_init_final']
    command_mapping = input_dict['command_mapping']
    test_loader = input_dict['test_loader']
    unseen_text = input_dict['unseen_text']
    

    # Output dictionary initialization
    out = {'test_dataset_ix' : [],
           'state_init' : [],
           'state_final' : [], 
           'behavior' : [],
           'command' : [],
           'feasible_cvx' : True,
           'J_cvx' : [],
           'runtime_cvx' : [],
           'states_cvx' : [],
           'actions_cvx' : [],
           'ctgs0_cvx': [],
           'cvx_problem' : False,
           'feasible_scp_cvx' : True,
           'J_vect_scp_cvx' : [],
           'iter_scp_cvx' : [],
           'runtime_scp_cvx' : [],
           'states_scp_cvx' : [],
           'actions_scp_cvx' : [],
           'J_DT_text' : [],
           'feasible_DT_text' : True,
           'J_DT_ctg' : [],
           'ctgs0_DT' : [],
           'feasible_DT_ctg' : True,
           'feasible_DT_ctg_op' : True,
           'J_vect_scp_DT_text' : [],
           'J_vect_scp_DT_ctg' : [],
           'J_vect_scp_DT_ctg_op' : [],
           'iter_scp_DT_text' : [],
           'iter_scp_DT_ctg' : [],
           'iter_scp_DT_ctg_op' : [],
           'runtime_DT_text' : [],
           'runtime_DT_ctg' : [],
           'runtime_scp_DT_text' : [],
           'runtime_scp_DT_ctg' : [],
           'runtime_scp_DT_ctg_op' : [],
           'states_ws_DT_text' : [],
           'actions_ws_DT_text' : [],
           'states_ws_DT_ctg' : [],
           'actions_ws_DT_ctg' : [],
           'states_scp_DT_text' : [],
           'actions_scp_DT_text' : [],
           'states_scp_DT_ctg' : [],
           'actions_scp_DT_ctg' : [],
           'states_scp_DT_ctg_op' : [],
           'actions_scp_DT_ctg_op' : []
          }
   
    test_sample = test_loader.dataset.getix(current_idx)
    data_stats = test_loader.dataset.data_stats
    seed = 7 + current_idx
    rng = np.random.default_rng(seed)

    # TODO: can also use new json for test analysis
    if unseen_text:
        c_id = np.random.randint(100,120) # a random text command in same sematics
    else:
        c_id = np.random.randint(0,100) # a random text command in same sematics 

    if sample_init_final:
        # TODO: write propoer sampling function based snippet
        state_init, state_final = sample_init_target()
        # --- NEW: decide case / timing / waypoint ---
        behavior_i, _ = sample_case(rng)
        k_T, k_wp = pick_terminal_index_and_wp_tidx(behavior_i, rng)
        wp = None
        if behavior_i in {0, 1}:
            wp = build_waypoint_for_obstacle(behavior_i)
        elif behavior_i in {2, 3}:
            wp = build_waypoint_for_obstacle(behavior_i)
        if wp is not None and k_wp is not None:
            wp['t_index'] = int(k_wp)

        (states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix, behavior_i_t, command_id_i_t) = test_sample

        test_sample[0][0,:,:] = (torch.tensor(np.repeat(state_init[None,:], n_time_rpod, axis=0)) - data_stats['states_mean'])/(data_stats['states_std'] + 1e-6)
        test_sample[1][0,:,:] = torch.zeros((n_time_rpod, N_ACTION))
        test_sample[2][0,:,0] = torch.zeros((n_time_rpod,))
        test_sample[3][:,0] = torch.zeros((n_time_rpod,))
        behavior_i_t      = behavior_i
        command_id_i_t    = c_id
        out['test_dataset_ix'] = test_sample[-3][0]
        test_sample = (states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix, behavior_i_t, command_id_i_t)
    else:
        states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix, behavior_i, command_id_i = test_sample
        out['test_dataset_ix'] = ix[0]
        state_init = np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
        state_final = np.array((goal_i[0, 0, :] * data_stats['goal_std'][0]) + data_stats['goal_mean'][0])


    out['behavior'] = behavior_i
    out['command'] = get_behavior_text(command_mapping, behavior_i, c_id)

    out['state_init'] = state_init
    out['state_final'] = state_final

    ff_model = FreeflyerModel()
    ##################################################################
    ####### Warmstart Convex Problem
    ##################################################################
    try:
        runtime0_cvx = time.time()
        traj_cvx, _, _, feas_cvx  = ocp_no_obstacle_avoidance(ff_model, state_init, state_final, n_time_override=k_T, waypoint=wp)
        runtime1_cvx = time.time()
        runtime_cvx = runtime1_cvx-runtime0_cvx
        states_cvx, actions_cvx = traj_cvx['states'], traj_cvx['actions_G']
    except:
        states_cvx = None
        actions_cvx = None
        feas_cvx = 'infeasible'
        runtime_cvx = None

    if np.char.equal(feas_cvx,'infeasible'):
        out['feasible_scp_cvx'] = False
        out['feasible_cvx'] = False
        out['states_scp_cvx'] = None
        out['actions_scp_cvx'] = None
    else:
        # pad the traj to always be shape [6,101] and [3, 100]
        states_cvx_full, actions_cvx_full = pad_traj_to_full(states_cvx, actions_cvx, n_time_rpod)
        out['states_cvx'] = states_cvx_full
        out['actions_cvx'] = actions_cvx_full
        states_ws_cvx = states_cvx # set warm start
        actions_ws_cvx = actions_cvx # set warm start
        out['J_cvx'] = np.sum(la.norm(actions_ws_cvx, ord=1, axis=0))
        # Evaluate Constraint Violation
        ctgs_cvx = compute_constraint_to_go(states_ws_cvx.T, obs['position'], (obs['radius'] + robot_radius)*safety_margin)
        ctgs0_cvx = ctgs_cvx[0,0]
        # Save cvx in the output dictionary
        out['runtime_cvx'] = runtime_cvx
        out['ctgs0_cvx'] = ctgs0_cvx
        out['cvx_problem'] = ctgs0_cvx == 0

        # Solve SCP Feasibility Problem
        runtime0_scp_cvx = time.time()
        traj_scp_cvx, J_vect_scp_cvx, iter_scp_cvx, feas_scp_cvx = ocp_obstacle_avoidance_feasibility(ff_model, states_ws_cvx, actions_ws_cvx, state_init, state_final) # feasibility problem
        # traj_scp_cvx, J_vect_scp_cvx, iter_scp_cvx, feas_scp_cvx = ocp_obstacle_avoidance(ff_model, states_ws_cvx, actions_ws_cvx, state_init, state_final, n_time_override=k_T, waypoint=wp) # expert solution
        runtime1_scp_cvx = time.time()
        runtime_scp_cvx = runtime1_scp_cvx - runtime0_scp_cvx
        
        if np.char.equal(feas_scp_cvx,'infeasible'):
            out['feasible_scp_cvx'] = False
        else:
            # Save scp_cvx data in the output dictionary
            states_scp_full, actions_scp_full = pad_traj_to_full(traj_scp_cvx['states'], traj_scp_cvx['actions_G'], n_time_rpod)
            out['J_vect_scp_cvx'] = J_vect_scp_cvx
            out['iter_scp_cvx'] = iter_scp_cvx    
            out['runtime_scp_cvx'] = runtime_scp_cvx
            out['states_scp_cvx'] = states_scp_full
            out['actions_scp_cvx'] = actions_scp_full

    ##################################################################
    ####### Warmstart ART text only
    ##################################################################
    # DT_text_trajectory, runtime_DT_text = DT_manager.torch_model_inference_dyn(model_text, test_loader, test_sample, text_encoder_text, command_mapping, ctg_condition=False)                        
    # out['J_DT_text'] = np.sum(la.norm(DT_text_trajectory['dv_dyn'], ord=1, axis=0))
    # states_ws_DT_text = np.hstack((DT_text_trajectory['xypsi_dyn'], state_final.reshape(-1,1))) # set warm start
    # actions_ws_DT_text = DT_text_trajectory['dv_dyn'] # set warm start
    # # Save DT in the output dictionary
    # out['runtime_DT_text'] = runtime_DT_text
    # out['actions_ws_DT_text'] = actions_ws_DT_text
    # out['states_ws_DT_text'] = states_ws_DT_text # clip the states to be of shape [6,100] not [6,101]

    # # Solve SCP
    # runtime0_scp_DT_text = time.time()
    # traj_scp_DT_text, J_vect_scp_DT_text, iter_scp_DT_text, feas_scp_DT_text = ocp_obstacle_avoidance_feasibility(ff_model, states_ws_DT_text, actions_ws_DT_text, state_init, state_final)
    # runtime1_scp_DT_text = time.time()
    # runtime_scp_DT_text = runtime1_scp_DT_text - runtime0_scp_DT_text

    # if np.char.equal(feas_scp_DT_text,'infeasible'):
    #     out['feasible_DT_text'] = False
    # else:
    #     # Save scp_DT_text data in the output dictionary
    #     out['J_vect_scp_DT_text'] = J_vect_scp_DT_text
    #     out['iter_scp_DT_text'] = iter_scp_DT_text    
    #     out['runtime_scp_DT_text'] = runtime_scp_DT_text
    #     out['states_scp_DT_text'] = traj_scp_DT_text['states']
    #     out['actions_scp_DT_text'] = traj_scp_DT_text['actions_G']
    
    ##################################################################
    ####### Warmstart ART ctg + text
    ##################################################################
    DT_ctg_trajectory, runtime_DT_ctg = DT_manager.torch_model_inference_dyn(model_ctg, test_loader, test_sample, text_encoder_ctg, command_mapping, ctg_condition=True)
    out['J_DT_ctg'] = np.sum(la.norm(DT_ctg_trajectory['dv_dyn'], ord=1, axis=0))
    states_ws_DT_ctg = np.hstack((DT_ctg_trajectory['xypsi_dyn'], state_final.reshape(-1,1))) # set warm start
    actions_ws_DT_ctg = DT_ctg_trajectory['dv_dyn'] # set warm start

    # Evaluate Constraint Violation
    ctgs_DT = compute_constraint_to_go(states_ws_DT_ctg.T, obs['position'], (obs['radius'] + robot_radius)*safety_margin)
    ctgs0_DT = ctgs_DT[0,0]
    # Save cvx in the output dictionary
    out['ctgs0_DT'] = ctgs0_DT

    # Save DT in the output dictionary
    out['runtime_DT_ctg'] = runtime_DT_ctg
    out['actions_ws_DT_ctg'] = actions_ws_DT_ctg
    out['states_ws_DT_ctg'] = states_ws_DT_ctg # clip the states to be of shape [6,100] not [6,101]

    # Solve SCP Feasibility Problem
    runtime0_scp_DT_ctg = time.time()
    traj_scp_DT_ctg, J_vect_scp_DT_ctg, iter_scp_DT_ctg, feas_scp_DT_ctg = ocp_obstacle_avoidance_feasibility(ff_model, states_ws_DT_ctg, actions_ws_DT_ctg, state_init, state_final)
    runtime1_scp_DT_ctg = time.time()
    runtime_scp_DT_ctg = runtime1_scp_DT_ctg - runtime0_scp_DT_ctg  
    if np.char.equal(feas_scp_DT_ctg,'infeasible'):
        out['feasible_DT_ctg'] = False
    else:
        # Save scp_DT_ctg data in the output dictionary
        out['J_vect_scp_DT_ctg'] = J_vect_scp_DT_ctg
        out['iter_scp_DT_ctg'] = iter_scp_DT_ctg    
        out['runtime_scp_DT_ctg'] = runtime_scp_DT_ctg
        out['states_scp_DT_ctg'] = traj_scp_DT_ctg['states']
        out['actions_scp_DT_ctg'] = traj_scp_DT_ctg['actions_G']
    
    # Solve SCP Optimality Probem without Added Waypoint
    runtime0_scp_DT_ctg_op = time.time()
    traj_scp_DT_ctg_op, J_vect_scp_DT_ctg_op, iter_scp_DT_ctg_op, feas_scp_DT_ctg_op = ocp_obstacle_avoidance_feasibility(ff_model, states_ws_DT_ctg, actions_ws_DT_ctg, state_init, state_final, w_tracking=0.0)
    runtime1_scp_DT_ctg_op = time.time()
    runtime_scp_DT_ctg_op = runtime1_scp_DT_ctg_op - runtime0_scp_DT_ctg_op  

    if np.char.equal(feas_scp_DT_ctg_op,'infeasible'):
        out['feasible_DT_ctg_op'] = False
    else:
        # Save scp_DT_ctg data in the output dictionary
        out['J_vect_scp_DT_ctg_op'] = J_vect_scp_DT_ctg_op
        out['iter_scp_DT_ctg_op'] = iter_scp_DT_ctg_op    
        out['runtime_scp_DT_ctg_op'] = runtime_scp_DT_ctg_op
        out['states_scp_DT_ctg_op'] = traj_scp_DT_ctg_op['states']
        out['actions_scp_DT_ctg_op'] = traj_scp_DT_ctg_op['actions_G']

    return out

if __name__ == '__main__':
    ws_version = 'v_06_80k_seen'  # warmstarting analysis version #'v_01_total_seen_fo'. #
    model_version_ctg = 'v_06' # v_03 for SCP+CVX IL with ctg; v_06 trained on 80k for 2e6 steps; v_07 trained on 200k for 2e6 steps
    model_version_text = 'v_04' # v_04 for SCP IL
    dataset_version = 'v01'
    num_processes = 4
    N_data = 1000
    unseen_text = False

    import_config = DT_manager.transformer_import_config(model_version_ctg)
    ctg_condition = import_config["ctg_condition"]
    timestep_norm = import_config["timestep_norm"]
    dataset_to_use = import_config["dataset_to_use"] 
    set_start_method('spawn')

    # Get the datasets and loaders from the torch data
    datasets, dataloaders = DT_manager.get_train_val_test_data( ctg_condition=ctg_condition, timestep_norm=timestep_norm, dataset_to_use=dataset_to_use, dataset_version=dataset_version)
    train_loader, eval_loader, test_loader = dataloaders

    # Get both ART models
    model_ctg = DT_manager.get_DT_model(model_version_ctg, train_loader, eval_loader, ctg_condition = True)
    model_text = DT_manager.get_DT_model(model_version_text, train_loader, eval_loader, ctg_condition = False)

    # load the text encoders for each models (and its weight)  
    MODEL = os.getenv("FTA_MODEL", "distilbert-base-uncased")
    text_encoder_ctg = FrozenTextAdapter(model_name=MODEL, out_dim=model_ctg.hidden_size, output_mode="tokens").to(device).eval()
    text_encoder_ctg.load_adapter(root_folder / "decision_transformer" / "saved_files" / "checkpoints" / f"{model_version_ctg}" / "text_adapter.pth") 

    text_encoder_text = FrozenTextAdapter(model_name=MODEL, out_dim=model_text.hidden_size, output_mode="tokens").to(device).eval()
    text_encoder_text.load_adapter(root_folder / "decision_transformer" / "saved_files" / "checkpoints" / f"{model_version_text}" / "text_adapter.pth") 
    
    # load the command mapping (master file) 
    command_mapping = load_behavior_texts(root_folder / "dataset" / "master_file_gen_me.json") # master_file_test has 100 +  30 unseen commands

    # Parallel for inputs
    N_data_test = np.min([test_loader.dataset.n_data, N_data]) 
    other_args = {
        'model_ctg' : model_ctg,
        'model_text' : model_text,
        'text_encoder_ctg' : text_encoder_ctg,
        'text_encoder_text' : text_encoder_text,
        'test_loader' : test_loader,
        'sample_init_final' : True,
        'command_mapping' : command_mapping,
        'unseen_text' : unseen_text,
    } 

    test_dataset_ix = np.empty(shape=(N_data_test, ), dtype=float)
    state_init = np.empty(shape=(N_data_test, 6), dtype=float)
    state_final = np.empty(shape=(N_data_test, 6), dtype=float)
    behavior = np.empty(shape=(N_data_test, ), dtype=float)
    command = np.empty(shape=(N_data_test, ), dtype=str)

    J_vect_scp_cvx = np.ones(shape=(N_data_test, iter_max_SCP), dtype=float)* 1e12
    J_vect_scp_DT_text = np.ones(shape=(N_data_test, iter_max_SCP), dtype=float)* 1e12
    J_vect_scp_DT_ctg = np.ones(shape=(N_data_test, iter_max_SCP), dtype=float)* 1e12
    J_vect_scp_DT_ctg_op = np.ones(shape=(N_data_test, iter_max_SCP), dtype=float)* 1e12

    J_cvx = np.ones(shape=(N_data_test, ), dtype=float) * 1e12           # converged cost of cvx warmstart
    J_DT_text = np.ones(shape=(N_data_test, ), dtype=float)  * 1e12    # converged cost of ART text only warmstart
    J_DT_ctg = np.ones(shape=(N_data_test, ), dtype=float) * 1e12      # converged cost of ART ctg+text warmstart

    iter_scp_cvx = np.empty(shape=(N_data_test, ), dtype=float)     # iterations to converge normal scp soln
    iter_scp_DT_text = np.empty(shape=(N_data_test, ), dtype=float)      # iterations to converge ART text + SCP soln
    iter_scp_DT_ctg = np.empty(shape=(N_data_test, ), dtype=float)     # iterations to converge ART ctg+text + SCP soln
    iter_scp_DT_ctg_op = np.empty(shape=(N_data_test, ), dtype=float)     # iterations to converge ART ctg+text + SCP soln

    runtime_cvx = np.empty(shape=(N_data_test, ), dtype=float)       # runtime to give the cvx warmstart soln
    runtime_DT_text = np.empty(shape=(N_data_test, ), dtype=float)
    runtime_DT_ctg = np.empty(shape=(N_data_test, ), dtype=float) 

    runtime_scp_cvx = np.empty(shape=(N_data_test, ), dtype=float)  # total runtime for final soln
    runtime_scp_DT_text = np.empty(shape=(N_data_test, ), dtype=float) 
    runtime_scp_DT_ctg = np.empty(shape=(N_data_test, ), dtype=float) 
    runtime_scp_DT_ctg_op = np.empty(shape=(N_data_test, ), dtype=float) 

    ctgs0_cvx = np.empty(shape=(N_data_test, ), dtype=float)           # cvx solns ctgs to go at t=0
    ctgs0_DT = np.empty(shape=(N_data_test, ), dtype=float)           # cvx solns ctgs to go at t=0
    cvx_problem = np.full(shape=(N_data_test, ), fill_value=False)

    states_cvx = np.empty(shape=(N_data_test, 6, n_time_rpod+1), dtype=float)   # warmstart soln
    actions_cvx = np.empty(shape=(N_data_test, 3, n_time_rpod), dtype=float)
    states_ws_DT_text = np.empty(shape=(N_data_test, 6, n_time_rpod+1), dtype=float) 
    actions_ws_DT_text = np.empty(shape=(N_data_test, 3, n_time_rpod), dtype=float)
    states_ws_DT_ctg = np.empty(shape=(N_data_test, 6, n_time_rpod+1), dtype=float)
    actions_ws_DT_ctg = np.empty(shape=(N_data_test, 3, n_time_rpod), dtype=float)

    states_scp_cvx = np.empty(shape=(N_data_test, 6, n_time_rpod+1), dtype=float)  # final traj soln
    actions_scp_cvx = np.empty(shape=(N_data_test, 3, n_time_rpod), dtype=float)
    states_scp_DT_text = np.empty(shape=(N_data_test, 6, n_time_rpod+1), dtype=float)  
    actions_scp_DT_text = np.empty(shape=(N_data_test, 3, n_time_rpod), dtype=float)
    states_scp_DT_ctg = np.empty(shape=(N_data_test, 6, n_time_rpod+1), dtype=float)
    actions_scp_DT_ctg = np.empty(shape=(N_data_test, 3, n_time_rpod), dtype=float)
    states_scp_DT_ctg_op = np.empty(shape=(N_data_test, 6, n_time_rpod+1), dtype=float)
    actions_scp_DT_ctg_op = np.empty(shape=(N_data_test, 3, n_time_rpod), dtype=float)

    i_unfeas_cvx = []
    i_unfeas_scp_cvx = []
    i_unfeas_DT_text = []
    i_unfeas_DT_ctg = []
    i_unfeas_DT_ctg_op = []

    # Pool creation --> Should automatically select the maximum number of processes
    p = Pool(processes=num_processes)
    for i, res in enumerate(tqdm(p.imap(for_computation, zip(np.arange(N_data_test), itertools.repeat(other_args))), total=N_data_test)):
    # for i, res in enumerate(tqdm(map(for_computation, zip(np.arange(N_data_test), itertools.repeat(other_args))), total=N_data_test)):
        # Save the input in the dataset
        test_dataset_ix[i] = res['test_dataset_ix']
        state_init[i] = res['state_init']
        state_final[i] = res['state_final']
        behavior[i] = res['behavior']
        command[i] = res['command']

        ##### CVX output
        # If the solution is feasible save the optimization output
        if res['feasible_cvx']:
            J_cvx[i] = res['J_cvx']
            runtime_cvx[i] = res['runtime_cvx']
            states_cvx[i] = res['states_cvx']
            actions_cvx[i] = res['actions_cvx']

            ctgs0_cvx[i] = res['ctgs0_cvx']
            cvx_problem[i] = res['cvx_problem']
        else:
            i_unfeas_cvx += [ i ]

        if res['feasible_scp_cvx']:
            J_vect_scp_cvx[i,:] = res['J_vect_scp_cvx']
            iter_scp_cvx[i] = res['iter_scp_cvx']
            runtime_scp_cvx[i] = res['runtime_scp_cvx']
            states_scp_cvx[i] = res['states_scp_cvx']
            actions_scp_cvx[i] = res['actions_scp_cvx']
        else:
            i_unfeas_scp_cvx += [ i ]

        ###### DT_text output

        # if res['feasible_DT_text']:
        #     J_DT_text[i] = res['J_DT_text']
        #     J_vect_scp_DT_text[i,:] = res['J_vect_scp_DT_text']
        #     iter_scp_DT_text[i] = res['iter_scp_DT_text']
        #     runtime_DT_text[i] = res['runtime_DT_text']
        #     runtime_scp_DT_text[i] = res['runtime_scp_DT_text']
        #     states_ws_DT_text[i] = res['states_ws_DT_text']
        #     actions_ws_DT_text[i] = res['actions_ws_DT_text']
        #     states_scp_DT_text[i] = res['states_scp_DT_text']
        #     actions_scp_DT_text[i] = res['actions_scp_DT_text']
        # else:
        #     i_unfeas_DT_text += [ i ]

        ###### DT_ctg output
        J_DT_ctg[i] = res['J_DT_ctg']
        runtime_DT_ctg[i] = res['runtime_DT_ctg']
        states_ws_DT_ctg[i] = res['states_ws_DT_ctg']
        actions_ws_DT_ctg[i] = res['actions_ws_DT_ctg']

        ctgs0_DT[i] = res['ctgs0_DT']

        if res['feasible_DT_ctg']:
            J_vect_scp_DT_ctg[i,:] = res['J_vect_scp_DT_ctg']
            iter_scp_DT_ctg[i] = res['iter_scp_DT_ctg']
            runtime_scp_DT_ctg[i] = res['runtime_scp_DT_ctg']
            states_scp_DT_ctg[i] = res['states_scp_DT_ctg']
            actions_scp_DT_ctg[i] = res['actions_scp_DT_ctg']
        else:
            i_unfeas_DT_ctg += [ i ]
        
        if res['feasible_DT_ctg_op']:
            J_vect_scp_DT_ctg_op[i,:] = res['J_vect_scp_DT_ctg_op']
            iter_scp_DT_ctg_op[i] = res['iter_scp_DT_ctg_op']
            runtime_scp_DT_ctg_op[i] = res['runtime_scp_DT_ctg_op']
            states_scp_DT_ctg_op[i] = res['states_scp_DT_ctg_op']
            actions_scp_DT_ctg_op[i] = res['actions_scp_DT_ctg_op']
        else:
            i_unfeas_DT_ctg_op += [ i ]

    

    i_unfeas_cvx     = np.array(i_unfeas_cvx, dtype=int)
    i_unfeas_scp_cvx = np.array(i_unfeas_scp_cvx, dtype=int)
    i_unfeas_DT_text = np.array(i_unfeas_DT_text, dtype=int)
    i_unfeas_DT_ctg  = np.array(i_unfeas_DT_ctg, dtype=int)
    i_unfeas_DT_ctg_op  = np.array(i_unfeas_DT_ctg_op, dtype=int)

    #  Save dataset (local folder for the workstation)
    np.savez_compressed(
        root_folder / "optimization" / "saved_files" / "warmstarting" / f"ws_analysis_{ws_version}.npz",
        J_vect_scp_cvx=J_vect_scp_cvx,
        J_vect_scp_DT_text=J_vect_scp_DT_text,
        J_vect_scp_DT_ctg=J_vect_scp_DT_ctg,
        J_vect_scp_DT_ctg_op = J_vect_scp_DT_ctg_op,
        J_cvx=J_cvx,
        J_DT_text=J_DT_text,
        J_DT_ctg=J_DT_ctg,
        iter_scp_cvx=iter_scp_cvx,
        iter_scp_DT_text=iter_scp_DT_text,
        iter_scp_DT_ctg=iter_scp_DT_ctg,
        iter_scp_DT_ctg_op =iter_scp_DT_ctg_op,
        runtime_cvx=runtime_cvx,
        runtime_DT_text=runtime_DT_text,
        runtime_DT_ctg=runtime_DT_ctg,
        runtime_scp_cvx=runtime_scp_cvx,
        runtime_scp_DT_text=runtime_scp_DT_text,
        runtime_scp_DT_ctg=runtime_scp_DT_ctg,
        runtime_scp_DT_ctg_op = runtime_scp_DT_ctg_op,
        ctgs0_cvx=ctgs0_cvx,
        cvx_problem=cvx_problem,
        states_cvx=states_cvx,
        actions_cvx=actions_cvx,
        states_ws_DT_text=states_ws_DT_text,
        actions_ws_DT_text=actions_ws_DT_text,
        states_ws_DT_ctg=states_ws_DT_ctg,
        actions_ws_DT_ctg=actions_ws_DT_ctg,
        states_scp_cvx=states_scp_cvx,
        actions_scp_cvx=actions_scp_cvx,
        states_scp_DT_text=states_scp_DT_text,
        actions_scp_DT_text=actions_scp_DT_text,
        states_scp_DT_ctg=states_scp_DT_ctg,
        actions_scp_DT_ctg=actions_scp_DT_ctg,
        states_scp_DT_ctg_op = states_scp_DT_ctg_op,
        actions_scp_DT_ctg_op = actions_scp_DT_ctg_op,
        test_dataset_ix=test_dataset_ix,
        state_init=state_init,
        state_final=state_final,
        behavior=behavior,
        command=command,
        i_unfeas_cvx=i_unfeas_cvx,
        i_unfeas_scp_cvx=i_unfeas_scp_cvx,
        i_unfeas_DT_text=i_unfeas_DT_text,
        i_unfeas_DT_ctg=i_unfeas_DT_ctg,
        i_unfeas_DT_ctg_op=i_unfeas_DT_ctg_op
        )