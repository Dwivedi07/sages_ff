"""
Currently the CVX solution is very similar to SCP because collision is avoided by a careful selection of the waypoint.
So as a sanity check, the script only compared the CVX and ART solutions. 
SCP extension can be done by uncommenting the relevant parts.
"""
import os
import sys
from pathlib import Path
root_folder = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_folder))
import numpy as np
import itertools

from multiprocessing import Pool, set_start_method
from tqdm import tqdm
import time
import json 

# /src/
import rpod.decision_transformer.manage as DT_manager
from rpod.decision_transformer.adapter import FrozenTextAdapter
device = DT_manager.device
from rpod.dynamics.dynamics_trans import cim_roe, roe_to_rtn, mtx_roe_to_rtn
from rpod.optimization.parameters import * 
from rpod.optimization.optimization import * 
from rpod.optimization.scvx import solve_scvx
from rpod.dataset_generation.traj_classifier import eval_semantics_traj

_worker_model = None
_worker_text_encoder = None
_worker_data_stats = None
_worker_test_loader = None
_worker_command_mapping = None
_worker_scp = None
_worker_tailored_command = None

def init_worker(model_name, model_state, text_state, data_stats, 
                test_loader, command_mapping, scp, tailored_command, max_tokens):
    global _worker_model, _worker_text_encoder, _worker_data_stats
    global _worker_test_loader, _worker_command_mapping
    global _worker_scp, _worker_tailored_command
    
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild model architecture
    m = DT_manager.get_DT_model(model_name, test_loader, test_loader)
    m.load_state_dict(model_state)
    m.to(device).eval()

    # Rebuild text encoder
    MODEL = os.getenv("FTA_MODEL", "distilbert-base-uncased")
    te = FrozenTextAdapter(
        model_name=MODEL,
        out_dim=m.hidden_size,
        output_mode="tokens",
        max_tokens=max_tokens,
    ).to(device).eval()
    te.load_state_dict(text_state)

    _worker_model = m
    _worker_text_encoder = te
    _worker_data_stats = data_stats

    _worker_test_loader = test_loader
    _worker_command_mapping = command_mapping
    _worker_scp = scp
    _worker_tailored_command = tailored_command
    
def for_computation(current_idx: int):
    global _worker_model, _worker_text_encoder, _worker_data_stats
    global _worker_test_loader, _worker_command_mapping
    global _worker_scp, _worker_tailored_command

    model           = _worker_model
    text_encoder    = _worker_text_encoder
    data_stats      = _worker_data_stats
    test_loader     = _worker_test_loader
    command_mapping = _worker_command_mapping
    scp             = _worker_scp
    tailored_command= _worker_tailored_command
    
    # Output dictionary initialization
    out = {'feasible_cvx' : False,
           'feasible_scp_cvx' : False,
           'feasible_scp_DT' : False,
           'feasible_scp_DT_fuel' : False,
           'feasible_scp_DT_r' : False,
           'J_cvx' : [],
           'J_DT' : [],
           'iter_scp_cvx': [],
           'iter_scp_DT': [],
           'runtime_cvx': [],
           'runtime_DT': [],
           'runtime_scp_cvx': [],
           'runtime_scp_DT': [],
           'ctgs0_cvx': [],
           'ctgs0_DT': [],
           'cvx_problem' : False,
           'state_init' : [],
           'state_final' : [], 
           'roe_cvx' : [],
           'roe_DT' : [], 
           'rtn_cvx' : [],
           'rtn_DT' : [], 
           'behavior' : [],
          }
    
    # Reproducible, order-independent RNG for this task
    base_seed = 12345
    ss = np.random.SeedSequence([base_seed, current_idx])
    rng = np.random.default_rng(np.random.PCG64(ss))

    idx = rng.integers(0, len(test_loader.dataset))
    test_sample = test_loader.dataset.getix(idx)
    
    # generate the text command corresponding to the behavior
    # command_text = command_mapping[behavior_i]['description'][np.random.randint(0, len(command_mapping[behavior_i]['description']))]
    if not tailored_command:
        states_i, actions_i, ctgs_i, timesteps_i, attention_mask_i, oe_i, dt, time_sec, horizons, ix, behavior_i, command_id_i = test_sample
        text_command = command_mapping[behavior_i]['description'][command_id_i]
    else:
        states_i, actions_i, ctgs_i, timesteps_i, attention_mask_i, oe_i, dt, time_sec, horizons, ix, behavior_i, command_id_i, text_command, wyp, wyp_times = test_sample
        if command_id_i >= len(command_mapping[behavior_i]['templates']):
            command_id_i = np.mod(command_id_i, len(command_mapping[behavior_i]['templates']))
        template_i = command_mapping[behavior_i]['templates'][command_id_i]
   
    # purely random sampling. Not using (testing) dataset_loader here because 
    # the saved data has a bias (only SCP-converged data are saved)
    behavior_i, roe_0, roe_f, t_idx_wyp, wyp = sample_reset_condition2(behavior=behavior_i, rng=rng)   
    
    # print('Sampled trajectory ' + str(ix) + ' from test_dataset.')
    out['behavior'] = behavior_i

    out['state_init'] = roe_0
    out['state_final'] = roe_f   # NOTE: we actually need to load the goal state from data, but also for ART it does not matter 
    out['target_state'] = roe_f
    out['dtime'] = dt_sec
    out['time'] = tvec_sec
    out['oe'] = oec

    current_obs = {'state' : roe_0, 'goal' : roe_f, 'ttg' : tf_sec, 'dt' : dt_sec, 'oe' : oec0}
    chance = False
    ct = True 
    
    # Do NOT load waypoiints for ART
    prob = NonConvexOCP(
        prob_definition={
            't_i' : 0,
            't_f' : n_time,
            'tvec_sec' : tvec_sec,
            'current_obs' : current_obs,
            'chance' : chance,
            'ct' : ct,
            'behavior' : behavior_i
        }
    )
    
    cim_f = cim_roe(oec[-1])
    psi_f = mtx_roe_to_rtn(oec[-1])
        
    ###########################################
    ######## Transformer Warmstarting  ########
    ###########################################

    # Perform inference
    data_stats = test_loader.dataset.data_stats
    out_DT = DT_manager.torch_model_inference_dyn(model, data_stats, text_encoder, prob, text_command=text_command, include_final=True)
    eval_dict = eval_semantics_traj(behavior_i, text_command, template_i, out_DT['traj'], verbose=False)
    out['eval_dict_DT'] = eval_dict

    traj_DT, runtime_DT = out_DT["traj"], out_DT["runtime"]
    out['J_DT'] = np.sum(np.linalg.norm(traj_DT['dv'], axis=1))
    out['runtime_DT'] = runtime_DT

    out['roe_DT'] = traj_DT['roe']  # (n_time+1, 6)
    out['rtn_DT'] = traj_DT['rtn']  # (n_time+1, 6)
    out['a_DT'] = traj_DT['dv']     # (n_time, 3)

    ctg_DT, _  = prob.compute_ctg(traj_DT['roe'][:-1], traj_DT['dv'], prob.tvec_sec, chance=chance, ct=ct)
    out['ctgs0_DT'] = ctg_DT[0, 0]

    # update the terminal state to the ART output
    prob.state_final = traj_DT['roe'][-1]  
    # Solve SCP
    if scp: 
        #### PART 1: feasibility problem ####
        # prob.verbose_scvx = True
        prob.zref = {'state':  traj_DT['roe'][:-1], 'action':  traj_DT['dv']}
        prob.sol_0 = {"z": prob.zref}
        prob.type = "feasibility"   # switch the objective to feasibility problem
        prob._cvx_built_AL = False; prob.update_flag = True
        prob.generate_scaling(traj_DT['roe'][:-1], traj_DT['dv'])
        t0 = time.time()
        sol_scp, log_scp = solve_scvx(prob)
        runtime_scp_DT = time.time() - t0
        
        if sol_scp['status'] in ['optimal', 'optimal_inaccurate']:
            # Save scp_DT in the output dictionary
            out['feasible_scp_DT'] = True
            out['J_scp_DT'] = np.sum(np.linalg.norm(sol_scp['z']['action'], axis=1))
            out['iter_scp_DT'] = len(log_scp["f0"])
            out['runtime_scp_DT'] = runtime_scp_DT
            
            roef_scp = sol_scp['z']['state'][-1] + cim_f @ sol_scp['z']['action'][-1]
            out['roe_scp_DT'] = np.concatenate((sol_scp['z']['state'], roef_scp.reshape(1, -1)), axis=0)
            out['rtn_scp_DT'] = np.concatenate((sol_scp['z']['state'], (psi_f @ roef_scp).reshape(1, -1)), axis=0)
            out['a_scp_DT'] = sol_scp['z']['action']
            
            traj = {"roe": out['roe_scp_DT']}
            eval_dict = eval_semantics_traj(behavior_i, text_command, template_i, traj, verbose=False)
            out['eval_dict_scp_DT'] = eval_dict
        
        #### PART 2: fuel-optimal problem ####
        prob.zref = {'state':  traj_DT['roe'][:-1], 'action':  traj_DT['dv']}
        prob.sol_0 = {"z": prob.zref}
        prob.type = "min_fuel"   # switch the objective to feasibility problem
        prob._cvx_built_AL = False; prob.update_flag = True
        prob.generate_scaling(traj_DT['roe'][:-1], traj_DT['dv'])
        t0 = time.time()
        sol_scp_fuel, log_scp_fuel = solve_scvx(prob)
        runtime_scp_DT = time.time() - t0
        
        if sol_scp_fuel['status'] in ['optimal', 'optimal_inaccurate']:
            # Save scp_DT in the output dictionary
            out['feasible_scp_DT_fuel'] = True
            out['J_scp_DT_fuel'] = np.sum(np.linalg.norm(sol_scp_fuel['z']['action'], axis=1))
            out['iter_scp_DT_fuel'] = len(log_scp_fuel["f0"])
            out['runtime_scp_DT_fuel'] = runtime_scp_DT

            roef_scp = sol_scp_fuel['z']['state'][-1] + cim_f @ sol_scp_fuel['z']['action'][-1]
            out['roe_scp_DT_fuel'] = np.concatenate((sol_scp_fuel['z']['state'], roef_scp.reshape(1, -1)), axis=0)
            out['rtn_scp_DT_fuel'] = np.concatenate((sol_scp_fuel['z']['state'], (psi_f @ roef_scp).reshape(1, -1)), axis=0)
            out['a_scp_DT_fuel'] = sol_scp_fuel['z']['action']

            traj = {"roe": out['roe_scp_DT_fuel']}
            eval_dict = eval_semantics_traj(behavior_i, text_command, template_i, traj, verbose=False)
            out['eval_dict_scp_DT_fuel'] = eval_dict
        

    ################################
    ####### CVX Warmstarting ####### 
    ################################
    
    # add waypoint conditions for cvx / cvx-scp 
    prob.waypoint_times = t_idx_wyp
    prob.waypoints = wyp
    # prob.state_final = roe_f  # reset the final state to the original target state
    prob.type = "min_fuel"  # switch back the objective to original problem
    t0 = time.time()
    sol_cvx = prob.ocp_cvx()
    runtime_cvx = time.time() - t0
    states_roe_cvx_i, actions_cvx_i = sol_cvx['z']['state'], sol_cvx['z']['action']
    
    if sol_cvx['status'] in ['optimal', 'optimal_inaccurate']:

        oe_ref = dyn.propagate_oe(prob.oe_i, prob.tvec_sec)
        states_rtn_cvx_i = roe_to_rtn(states_roe_cvx_i, oe_ref)  # Transpose back to (n_time, 6)
        ctg_cvx, _  = prob.compute_ctg(states_roe_cvx_i, actions_cvx_i, prob.tvec_sec, chance=chance, ct=ct)
        roef_cvx = states_roe_cvx_i[-1] + cim_f @ actions_cvx_i[-1]
        states_roe_cvx_i = np.concatenate( (states_roe_cvx_i, roef_cvx.reshape(1, -1)), axis=0 )  # add final state
        states_rtn_cvx_i = np.concatenate( (states_rtn_cvx_i, (psi_f @ roef_cvx).reshape(1, -1)), axis=0 )
        out['rtgs_cvx'] = prob.compute_rtg(actions_cvx_i)
        out['ctgs_cvx'] = ctg_cvx
        out['runtime_cvx'] = runtime_cvx
        out['ctgs0_cvx'] = ctg_cvx[0, 0]
        out['roe_cvx'] = states_roe_cvx_i
        out['rtn_cvx'] = states_rtn_cvx_i
        out['a_cvx'] = actions_cvx_i
        out['J_cvx'] = sol_cvx['cost']
        out['feasible_cvx'] = True
        
        #  Solve SCP
        if scp:
            # reset the waypoint conditions for cvx-scp
            prob.waypoint_times = []
            prob.waypoints = []
            prob.zref = {'state': states_roe_cvx_i[:-1], 'action': actions_cvx_i}
            prob.sol_0 = {"z": prob.zref}
            prob.type = "feasibility"   # switch the objective to feasibility problem
            prob._cvx_built_AL = False; prob.update_flag = True
            prob.generate_scaling(states_roe_cvx_i[:-1], actions_cvx_i)
            t0 = time.time()
            sol_scp, log_scp = solve_scvx(prob)
            runtime_scp = time.time() - t0
            feas_scp_i = sol_scp['status']
            
            if feas_scp_i in ['optimal', 'optimal_inaccurate']:
                # Save scp_cvx data in the output dictionary
                out['J_scp_cvx'] = np.sum(np.linalg.norm(sol_scp['z']['action'], axis=1))
                out['iter_scp_cvx'] = len(log_scp["f0"])
                out['runtime_scp_cvx'] = runtime_scp

                # save SCP trajectory
                roef_scp = sol_scp['z']['state'][-1] + cim_f @ sol_scp['z']['action'][-1]
                out['roe_scp_cvx'] = np.concatenate((sol_scp['z']['state'], roef_scp.reshape(1, -1)), axis=0)
                out['rtn_scp_cvx'] = np.concatenate((sol_scp['z']['state'], (psi_f @ roef_scp).reshape(1, -1)), axis=0)
                out['a_scp_cvx'] = sol_scp['z']['action']
                out['feasible_scp_cvx'] = True

                traj = {"roe": out['roe_scp_cvx']}
                eval_dict = eval_semantics_traj(behavior_i, text_command, template_i, traj, verbose=False)
                out['eval_dict_scp_cvx'] = eval_dict


    #################################################
    ######### REDUCED GRID ##########################
    #################################################
    
    N_burns = 8
    dv_mag = np.linalg.norm(traj_DT['dv'], axis=1)
    largest_burn_indices = np.argsort(dv_mag)[-N_burns:]
    # add the beginning and the end burns, make sure they are sorted / unique 
    largest_burn_indices = np.unique(np.sort(np.concatenate(([0], largest_burn_indices, [len(dv_mag)-1]))))
    dv_pruned = traj_DT['dv'][largest_burn_indices]
    tvec_pruned = param.tvec_sec[largest_burn_indices]
    roe_pruned = traj_DT['roe'][largest_burn_indices]

    zref = {'state' : roe_pruned, 'action' : dv_pruned}
    prob_r = NonConvexOCP(
        prob_definition={
            't_i' : 0,
            't_f' : len(tvec_pruned),
            'tvec_sec' : tvec_pruned,
            'current_obs' : current_obs,
            'chance' : chance,
            'ct' : ct,
            'behavior' : behavior_i
        }, 
        zref = zref 
    )
    
    prob_r.type = "feasibility"
    prob_r.scvx_param.r0 = 0.5
    prob_r.feas_w = 0.2
    t0 = time.time()
    sol_scp_r, log_scp_r = solve_scvx(prob_r)
    runtime_scp_DT_r = time.time() - t0

    if sol_scp_r['status'] in ['optimal', 'optimal_inaccurate']:
        # print("Reduced-grid SCP from DT warmstart is feasible!")
        oe_ref_r = dyn.propagate_oe(prob_r.oe_i, prob_r.tvec_sec)
        out['feasible_scp_DT_r'] = True
        out['J_scp_DT_r'] = np.sum(np.linalg.norm(sol_scp_r['z']['action'], axis=1))
        out['iter_scp_DT_r'] = len(log_scp_r["f0"])
        out['runtime_scp_DT_r'] = runtime_scp_DT_r
        
        roef_scp = sol_scp_r['z']['state'][-1] + cim_f @ sol_scp_r['z']['action'][-1]
        rtnf_scp = roe_to_rtn(sol_scp_r['z']['state'], oe_ref_r)

        out['roe_scp_DT_r'] = np.concatenate((sol_scp_r['z']['state'], roef_scp.reshape(1, -1)), axis=0)
        out['rtn_scp_DT_r'] = np.concatenate((rtnf_scp, (psi_f @ roef_scp).reshape(1, -1)), axis=0)
        out['a_scp_DT_r'] = sol_scp_r['z']['action']
        out['t_scp_DT_r'] = prob_r.tvec_sec

        # traj = {"roe": out['roe_scp_DT_r']}
        # eval_dict = eval_semantics_traj(behavior_i, text_command, template_i, traj, verbose=False)
        out['eval_dict_scp_DT_r'] = {}

    return out


if __name__ == '__main__':

    ############# INPUTS ##########################################################################
    
    dataset_name = 'v08'
    model_name = 'v08_w3' 
    ctg_condition = True  # True if you activate CTG token. False if not (= imitation learning only)
    max_tokens = 50

    num_processes = 10
    N_data = 50  # number of data to try warm-start analysis 
    scp = True           # whether to solve SCP after CVX/ART

    save_fname = 'v08_w3_unseen_orin_r_test'
    tailored_command = True  # whether to use tailored command inputs (with waypoint info)
    unseen_command = True  # whether to use unseen commands (only works when tailored_command=True)
    master_file = 'commands_summary_w3_val.jsonl'  # doesn't matter, deprecated

    timestep_norm = False   # no need to touch 
    
    ################################################################################################

    set_start_method('spawn', force=True)

    # Get the datasets and loaders from the torch data
    datasets, dataloaders = DT_manager.get_train_val_test_data(ctg_condition, dataset_name, timestep_norm, tailored_data=tailored_command)
    train_loader, eval_loader, test_loader = dataloaders  
    data_stats = train_loader.dataset.data_stats

    # Build model on CPU
    model = DT_manager.get_DT_model(model_name, train_loader, eval_loader)
    model.to("cpu").eval()
    model_state = model.state_dict()

    # Build text encoder on CPU
    MODEL = os.getenv("FTA_MODEL", "distilbert-base-uncased")
    text_encoder = FrozenTextAdapter(
        model_name=MODEL, 
        out_dim=model.hidden_size, 
        output_mode="tokens", 
        max_tokens=max_tokens
    ).to("cpu").eval()
    text_state = text_encoder.state_dict()
    del text_encoder, model

    # load the command mapping (master file) 
    command_mapping = []
    with open(root_folder / "rpod/dataset" / master_file, "r") as f:
        for line in f:
            command_mapping.append(json.loads(line))

    # IMPORTANT: eval_loader is using unseen text + unseen numbers, test_loader is using seen commands + unseen numbers
    if unseen_command and tailored_command:
        test_loader = eval_loader

    # Parallel for inputs
    N_data_test = np.min([test_loader.dataset.n_data, N_data])  
    other_args = {
        'test_loader' : test_loader,
        'command_mapping' : command_mapping,
        'scp' : scp,
        'tailored_command' : tailored_command
    }
    
    J_cvx           = np.empty(shape=(N_data_test, ), dtype=float)
    J_DT            = np.empty(shape=(N_data_test, ), dtype=float)
    J_scp_cvx       = np.empty(shape=(N_data_test, ), dtype=float)
    J_scp_DT        = np.empty(shape=(N_data_test, ), dtype=float)
    J_scp_DT_r      = np.empty(shape=(N_data_test, ), dtype=float)
    iter_scp_cvx    = np.empty(shape=(N_data_test, ), dtype=float)
    iter_scp_DT     = np.empty(shape=(N_data_test, ), dtype=float) 
    iter_scp_DT_r   = np.empty(shape=(N_data_test, ), dtype=float)  
    runtime_cvx     = np.empty(shape=(N_data_test, ), dtype=float) 
    runtime_DT      = np.empty(shape=(N_data_test, ), dtype=float)
    runtime_scp_cvx = np.empty(shape=(N_data_test, ), dtype=float) 
    runtime_scp_DT  = np.empty(shape=(N_data_test, ), dtype=float)
    runtime_scp_DT_r = np.empty(shape=(N_data_test, ), dtype=float)
    ctgs0_cvx       = np.empty(shape=(N_data_test, ), dtype=float)
    ctgs0_DT        = np.empty(shape=(N_data_test, ), dtype=float)
    cvx_problem     = np.full(shape=(N_data_test, ), fill_value=False)
    test_dataset_ix = np.empty(shape=(N_data_test, ), dtype=float)
    state_init      = np.empty(shape=(N_data_test, 6), dtype=float)
    state_final     = np.empty(shape=(N_data_test, 6), dtype=object)
    roe_cvx         = np.empty(shape=(N_data_test, n_time+1, 6), dtype=float)
    roe_DT          = np.empty(shape=(N_data_test, n_time+1, 6), dtype=float)
    roe_scp_cvx     = np.empty(shape=(N_data_test, n_time+1, 6), dtype=float)
    roe_scp_DT      = np.empty(shape=(N_data_test, n_time+1, 6), dtype=float)
    roe_scp_DT_r    = np.empty(shape=(N_data_test, n_time+1, 6), dtype=float)
    rtn_cvx         = np.empty(shape=(N_data_test, n_time+1, 6), dtype=float)
    rtn_DT          = np.empty(shape=(N_data_test, n_time+1, 6), dtype=float)
    rtn_scp_cvx     = np.empty(shape=(N_data_test, n_time+1, 6), dtype=float)  
    rtn_scp_DT      = np.empty(shape=(N_data_test, n_time+1, 6), dtype=float)
    rtn_scp_DT_r    = np.empty(shape=(N_data_test, n_time+1, 6), dtype=float)
    a_cvx           = np.empty(shape=(N_data_test, n_time, 3), dtype=float)
    a_DT            = np.empty(shape=(N_data_test, n_time, 3), dtype=float)
    a_scp_cvx       = np.empty(shape=(N_data_test, n_time, 3), dtype=float)
    a_scp_DT        = np.empty(shape=(N_data_test, n_time, 3), dtype=float)
    a_scp_DT_r      = np.empty(shape=(N_data_test, n_time, 3), dtype=float)
    t_scp_DT_r      = np.empty(shape=(N_data_test, n_time), dtype=float)
    behavior        = np.empty(shape=(N_data_test, ), dtype=float)
    oe              = np.empty(shape=(N_data_test, n_time, 6), dtype=float)

    eval_dict_DT = []
    eval_dict_scp_cvx = []
    eval_dict_scp_DT = []
    eval_dict_scp_DT_fuel = []
    eval_dict_scp_DT_r = []
    
    i_infeas_cvx = []
    i_infeas_scp_cvx = []
    i_infeas_scp_DT = []
    i_infeas_scp_DT_fuel = []
    i_infeas_scp_DT_r = []

    # input aggregation 
    inputs = np.arange(N_data_test)
    with Pool(processes=num_processes, initializer=init_worker, 
              initargs=(model_name, model_state, text_state, data_stats,
              test_loader, command_mapping, scp, tailored_command, max_tokens)) as p:  
        iterator = p.imap(for_computation, inputs)
        
        for i, res in enumerate(tqdm(iterator, total=N_data_test)):   # Save the input in the dataset
            # test_dataset_ix[i] = res['test_dataset_ix']
            state_init[i]  = res['state_init']
            state_final[i] = res['state_final']
            behavior[i]    = res['behavior']
            oe[i,:,:]      = res['oe']
            
            # ART output 
            roe_DT[i,:,:] = res['roe_DT']
            rtn_DT[i,:,:] = res['rtn_DT']      
            a_DT[i,:,:]   = res['a_DT']     
            ctgs0_DT[i]   = res['ctgs0_DT']
            J_DT[i]       = res['J_DT']
            eval_dict_DT.append(res['eval_dict_DT'])
            runtime_DT[i]  = res['runtime_DT']

            # If the solution is feasible save the optimization output
            if res['feasible_cvx']:
                J_cvx[i] = res['J_cvx']
                runtime_cvx[i] = res['runtime_cvx']
                ctgs0_cvx[i]   = res['ctgs0_cvx']
                cvx_problem[i] = res['cvx_problem']
                roe_cvx[i,:,:] = res['roe_cvx']
                rtn_cvx[i,:,:] = res['rtn_cvx']
                a_cvx[i,:,:]   = res['a_cvx']
            else:
                i_infeas_cvx += [ i ]

            if res['feasible_scp_cvx']:
                iter_scp_cvx[i] = res['iter_scp_cvx']
                runtime_scp_cvx[i] = res['runtime_scp_cvx']
                J_scp_cvx[i] = res['J_scp_cvx']
                roe_scp_cvx[i,:,:] = res['roe_scp_cvx']
                rtn_scp_cvx[i,:,:] = res['rtn_scp_cvx']
                a_scp_cvx[i,:,:]   = res['a_scp_cvx']
                eval_dict_scp_cvx.append(res['eval_dict_scp_cvx'])
            else:
                i_infeas_scp_cvx += [ i ]
                
            if res['feasible_scp_DT']:
                iter_scp_DT[i] = res['iter_scp_DT']
                runtime_scp_DT[i] = res['runtime_scp_DT']
                J_scp_DT[i] = res['J_scp_DT']
                roe_scp_DT[i,:,:] = res['roe_scp_DT']
                rtn_scp_DT[i,:,:] = res['rtn_scp_DT']
                a_scp_DT[i,:,:]   = res['a_scp_DT']
                eval_dict_scp_DT.append(res['eval_dict_scp_DT'])
            else:
                i_infeas_scp_DT += [ i ]
                
            if res['feasible_scp_DT_r']:
                # careful! reduced grid
                iter_scp_DT_r[i] = res['iter_scp_DT_r']
                runtime_scp_DT_r[i] = res['runtime_scp_DT_r']
                J_scp_DT_r[i] = res['J_scp_DT_r']
                
                n_time = res['roe_scp_DT_r'].shape[0] -1 
                roe_scp_DT_r[i,:n_time+1,:] = res['roe_scp_DT_r']
                rtn_scp_DT_r[i,:n_time+1,:] = res['rtn_scp_DT_r']
                a_scp_DT_r[i,:n_time,:]   = res['a_scp_DT_r']
                t_scp_DT_r[i,:n_time]     = res['t_scp_DT_r']
                eval_dict_scp_DT_r.append(res['eval_dict_scp_DT_r'])
            else:
                i_infeas_scp_DT_r += [ i ]

            if res['feasible_scp_DT_fuel']:
                eval_dict_scp_DT_fuel.append(res['eval_dict_scp_DT_fuel'])
            else: 
                i_infeas_scp_DT_fuel += [ i ]
        
    #  Save dataset (local folder for the workstation)
    save_path = root_folder / "rpod/optimization/saved_files/warmstarting"
    save_path.mkdir(parents=True, exist_ok=True)  # create if missing
    np.savez_compressed(
        save_path / f"ws_analysis_{save_fname}.npz",
        J_cvx = J_cvx,
        J_DT = J_DT,
        J_scp_cvx = J_scp_cvx,
        J_scp_DT = J_scp_DT,
        J_scp_DT_r = J_scp_DT_r,
        iter_scp_cvx = iter_scp_cvx,
        iter_scp_DT = iter_scp_DT,
        iter_scp_DT_r = iter_scp_DT_r,
        runtime_cvx = runtime_cvx,
        runtime_DT = runtime_DT,
        runtime_scp_cvx = runtime_scp_cvx,
        runtime_scp_DT = runtime_scp_DT,
        runtime_scp_DT_r = runtime_scp_DT_r,
        ctgs0_cvx = ctgs0_cvx, 
        ctgs0_DT = ctgs0_DT,
        cvx_problem = cvx_problem,
        test_dataset_ix = test_dataset_ix,
        state_init = state_init,
        state_final = state_final,
        i_infeas_cvx = i_infeas_cvx,
        i_infeas_scp_cvx = i_infeas_scp_cvx,
        i_infeas_scp_DT = i_infeas_scp_DT, 
        i_infeas_scp_DT_fuel = i_infeas_scp_DT_fuel,
        i_infeas_scp_DT_r = i_infeas_scp_DT_r,
        roe_cvx = roe_cvx, rtn_cvx = rtn_cvx, a_cvx = a_cvx,
        roe_scp_cvx = roe_scp_cvx, rtn_scp_cvx = rtn_scp_cvx, a_scp_cvx = a_scp_cvx,
        roe_DT = roe_DT, rtn_DT = rtn_DT, a_DT = a_DT,
        roe_scp_DT = roe_scp_DT, rtn_scp_DT = rtn_scp_DT, a_scp_DT = a_scp_DT,
        roe_scp_DT_r = roe_scp_DT_r, rtn_scp_DT_r = rtn_scp_DT_r, a_scp_DT_r = a_scp_DT_r,
        behavior = behavior, 
        oe = oe, 
        eval_dict_DT = eval_dict_DT, eval_dict_scp_cvx = eval_dict_scp_cvx, 
        eval_dict_scp_DT = eval_dict_scp_DT, eval_dict_scp_DT_fuel = eval_dict_scp_DT_fuel,
        eval_dict_scp_DT_r = eval_dict_scp_DT_r
        )
    print(f"Warm-starting analysis data saved in {save_path}/ws_analysis_{save_fname}.npz")
