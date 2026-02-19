import os, sys
def find_root_path(path:str, word:str):
    parts = path.split(word, 1)
    return parts[0] + word if len(parts) > 1 else path 
root_folder = find_root_path(os.getcwd(), 'art_lang')
sys.path.append(root_folder)

import numpy as np
import torch 
import json
from multiprocessing import Pool, get_context
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()   

from rpod.optimization.parameters import *
from rpod.dynamics.dynamics_trans import roe_to_rtn
from rpod.optimization.scvx import solve_scvx
from rpod.optimization.optimization import NonConvexOCP
from rpod.dataset_generation.annotation import annotate_number2
# HACK 
root_folder = str(root_folder)

def for_computation(current_data_index):
    
    # behav = np.random.randint(0,2) + 4
    behav = None
    
    behavior_mode, roe0, roef, t_idx_wyp, wyp = sample_reset_condition2(behavior=behav)

    # Output dictionary initialization
    out = {'feasible' : True,
           'states_roe_cvx' : [],
           'states_rtn_cvx' : [],
           'actions_cvx' : [],
           'states_roe_scp': [],
           'states_rtn_scp' : [],
           'actions_scp' : [],
           'target_state' : [],
           'horizons' : [],
           'dtime' : [],
           'time' : [],
           'oe' : [],
           'rtgs_cvx' : [], 
           'rtgs_scp' : [],
           'ctgs_cvx' : [],
           'ctgs_scp' : [], 
           'behavior' : behavior_mode,
           'waypoints' : np.vstack((roe0, np.array(wyp), roef)) if len(wyp) > 0 else np.vstack((roe0, roef)),
           'waypoint_times' : np.concatenate(([0], np.array(t_idx_wyp), [n_time-1])) if len(wyp) > 0 else np.array([0, n_time-1]),
           }   

    # Define current observation
    current_obs = {'state' : roe0, 'goal' : roef, 'ttg' : tf_sec, 'dt' : dt_sec, 'oe' : oec0}
    
    chance = False
    ct = True 

    prob = NonConvexOCP(
        prob_definition={
            't_i' : 0,
            't_f' : n_time,
            'tvec_sec' : tvec_sec,
            'chance' : chance,
            'current_obs' : current_obs,
            'waypoint_times' : t_idx_wyp,
            'waypoints' : wyp,
            'ct' : ct,
        }
    )

    sol_cvx = prob.ocp_cvx()

    states_roe_cvx_i, actions_cvx_i = sol_cvx['z']['state'], sol_cvx['z']['action']
    feas_cvx_i = sol_cvx['status']

    if feas_cvx_i in ['optimal', 'optimal_inaccurate']:
        
        # Mapping done after the feasibility check to avoid NoneType errors
        # roe_to_rtn expects (6, n_times) format, transpose if needed and use full OE history
        oe = propagate_oe(prob.oe_i, prob.tvec_sec)  # (n_time, 6)
        states_rtn_cvx_i = roe_to_rtn(states_roe_cvx_i, oe)  # Transpose back to (n_time, 6)

        out['rtgs_cvx'] = prob.compute_rtg(actions_cvx_i)
        ctg_cvx, _  = prob.compute_ctg(states_roe_cvx_i, actions_cvx_i, prob.tvec_sec, chance=chance, ct=ct)
        out['ctgs_cvx'] = ctg_cvx
        
        #  Solve transfer scp
        prob.zref = {'state': states_roe_cvx_i, 'action': actions_cvx_i}
        prob.sol_0 = {"z": prob.zref}
        prob.generate_scaling(states_roe_cvx_i, actions_cvx_i)
        sol_scp, log_scp = solve_scvx(prob)
        feas_scp_i = sol_scp['status']

        if feas_scp_i in ['optimal', 'optimal_inaccurate']:
            # Mapping done after feasibility check to avoid NoneType errors
            # roe_to_rtn expects (6, n_times) format, transpose if needed and use full OE history
            states_roe_scp_i = sol_scp['z']['state']
            actions_scp_i = sol_scp['z']['action']
            
            states_rtn_scp_i = roe_to_rtn(states_roe_scp_i, oe) 
            out['states_roe_cvx'] = states_roe_cvx_i
            out['states_rtn_cvx'] = states_rtn_cvx_i
            out['actions_cvx']    = actions_cvx_i
            out['states_roe_scp'] = states_roe_scp_i
            out['states_rtn_scp'] = states_rtn_scp_i
            out['actions_scp']    = actions_scp_i
            
            out['target_state'] = roef
            out['horizons'] = prob.horizon
            out['dtime'] = dt_sec
            out['time'] = tvec_sec
            out['oe'] = oe
            
            # post-process rtg and ctg
            out['rtgs_scp'] = prob.compute_rtg(actions_scp_i)   # (1, n_time)
            ctg_scp, _  = prob.compute_ctg(states_roe_scp_i, actions_scp_i, prob.tvec_sec, chance=chance, ct=ct)  # (1, n_time)
            out['ctgs_scp'] = ctg_scp
        else:
            out['feasible'] = False
        
    #     # quick data-gen using only cvx 
    #     # Save cvx and scp problems in the output dictionary
    #     out['states_roe_cvx'] = np.transpose(states_roe_cvx_i)
    #     out['states_rtn_cvx'] = np.transpose(states_rtn_cvx_i)
    #     out['actions_cvx'] = np.transpose(actions_cvx_i)
        
    #     out['states_roe_scp'] = np.transpose(states_roe_cvx_i)
    #     out['states_rtn_scp'] = np.transpose(states_rtn_cvx_i)
    #     out['actions_scp'] = np.transpose(actions_cvx_i)

    #     out['target_state'] = roef
    #     out['horizons'] = horizon
    #     out['dtime'] = prob.dt_sec
    #     out['time'] = np.transpose(prob.tvec_sec)
    #     out['oe'] = np.transpose(prob.oe)

    #     out['rtgs_cvx'] = compute_reward_to_go(actions_cvx_i.reshape((1, n_time, 3)), 1, n_time).T
    #     out['rtgs_scp'] = compute_reward_to_go(actions_cvx_i.reshape((1, n_time, 3)), 1, n_time).T
    #     out['ctgs_cvx'] = compute_constraint_to_go(states_rtn_cvx_i.reshape((1, n_time, 6)), 1, n_time).T
    #     out['ctgs_scp'] = compute_constraint_to_go(states_rtn_cvx_i.reshape((1, n_time, 6)), 1, n_time).T 
        
    else:
        out['feasible'] = False
    
    return out

if __name__ == '__main__':

    N_data = 30
    N_proc = 20
    ver_name = 'test'
    master_file = "w3"   # command file version (unfilled) 
    
    n_S = 6 # state size
    n_A = 3 # action size

    dataset_dir = root_folder + '/rpod/dataset/torch/' + ver_name   
    os.makedirs(dataset_dir, exist_ok=True)   # ensures all subfolders exist
    dataset_path = Path(dataset_dir) / "dataset-rpod-param.npz"
    assert not dataset_path.exists() or ver_name == 'test', f"Error: Seems like dataset already exists in {dataset_dir}. Please remove it manually if you want to regenerate it with this name."

    states_roe_cvx = np.empty(shape=(N_data, n_time, n_S), dtype=float) # [m]
    states_rtn_cvx = np.empty(shape=(N_data, n_time, n_S), dtype=float) # [m,m,m,m/s,m/s,m/s]
    actions_cvx = np.empty(shape=(N_data, n_time, n_A), dtype=float) # [m/s]
    rtgs_cvx = np.empty(shape=(N_data, n_time, 1), dtype=float) 
    ctgs_cvx = np.empty(shape=(N_data, n_time, 1), dtype=float) 

    states_roe_scp = np.empty(shape=(N_data, n_time, n_S), dtype=float) # [m]
    states_rtn_scp = np.empty(shape=(N_data, n_time, n_S), dtype=float) # [m,m,m,m/s,m/s,m/s]
    actions_scp = np.empty(shape=(N_data, n_time, n_A), dtype=float) # [m/s]
    rtgs_scp = np.empty(shape=(N_data, n_time, 1), dtype=float)
    ctgs_scp = np.empty(shape=(N_data, n_time, 1), dtype=float) 
    ctg_cvx_full = np.empty(shape=(N_data, ), dtype=float)   # final ctg value for each data point (including infeasible ones) 

    target_state = np.empty(shape=(N_data, n_S), dtype=float)
    horizons = np.empty(shape=(N_data, ), dtype=float)
    dtime = np.empty(shape=(N_data, ), dtype=float)
    time = np.empty(shape=(N_data, n_time), dtype=float)
    oe = np.empty(shape=(N_data, n_time, n_S), dtype=float)
    
    behavior_mode = np.empty(shape=(N_data, ), dtype=int)
    
    wyp = np.full((N_data, 5, n_S), np.nan, dtype=float)
    t_idx_wyp = np.full((N_data, 5), -1, dtype=int)

    i_unfeas = []

    # Pool creation --> Should automatically select the maximum number of processes
    ctx = get_context("spawn")  # Windows-safe
    with ctx.Pool(processes=N_proc) as p:  # avoiding the pool shutdown issue on Windows
        for i, res in enumerate(tqdm(p.imap(for_computation, np.arange(N_data)), total=N_data)):
            # If the solution is feasible save the optimization output
            
            behavior_mode[i] = res['behavior']
            
            if res['feasible']:
                states_roe_cvx[i,:,:] = res['states_roe_cvx']
                states_rtn_cvx[i,:,:] = res['states_rtn_cvx']
                actions_cvx[i,:,:] = res['actions_cvx']

                states_roe_scp[i,:,:] = res['states_roe_scp']
                states_rtn_scp[i,:,:] = res['states_rtn_scp']
                actions_scp[i,:,:] = res['actions_scp']

                target_state[i,:] = res['target_state']
                horizons[i] = res['horizons']
                dtime[i] = res['dtime']
                time[i,:] = res['time']
                oe[i,:,:] = res['oe']

                rtgs_scp[i,:,:] = res['rtgs_scp'].T
                ctgs_scp[i,:,:] = res['ctgs_scp'].T
                
                rtgs_cvx[i,:,:] = res['rtgs_cvx'].T
                ctgs_cvx[i,:,:] = res['ctgs_cvx'].T
                
                n_wyp = res["waypoints"].shape[0]  # > 0 because start and goal are included
                wyp[i, :n_wyp, :] = res["waypoints"]
                t_idx_wyp[i, :n_wyp] = res["waypoint_times"]

            else:
                i_unfeas += [ i ]

            if len(res['ctgs_cvx']) > 0:
                arr = np.asarray(res['ctgs_cvx'])[0, :, 0] if res['ctgs_cvx'].ndim == 3 else np.asarray(res['ctgs_cvx'])[0, :]
                ctg_cvx_full[i] = float(np.nanmean(arr))   # collapse to scalar
            else:
                ctg_cvx_full[i] = np.nan
                
        behavior_mode_full = behavior_mode.copy()
            
        # Remove unfeasible data points
        if i_unfeas:
            states_roe_cvx = np.delete(states_roe_cvx, i_unfeas, axis=0)
            states_rtn_cvx = np.delete(states_rtn_cvx, i_unfeas, axis=0)
            actions_cvx    = np.delete(actions_cvx, i_unfeas, axis=0)
            
            states_roe_scp = np.delete(states_roe_scp, i_unfeas, axis=0)
            states_rtn_scp = np.delete(states_rtn_scp, i_unfeas, axis=0)
            actions_scp    = np.delete(actions_scp, i_unfeas, axis=0)
            
            target_state = np.delete(target_state, i_unfeas, axis=0)
            horizons = np.delete(horizons, i_unfeas, axis=0)
            dtime = np.delete(dtime, i_unfeas, axis=0)
            time = np.delete(time, i_unfeas, axis=0)
            oe = np.delete(oe, i_unfeas, axis=0)
            
            rtgs_cvx = np.delete(rtgs_cvx, i_unfeas, axis=0)
            ctgs_cvx = np.delete(ctgs_cvx, i_unfeas, axis=0)
            rtgs_scp = np.delete(rtgs_scp, i_unfeas, axis=0)
            ctgs_scp = np.delete(ctgs_scp, i_unfeas, axis=0)
            
            behavior_mode = np.delete(behavior_mode, i_unfeas, axis=0)
            
            wyp =  np.delete(wyp, i_unfeas, axis=0)
            t_idx_wyp = np.delete(t_idx_wyp, i_unfeas, axis=0)


    """
    This part is required only if you want to annotate for each data point 
    """    
    # from decision_transformer.adapter import FrozenTextAdapter  
    # device =  "cuda" if torch.cuda.is_available() else "cpu"
    # print(device) 
    # from dataset_generation.gpt_prompting import annotate_traj_behaviors2
    
    # # Annotate the behavior mode
    # # Set Gemini API Key as an environment variable
    # host = "openai"
    # if host == "openai":    
    #     api_key = os.getenv("OPENAI_API_KEY")
    # elif host == "google":
    #     api_key = os.getenv("GOOGLE_API_KEY")
    
    # behavior_mode_annotated = annotate_traj_behaviors2(behavior_mode, api_key, host=host)
    
    # # Extract IDs and descriptions
    # ids = [behavior_mode_annotated[i]['id'] for i in range(len(behavior_mode_annotated))]
    # descriptions = [behavior_mode_annotated[i]['description'] for i in range(len(behavior_mode_annotated))]
    # commands = [COMMAND_LIST[behavior_mode_annotated[i]['id']] for i in range(len(behavior_mode_annotated))]
    
    # with open(dataset_dir + "/behavior_labels.jsonl", "a", encoding="utf-8") as f:
    #     for cid, name, text in zip(ids, commands, descriptions):
    #         json.dump({"class_id": int(cid), "case_name": str(name), "case_text": str(text)}, f, ensure_ascii=False)
    #         f.write("\n")
    
    # # collect embeddings from the text command 
    # MODEL = os.getenv("FTA_MODEL", "distilbert-base-uncased")   # this is encoder only     
    # adapter = FrozenTextAdapter(model_name=MODEL, out_dim=384, output_mode="tokens").to(device).eval()
    # with torch.inference_mode():
    #     out = adapter(commands)  # forward pass 
    # torch.save(out.cpu(), dataset_dir + '/torch_text_embeddings.pth')    

    # create random command ids for each behavior entry (integers in [0,100))
    command_id = np.random.randint(0, 100, size=behavior_mode.shape[0])
    np.savez_compressed(dataset_dir + '/dataset-rpod-param', target_state=target_state, time=time, oe=oe, dtime=dtime, horizons=horizons, 
                                                            behavior=behavior_mode, command_id=command_id, 
                                                            i_unfeas=i_unfeas, behavior_full=behavior_mode_full, ctg_all=ctg_cvx_full, 
                                                            waypoints=wyp, waypoint_times=t_idx_wyp,)

    # save torch file directly
    torch_states_roe_cvx = torch.from_numpy(states_roe_cvx)
    torch_states_rtn_cvx = torch.from_numpy(states_rtn_cvx)
    torch_actions_cvx = torch.from_numpy(actions_cvx)
    torch_states_roe_scp = torch.from_numpy(states_roe_scp)
    torch_states_rtn_scp = torch.from_numpy(states_rtn_scp)
    torch_actions_scp = torch.from_numpy(actions_scp)
    torch_behavior_mode = torch.from_numpy(behavior_mode)
    torch_command_id = torch.from_numpy(command_id)

    torch.save(torch_states_roe_cvx, dataset_dir + '/torch_states_roe_cvx.pth')
    torch.save(torch_states_rtn_cvx, dataset_dir + '/torch_states_rtn_cvx.pth')    
    torch.save(torch_states_roe_scp, dataset_dir + '/torch_states_roe_scp.pth')
    torch.save(torch_states_rtn_scp, dataset_dir + '/torch_states_rtn_scp.pth')
    torch.save(torch_actions_scp, dataset_dir + '/torch_actions_scp.pth')
    torch.save(torch_actions_cvx, dataset_dir + '/torch_actions_cvx.pth')
    torch.save(torch_behavior_mode, dataset_dir + '/torch_behavior_mode.pth')
    torch.save(torch_command_id, dataset_dir + '/torch_command_id.pth')

    torch_rtgs_cvx = torch.from_numpy(rtgs_cvx)
    torch_rtgs_scp = torch.from_numpy(rtgs_scp)
    torch_ctgs_cvx = torch.from_numpy(ctgs_cvx)
    torch_ctgs_scp = torch.from_numpy(ctgs_scp)

    torch.save(torch_rtgs_scp, dataset_dir + '/torch_rtgs_scp.pth')
    torch.save(torch_rtgs_cvx, dataset_dir + '/torch_rtgs_cvx.pth')
    torch.save(torch_ctgs_scp, dataset_dir + '/torch_ctgs_scp.pth')
    torch.save(torch_ctgs_cvx, dataset_dir + '/torch_ctgs_cvx.pth')

    # Permutation
    if states_rtn_cvx.shape[0] != states_rtn_scp.shape[0]:
        raise RuntimeError('Different dimensions of cvx and scp datasets.')
    perm = np.random.permutation(states_rtn_cvx.shape[0]*2)
    np.save(dataset_dir + '/permutation.npy', perm)

    # Annotate commands with numbers filled in
    print("Annotating commands with numbers filled in...")
    data_param = np.load(dataset_dir + '/dataset-rpod-param.npz', allow_pickle=True)
    command_path = f'rpod/dataset/commands_summary_{master_file}_train.jsonl'
    # load command file
    with open(command_path, 'r') as f:
        command_list = [json.loads(line) for line in f]
    _ = annotate_number2(data_param, command_list, save_dir=dataset_dir, command_id=command_id)

    print('dataset generation completed successfully.')
