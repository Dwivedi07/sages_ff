import os, sys
from pathlib import Path
root_folder = Path(__file__).resolve().parent.parent.parent  # /art_lang/
sys.path.append(str(root_folder))
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import argparse

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from transformers import DecisionTransformerConfig #, DecisionTransformerModel, Trainer, TrainingArguments, get_scheduler
from accelerate import Accelerator

# /src/
from rpod.dynamics.dynamics_trans import mu_E
from rpod.optimization.optimization import NonConvexOCP
from rpod.decision_transformer.art import AutonomousRendezvousTransformerLang

# FIXME: mixing up root_folder now (str and Path objects)
root_folder = str(root_folder)

# select device based on availability of GPU
verbose = False # set to True to get additional print statements
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

class RpodDatasetLang(Dataset):
    """
    Arguments: 
        - data: dictionary containing the dataset information
        - ctg_condition (bool): whether the dataset includes constraint information (True: Text + CTG, False: Text only)
        - tailored_data (bool): whether to use the individually annotated data (True) or the original data without annotation (False)
            If True, then behavior and command_id are ignored, and text_command (str) is used instead.
    """
    def __init__(self, data, ctg_condition, tailored_data=False):
        self.data_stats = data['data_stats']
        self.data = data
        self.n_data, self.max_len, self.n_state = self.data['states'].shape
        self.n_action = self.data['actions'].shape[2]
        self.ctg_condition = ctg_condition
        self.tailored_data = tailored_data  # Ture: use the individually annotated data, False: use the original data without annotation, and randomly draw the command ad hoc

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        ix = torch.randint(self.n_data, (1,))
        states  = torch.stack([self.data['states'][i, :, :] for i in ix]).view(self.max_len, self.n_state).float()
        actions = torch.stack([self.data['actions'][i, :, :] for i in ix]).view(self.max_len, self.n_action).float()
        ctgs = torch.stack([self.data['ctgs'][i, :] for i in ix]).view(self.max_len, 1).float()   
        timesteps = torch.tensor([[i for i in range(self.max_len)] for _ in ix]).view(self.max_len).long()
        attention_mask = torch.ones(1, self.max_len).view(self.max_len).long()

        horizons = self.data['data_param']['horizons'][ix].item()
        oe = np.transpose(self.data['data_param']['oe'][ix])
        time_discr = self.data['data_param']['time_discr'][ix].item()
        time_sec = self.data['data_param']['time_sec'][ix].reshape((1, self.max_len))
        behavior = self.data['data_param']['behavior'][ix].item()
        command_id = self.data['data_param']['command_id'][ix].item()

        # command 
        if not self.tailored_data:
            return states, actions, ctgs, timesteps, attention_mask, oe, time_discr, time_sec, horizons, ix, behavior, command_id
        else: 
            text_command = self.data['data_param']['text_command'][ix]
            wyp = self.data['data_param']['waypoints'][ix]
            wyp_times = self.data['data_param']['waypoint_times'][ix]
            return states, actions, ctgs, timesteps, attention_mask, oe, time_discr, time_sec, horizons, ix, behavior, command_id, text_command, wyp, wyp_times

    def getix(self, ix):
        ix = [ix]
        states = torch.stack([self.data['states'][i, :, :] for i in ix]).view(self.max_len, self.n_state).float().unsqueeze(0)
        actions = torch.stack([self.data['actions'][i, :, :] for i in ix]).view(self.max_len, self.n_action).float().unsqueeze(0)
        ctgs = torch.stack([self.data['ctgs'][i, :] for i in ix]).view(self.max_len, 1).float()
        timesteps = torch.tensor([[i for i in range(self.max_len)] for _ in ix]).view(self.max_len).long().unsqueeze(0)
        attention_mask = torch.ones(1, self.max_len).view(self.max_len).long().unsqueeze(0)

        horizons = torch.tensor(self.data['data_param']['horizons'][ix].item())
        oe = torch.tensor(np.transpose(self.data['data_param']['oe'][ix]))
        time_discr = torch.tensor(self.data['data_param']['time_discr'][ix].item())
        time_sec = torch.tensor(self.data['data_param']['time_sec'][ix].reshape((1, self.max_len))).unsqueeze(0)
        behavior = self.data['data_param']['behavior'][ix].item()
        command_id = self.data['data_param']['command_id'][ix].item()

        # ship command
        if not self.tailored_data:
            return states, actions, ctgs, timesteps, attention_mask, oe, time_discr, time_sec, horizons, ix, behavior, command_id
        else:
            text_command = self.data['data_param']['text_command'][ix[0]]
            wyp = self.data['data_param']['waypoints'][ix]
            wyp_times = self.data['data_param']['waypoint_times'][ix]
            return states, actions, ctgs, timesteps, attention_mask, oe, time_discr, time_sec, horizons, ix, behavior, command_id, text_command, wyp, wyp_times

    def get_data_size(self):
        return self.n_data


def get_train_val_test_data(ctg_condition, dataset_name, timestep_norm, equal_datasize=False, tailored_data=False):

    # Import and normalize torch dataset, then save data statistics
    torch_data, data_param = import_dataset(ctg_condition, dataset_name, equal_datasize, tailored_data)
    states_norm, states_mean, states_std = normalize(torch_data['torch_states'], timestep_norm)
    actions_norm, actions_mean, actions_std = normalize(torch_data['torch_actions'], timestep_norm)
    # goal_norm, goal_mean, goal_std = normalize(torch_data['torch_goal'], timestep_norm)
    target_states_norm = states_norm[:,1:,:].clone().detach()
    target_actions_norm = actions_norm.clone().detach()
    ctgs_norm, ctgs_mean, ctgs_std = torch_data['torch_ctgs'], None, None

    data_stats = {
        'states_mean' : states_mean,
        'states_std' : states_std,
        'actions_mean' : actions_mean,
        'actions_std' : actions_std,
        'ctgs_mean' : ctgs_mean,
        'ctgs_std' : ctgs_std,
    }

    # Split dataset into training and validation
    n = int(0.9*states_norm.shape[0])
    train_data = {
        'states' : states_norm[:n, :],
        'actions' : actions_norm[:n, :],
        'ctgs' : ctgs_norm[:n, :],
        'target_states' : target_states_norm[:n, :],
        'target_actions' : target_actions_norm[:n, :],
        'data_param' : {
            'horizons' : data_param['horizons'][:n],
            'time_discr' : data_param['time_discr'][:n],
            'time_sec' : data_param['time_sec'][:n, :],
            'oe' : data_param['oe'][:n, :], 
            'behavior' : data_param['behavior'][:n],
            'command_id' : data_param['command_id'][:n]
            },
        'data_stats' : data_stats
        }
    val_data = {
        'states' : states_norm[n:, :],
        'actions' : actions_norm[n:, :],
        'ctgs' : ctgs_norm[n:, :],
        'target_states' : target_states_norm[n:, :],
        'target_actions' : target_actions_norm[n:, :],
        'data_param' : {
            'horizons' : data_param['horizons'][n:],
            'time_discr' : data_param['time_discr'][n:],
            'time_sec' : data_param['time_sec'][n:, :],
            'oe' : data_param['oe'][n:, :],
            'behavior' : data_param['behavior'][n:],
            'command_id' : data_param['command_id'][n:]
            },
        'data_stats' : data_stats
        }
    test_data = {
        'states' : states_norm[n:, :],
        'actions' : actions_norm[n:, :],
        'ctgs' : ctgs_norm[n:, :],
        'target_states' : target_states_norm[n:, :],
        'target_actions' : target_actions_norm[n:, :],
        'data_param' : {
            'horizons' : data_param['horizons'][n:],
            'time_discr' : data_param['time_discr'][n:],
            'time_sec' : data_param['time_sec'][n:, :],
            'oe' : data_param['oe'][n:, :],
            'behavior' : data_param['behavior'][n:],
            'command_id' : data_param['command_id'][n:]
            },
        'data_stats' : data_stats
        }

    if tailored_data:
        train_data['data_param']['text_command'] = data_param['text_command'][:n]        
        train_data['data_param']['waypoints'] = data_param['waypoints'][:n]
        train_data['data_param']['waypoint_times'] = data_param['waypoint_times'][:n]
        
        val_data['data_param']['text_command']   = data_param['text_command_val'][n:]  # IMPORTANT: use text_command_val for validation
        val_data['data_param']['waypoints'] = data_param['waypoints'][n:]
        val_data['data_param']['waypoint_times'] = data_param['waypoint_times'][n:]
        
        test_data['data_param']['text_command']  = data_param['text_command'][n:]  # IMPORTANT: use text_command for testing
        test_data['data_param']['waypoints'] = data_param['waypoints'][n:]
        test_data['data_param']['waypoint_times'] = data_param['waypoint_times'][n:]

    # Create datasets
    train_dataset = RpodDatasetLang(train_data, ctg_condition, tailored_data)
    val_dataset   = RpodDatasetLang(val_data, ctg_condition, tailored_data)
    test_dataset  = RpodDatasetLang(test_data, ctg_condition, tailored_data)
    datasets = (train_dataset, val_dataset, test_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(
            train_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=4,
        num_workers=0,
    )
    eval_loader = DataLoader(
        val_dataset,
        sampler=torch.utils.data.RandomSampler(
            val_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=4,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=torch.utils.data.RandomSampler(
            test_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=1,
        num_workers=0,
    )
    dataloaders = (train_loader, eval_loader, test_loader)
    
    return datasets, dataloaders

def import_dataset(ctg_condition, dataset_version, equal_datasize=False, tailored_data=False):
    
    print('Loading data from root/dataset/torch/...', end='')
    data_dir = root_folder + '/rpod/dataset/torch/' + dataset_version
    states_cvx  = torch.load(data_dir + '/torch_states_roe_cvx.pth')
    states_scp  = torch.load(data_dir + '/torch_states_roe_scp.pth')
    actions_cvx = torch.load(data_dir + '/torch_actions_cvx.pth')
    actions_scp = torch.load(data_dir + '/torch_actions_scp.pth')
    rtgs_cvx    = torch.load(data_dir + '/torch_rtgs_cvx.pth')
    rtgs_scp    = torch.load(data_dir + '/torch_rtgs_scp.pth')
    ctgs_cvx    = torch.load(data_dir + '/torch_ctgs_cvx.pth')
    ctgs_scp    = torch.load(data_dir + '/torch_ctgs_scp.pth')
    data_param  = np.load(data_dir + '/dataset-rpod-param.npz')
    
    # load a list of string from .pth file, and convert to numpy array for easier indexing later
    if tailored_data:
        text_commands = np.array(torch.load(data_dir + '/annotation_texts.pth'))
        text_commands_val = np.array(torch.load(data_dir + '/annotation_texts_val.pth'))
    
    print('Completed, DATA IS NOT SHUFFLED YET.\n')

    horizons = data_param['horizons']; dtime = data_param['dtime']; time = data_param['time']
    oe = data_param['oe']; beh = data_param['behavior']; cmd = data_param['command_id']; tgt = data_param['target_state']

    # === (1) Optional balancing on SCP only, mirror to CVX with SAME indices ===
    if equal_datasize:
        classes = np.unique(beh)
        min_count = min((beh == c).sum() for c in classes)
        taken = {int(c): 0 for c in classes}
        keep = []
        for i in np.random.permutation(beh.shape[0]):
            c = int(beh[i])
            if taken[c] < min_count:
                keep.append(i); taken[c] += 1
            if all(taken[int(cc)] >= min_count for cc in classes):
                break
        keep = np.array(keep, dtype=int)

        states_scp = states_scp[keep]; actions_scp = actions_scp[keep]
        rtgs_scp   = rtgs_scp[keep];   ctgs_scp    = ctgs_scp[keep]
        states_cvx = states_cvx[keep]; actions_cvx = actions_cvx[keep]
        rtgs_cvx   = rtgs_cvx[keep];   ctgs_cvx    = ctgs_cvx[keep]
        horizons   = horizons[keep];   dtime       = dtime[keep]
        time       = time[keep];       oe          = oe[keep]
        beh        = beh[keep];        cmd         = cmd[keep]
        tgt        = tgt[keep]         # << carry target_state through the same selection
        
        if tailored_data:
            wyp, wyp_times = data_param['waypoints'][keep], data_param['waypoint_times'][keep]
            text_commands_f = text_commands[keep]
            text_commands_val_f = text_commands_val[keep]

    # === (2) Build final (concat if ctg_condition), then shuffle policy ===
    if ctg_condition:
        states  = torch.concatenate((states_scp,  states_cvx),  axis=0)
        actions = torch.concatenate((actions_scp, actions_cvx), axis=0)
        rtgs    = torch.concatenate((rtgs_scp,    rtgs_cvx),    axis=0)
        ctgs    = torch.concatenate((ctgs_scp,    ctgs_cvx),    axis=0)

        horizons_f = np.concatenate((horizons, horizons), axis=0)
        dtime_f    = np.concatenate((dtime,    dtime),    axis=0)
        time_f     = np.concatenate((time,     time),     axis=0)
        oe_f       = np.concatenate((oe,       oe),       axis=0)
        beh_f      = np.concatenate((beh,      beh),      axis=0)
        cmd_f      = np.concatenate((cmd,      cmd),      axis=0)
        tgt_f      = np.concatenate((tgt,      tgt),      axis=0)
        goal = torch.tensor(np.repeat(tgt_f[:, None, :], states.shape[1], axis=1))
        
        if tailored_data:
            wyp = np.concatenate((data_param['waypoints'], data_param['waypoints']), axis=0)
            wyp_times = np.concatenate((data_param['waypoint_times'], data_param['waypoint_times']), axis=0)
            text_commands_f = np.concatenate((text_commands, text_commands), axis=0)
            text_commands_val_f = np.concatenate((text_commands_val, text_commands_val), axis=0)

        idx = np.random.permutation(states.shape[0]) if not equal_datasize else np.random.permutation(states.shape[0])

    else:
        states, actions, rtgs, ctgs = states_scp, actions_scp, rtgs_scp, ctgs_scp
        horizons_f, dtime_f, time_f, oe_f, beh_f, cmd_f = horizons, dtime, time, oe, beh, cmd
        goal = torch.tensor(tgt[:, None, :])
        idx = np.random.permutation(states.shape[0]) if equal_datasize else slice(None)

    torch_data = {
        'torch_states': states[idx],
        'torch_actions': actions[idx],
        'torch_rtgs': rtgs[idx],
        'torch_ctgs': ctgs[idx],
        'torch_goal': goal[idx],
    }
    data_param = {
        'horizons':  horizons_f[idx],
        'time_discr': dtime_f[idx],
        'time_sec':  time_f[idx],
        'oe':        oe_f[idx],
        'behavior':  beh_f[idx],
        'command_id': cmd_f[idx],
        'target_state': tgt_f[idx],
    }
    
    if tailored_data:
        data_param['waypoints'] = wyp[idx]
        data_param['waypoint_times'] = wyp_times[idx]
        data_param['text_command'] = text_commands_f[idx]
        data_param['text_command_val'] = text_commands_val_f[idx]

    return torch_data, data_param

def normalize(data, timestep_norm):
    # Normalize and return normalized data, mean and std
    if timestep_norm:
        data_mean = data.mean(dim=0)
        data_std = data.std(dim=0)
        data_norm = (data - data_mean)/(data_std + 1e-6)
    else:
        time_length, size_data = data.shape[1:]
        data_mean = torch.ones((time_length, size_data)) * data.view(-1,size_data).mean(dim=0)
        data_std = torch.ones((time_length, size_data)) * data.view(-1,size_data).std(dim=0)
        data_norm = (data - data_mean)/(data_std + 1e-6)

    return data_norm, data_mean, data_std

def get_DT_model(model_name, train_loader, eval_loader):
    # DT model creation
    config = DecisionTransformerConfig(
        state_dim=train_loader.dataset.n_state, 
        act_dim=train_loader.dataset.n_action,
        hidden_size=384,
        max_ep_len=50,
        vocab_size=1,
        action_tanh=False,
        n_positions=1024,
        n_layer=6,
        n_head=6,
        n_inner=None,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        )

    model = AutonomousRendezvousTransformerLang(config)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT size: {model_size/1000**2:.1f}M parameters")
    model.to(device)

    # DT optimizer and accelerator
    optimizer = AdamW(model.parameters(), lr=3e-5)
    accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )
    accelerator.load_state(root_folder + '/rpod/decision_transformer/saved_files/checkpoints/' + model_name)

    return model.eval()

def torch_model_inference_dyn(model, data_stats:dict, text_encoder, prob:NonConvexOCP, text_command:str=None, include_final=False):
    
    # Get statistics from the dataset
    data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
    data_stats['states_std'] = data_stats['states_std'].float().to(device)
    data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
    data_stats['actions_std'] = data_stats['actions_std'].float().to(device)
    
    n_state,  n_action, n_time = prob.nx, prob.nu, prob.n_time
    
    commands_emb_i = text_encoder([text_command])
    
    # using truely-randomized state_i based on sample_reset_condition() -> loaded to prob object
    roe_0 = torch.from_numpy(prob.state_init.reshape(1,1,6)).float().to(device)
    states_i = (roe_0 - data_stats['states_mean'][0]) / (data_stats['states_std'][0] + 1e-6) 
    
    # manually set up timesteps and attention mask
    timesteps_i = torch.tensor([[i for i in range(n_time)] for _ in [1]]).view(n_time).long().unsqueeze(0).to(device)
    attention_mask_i = torch.ones(1, n_time).view(n_time).long().unsqueeze(0).to(device)
    
    stm = torch.from_numpy(prob.stm).float().to(device)
    cim = torch.from_numpy(prob.cim).float().to(device)
    psi = torch.from_numpy(prob.psi).float().to(device)

    # Retrieve decoded states and actions for different inference cases
    if include_final:
        roe_dyn     = torch.empty(size=(n_time+1, n_state), device=device).float()
        rtn_dyn     = torch.empty(size=(n_time+1, n_state), device=device).float()
    else:
        roe_dyn     = torch.empty(size=(n_time, n_state), device=device).float()
        rtn_dyn     = torch.empty(size=(n_time, n_state), device=device).float()
    dv_dyn      = torch.empty(size=(n_time, n_action), device=device).float()
    states_dyn  = torch.empty(size=(1, n_time, n_state), device=device).float()
    actions_dyn = torch.zeros(size=(1, n_time, n_action), device=device).float()
    # rtgs_dyn    = torch.empty(size=(1, n_time, 1), device=device).float()
    ctgs_dyn    = torch.zeros(size=(1, n_time, 1), device=device).float()  # CTG must be always zero

    t0 = time.time()
    # Dynamics-in-the-loop initialization
    states_dyn[:,0,:] = states_i[:,0,:]

    # state representation is always ROE 
    roe_dyn[0] = roe_0 # states_dyn[:,0,:] * (data_stats['states_std'][0]+1e-6) + data_stats['states_mean'][0]
    rtn_dyn[0] = psi[0] @ roe_dyn[0]
    
    # For loop trajectory generation
    for t in np.arange(n_time):
        
        ##### Dynamics inference        
        # Compute action pred for dynamics model
        with torch.no_grad():
            output_dyn = model(
                states=states_dyn[:,:t+1,:],
                actions=actions_dyn[:,:t+1,:],
                constraints=ctgs_dyn[:,:t+1,:],
                commands_emb=commands_emb_i,
                timesteps=timesteps_i[:,:t+1],
                attention_mask=attention_mask_i[:,:t+1],
                return_dict=False,
            )
            (_, action_preds_dyn) = output_dyn

        action_dyn_t = action_preds_dyn[0,t]
        actions_dyn[:,t,:] = action_dyn_t
        dv_dyn[t] = (action_dyn_t * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]

        # Dynamics propagation of state variable 
        if t != n_time-1:
            roe_dyn[t+1] = stm[t] @ (roe_dyn[t] + cim[t] @ dv_dyn[t])
            rtn_dyn[t+1] = psi[t+1] @ roe_dyn[t+1]
            states_dyn_norm = (roe_dyn[t+1] - data_stats['states_mean'][t+1]) / (data_stats['states_std'][t+1] + 1e-6)
            states_dyn[:,t+1,:] = states_dyn_norm
            
            # we do not need to propagate CTG since it is always zero
            # ctgs_dyn[:,t+1,:] = prob.propagate_ctg(ctgs_dyn[:,t,:], roe_dyn, dv_dyn, None, t, ctg_clipped=True, lib=torch)
            actions_dyn[:,t+1,:] = 0
        
    #     time_orb[:, t] = time_sec[:, t]/period_ref
    # time_orb[:, n_time] = time_orb[:, n_time-1] + dt/period_ref

    if include_final:
        roe_dyn[-1] = roe_dyn[n_time-1] + cim[n_time-1] @ dv_dyn[n_time-1]
        rtn_dyn[-1] = psi[n_time-1] @ roe_dyn[-1]

    # Pack trajectory's data in a dictionary and compute runtime
    runtime_DT = time.time() - t0
    DT_trajectory = {
        'rtn' : rtn_dyn.cpu().numpy(),
        'roe' : roe_dyn.cpu().numpy(),
        'dv' : dv_dyn.cpu().numpy(),
        # 'time_orb' : time_orb
    }

    return {"traj": DT_trajectory, "runtime": runtime_DT}

