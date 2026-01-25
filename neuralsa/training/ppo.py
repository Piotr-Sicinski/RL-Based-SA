# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer

from neuralsa.model import SAModel
from neuralsa.training.replay import Replay, Transition
from neuralsa.utils import extend_to


def ppo(
    actor: SAModel,
    critic: nn.Module,
    replay: Replay,
    actor_opt: Optimizer,
    critic_opt: Optimizer,
    cfg: DictConfig,
    criterion=torch.nn.MSELoss(),
) -> Tuple[float, float]:
    """
    Optimises the actor and the critic in PPO for 'ppo_epochs' epochs using the transitions
    recorded in 'replay'.

    Parameters
    ----------
    actor, critic: nn.Module
    replay: Replay object (see replay.py)
    actor_opt, critic_opt: torch.optim
    cfg: OmegaConf DictConfig
        Config containing PPO hyperparameters (see below).
    criterion: torch loss
        Loss function for the critic.

    Returns
    -------
    actor_loss, critic_loss
    """

    # PPO hyper-parameters
    ppo_epochs = cfg.training.ppo_epochs
    trace_decay = cfg.training.trace_decay
    eps_clip = cfg.training.eps_clip
    batch_size = cfg.training.batch_size
    n_problems = cfg.n_problems
    problem_dim = cfg.problem_dim
    device = cfg.device

    actor.train()
    critic.train()
    
    # Detect if actor uses LSTM
    use_lstm = hasattr(actor, 'lstm')
    
    # Get transitions
    with torch.no_grad():
        transitions = replay.memory
        nt = len(transitions)
        # Gather transition information into tensors
        batch = Transition(*zip(*transitions))
        
        # Detect if we have flat states (Rosenbrock) or per-item states (Knapsack, BinPacking, TSP)
        first_state = batch.state[0]
        is_flat_state = (first_state.dim() == 2)  # [batch, features] vs [batch, problem_dim, features]
        
        if is_flat_state:
            # Flat state structure (Rosenbrock - continuous optimization)
            # States: [nt, n_problems, features] -> [nt*n_problems, features]
            # Actions: [nt, n_problems, action_dim] -> [nt*n_problems, action_dim]
            state = torch.stack(batch.state).reshape(nt * n_problems, -1).to(device)
            action = torch.stack(batch.action).detach().reshape(nt * n_problems, -1)
            next_state = torch.stack(batch.next_state).detach().reshape(nt * n_problems, -1).to(device)
        else:
            # Per-item state structure (Knapsack, BinPacking, TSP - discrete problems)
            # States: [nt, n_problems, problem_dim, features] -> [nt*n_problems, problem_dim, features]
            # Actions: [nt, n_problems, problem_dim] -> [nt*n_problems, problem_dim] (one-hot or indices)
            state = torch.stack(batch.state).reshape(nt * n_problems, problem_dim, -1).to(device)
            action_stacked = torch.stack(batch.action).detach()
            if action_stacked.shape[-1] == 2:
                # BinPacking: actions are [nt, n_problems, 2] -> reshape to [nt * n_problems, 2]
                action = action_stacked.reshape(nt * n_problems, 2)
            else:
                # Knapsack: actions are [nt, n_problems, problem_dim] -> reshape to [nt * n_problems, problem_dim]
                action = action_stacked.reshape(nt * n_problems, problem_dim)
            next_state = torch.stack(batch.next_state).detach().reshape(nt * n_problems, problem_dim, -1).to(device)
        
        old_log_probs = torch.stack(batch.old_log_probs).view(nt * n_problems, -1)
        
        # Extract and prepare hidden states for LSTM models
        if use_lstm:
            # batch.hidden is a list of hidden states (each is tuple of (h, c) or None)
            # We need to stack them properly
            hidden_list = batch.hidden
            if hidden_list[0] is not None:
                # Stack hidden states: each hidden is (h, c) tuple
                # h and c have shape [num_layers, batch, hidden_dim]
                h_list = [h[0] for h in hidden_list]  # Extract h from each tuple
                c_list = [h[1] for h in hidden_list]  # Extract c from each tuple
                # Stack along the batch dimension (dimension 1)
                stacked_h = torch.cat(h_list, dim=1)  # [num_layers, nt*n_problems, hidden_dim]
                stacked_c = torch.cat(c_list, dim=1)  # [num_layers, nt*n_problems, hidden_dim]
                batch_hidden = (stacked_h, stacked_c)
            else:
                # No hidden states were stored (shouldn't happen with LSTM but handle gracefully)
                batch_hidden = None
        else:
            batch_hidden = None
        
        # Evaluate the critic
        state_values = critic(state).view(nt, n_problems, 1)
        next_state_values = critic(next_state).view(nt, n_problems, 1)
        # Get rewards and advantage estimate
        rewards_to_go = torch.zeros((nt, n_problems, 1), device=device, dtype=torch.float32)
        advantages = torch.zeros((nt, n_problems, 1), device=device, dtype=torch.float32)
        discounted_reward = torch.zeros((n_problems, 1), device=device)
        advantage = torch.zeros((n_problems, 1), device=device)
        # Loop through the batch transitions starting from the end of the episode
        # Compute discounted rewards, and advantage using td error
        for i, reward, gamma in zip(
            reversed(np.arange(len(transitions))), reversed(batch.reward), reversed(batch.gamma)
        ):
            if gamma == 0:
                discounted_reward = torch.zeros((n_problems, 1), device=device)
                advantage = torch.zeros((n_problems, 1), device=device)
            discounted_reward = reward + (gamma * discounted_reward)
            td_error = reward + gamma * next_state_values[i, ...] - state_values[i, ...]
            advantage = td_error + gamma * trace_decay * advantage
            rewards_to_go[i, ...] = discounted_reward
            advantages[i, ...] = advantage
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.view(n_problems * nt, -1)
    rewards_to_go = rewards_to_go.view(n_problems * nt, -1)

    actor_loss, critic_loss = None, None
    for _ in range(ppo_epochs):
        actor_opt.zero_grad()
        critic_opt.zero_grad()
        if nt > 1:  # Avoid instabilities
            # IMPORTANT: Do NOT shuffle when using LSTM
            # LSTMs need temporal order to learn sequential dependencies
            # Paper section 3.5: "training requires inputting the entire SA rollout up to a given time step"
            if not use_lstm:
                # Shuffle the trajectory for MLP models (good for training stability)
                perm = np.arange(state.shape[0])
                np.random.shuffle(perm)
                perm = torch.LongTensor(perm).to(device)
                state = state[perm].clone()
                action = action[perm].clone()
                rewards_to_go = rewards_to_go[perm, :].clone()
                advantages = advantages[perm, :].clone()
                old_log_probs = old_log_probs[perm, :].clone()
            
            # Run batch optimization
            if use_lstm:
                # For LSTM: Process entire trajectory sequentially (no mini-batches)
                # This maintains temporal dependencies crucial for LSTM learning
                # Paper section 3.5: LSTM needs "entire SA rollout up to a given time step"
                
                # We need to reshape data back to [nt, n_problems, ...] for sequential processing
                # Then process each problem independently through its temporal sequence
                
                # Reshape states and actions back to temporal structure
                if is_flat_state:
                    state_seq = state.view(nt, n_problems, -1)  # [nt, n_problems, features]
                    action_seq = action.view(nt, n_problems, -1)  # [nt, n_problems, action_dim]
                else:
                    state_seq = state.view(nt, n_problems, problem_dim, -1)  # [nt, n_problems, problem_dim, features]
                    # For Knapsack: action has shape [nt*n_problems, problem_dim] (one-hot)
                    action_seq = action.view(nt, n_problems, problem_dim)  # [nt, n_problems, problem_dim]
                
                advantages_seq = advantages.view(nt, n_problems)  # [nt, n_problems]
                rewards_to_go_seq = rewards_to_go.view(nt, n_problems)  # [nt, n_problems]
                old_log_probs_seq = old_log_probs.view(nt, n_problems)  # [nt, n_problems]
                
                # Process each problem's trajectory sequentially through LSTM
                batch_log_probs_list = []
                batch_state_values_list = []
                
                # Initialize hidden state for the batch
                hidden_lstm = None
                
                for t in range(nt):
                    # Get states and actions at time t for all problems
                    state_t = state_seq[t]  # [n_problems, ...] 
                    action_t = action_seq[t]  # [n_problems, ...]
                    
                    # Evaluate critic
                    state_values_t = critic(state_t)  # [n_problems]
                    batch_state_values_list.append(state_values_t)
                    
                    # Evaluate actor with LSTM (forward pass)
                    if is_flat_state:
                        # Continuous action (Gaussian) - Rosenbrock with LSTM
                        # For Rosenbrock RLBSA, forward() returns log_var, not logits
                        # We need to call evaluate() which internally calls forward()
                        log_probs_t = actor.evaluate(state_t, action_t, hidden=hidden_lstm)
                        # Update hidden state from the forward pass
                        # We need to manually update it by calling forward to get the new hidden
                        _, hidden_lstm = actor.forward(state_t, hidden_lstm)
                    else:
                        # Discrete action - compute log probs from logits
                        # For Knapsack, forward() returns logits
                        logits_t, hidden_lstm = actor.forward(state_t, hidden_lstm)
                        
                        # Need to apply masking like in sample()
                        n_probs = state_t.shape[0]
                        x = state_t[..., 0]
                        weights = state_t[..., 1]
                        capacity = state_t[..., 3]
                        
                        total_weight = torch.sum(weights * x, -1)
                        free_space = capacity - extend_to(total_weight, capacity)
                        oversized = (weights > free_space) * (x == 0)
                        mask = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
                        
                        logits_masked = logits_t + mask
                        log_probs_all = torch.log_softmax(logits_masked, dim=-1)
                        
                        # Extract log prob for taken action
                        # action_t is one-hot encoded with shape [n_problems, problem_dim]
                        # We need to get log_prob for each problem's selected item
                        # Convert one-hot to indices
                        action_indices = torch.argmax(action_t, dim=-1)  # [n_problems]
                        log_probs_t = log_probs_all[torch.arange(n_probs, device=log_probs_all.device), action_indices]
                    
                    batch_log_probs_list.append(log_probs_t)
                    
                    # Detach hidden state for next timestep (truncated BPTT)
                    if hidden_lstm is not None:
                        hidden_lstm = tuple(h.detach() for h in hidden_lstm)
                
                # Stack results
                batch_log_probs = torch.stack(batch_log_probs_list).view(-1)  # [nt * n_problems]
                batch_state_values = torch.stack(batch_state_values_list).view(-1)  # [nt * n_problems]
                
                # Flatten advantages and rewards
                batch_advantages = advantages_seq.view(-1)
                batch_rewards_to_go = rewards_to_go_seq.view(-1)
                batch_old_log_probs = old_log_probs_seq.view(-1)
                
                # Compute losses
                critic_loss = 0.5 * criterion(batch_state_values, batch_rewards_to_go.detach())
                ratios = torch.exp(batch_log_probs - batch_old_log_probs.detach())
                surr1 = ratios * batch_advantages.detach()
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * batch_advantages.detach()
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Optimize
                actor_loss.backward()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                actor_opt.step()
                critic_opt.step()
                
            else:
                # For MLP: Use mini-batch processing with shuffled data
                for j in range(nt * n_problems, 0, -batch_size):
                    nb = min(j, batch_size)
                    if nb <= 1:  # Avoid instabilities
                        continue
                    # Get a batch of transitions
                    batch_idx = np.arange(j - nb, j)
                    # Gather batch information into tensors
                    batch_state = state[batch_idx, ...]
                    batch_action = action[batch_idx, ...]
                    batch_advantages = advantages[batch_idx, 0]
                    batch_rewards_to_go = rewards_to_go[batch_idx, 0]
                    batch_old_log_probs = old_log_probs[batch_idx, 0]
                    # Evaluate the critic
                    batch_state_values = critic(batch_state)
                    # Evaluate the actor (MLP doesn't need hidden states)
                    batch_log_probs = actor.evaluate(batch_state, batch_action)
                    # Compute critic loss
                    critic_loss = 0.5 * criterion(
                        batch_state_values.squeeze(), batch_rewards_to_go.detach()
                    )
                    # Compute actor loss
                    ratios = torch.exp(batch_log_probs - batch_old_log_probs.detach())
                    surr1 = ratios * batch_advantages.detach()
                    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * batch_advantages.detach()
                    actor_loss = -torch.min(surr1, surr2).mean()
                    # Optimize
                    actor_loss.backward()
                    critic_loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                    actor_opt.step()
                    critic_opt.step()
    return actor_loss.item(), critic_loss.item()
