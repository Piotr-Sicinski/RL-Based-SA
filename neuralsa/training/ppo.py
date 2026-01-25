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
        advantages = advantages - advantages.mean() / (advantages.std() + 1e-8)
    advantages = advantages.view(n_problems * nt, -1)
    rewards_to_go = rewards_to_go.view(n_problems * nt, -1)

    actor_loss, critic_loss = None, None
    for _ in range(ppo_epochs):
        actor_opt.zero_grad()
        critic_opt.zero_grad()
        if nt > 1:  # Avoid instabilities
            # Shuffle the trajectory, good for training
            perm = np.arange(state.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)
            state = state[perm].clone()
            action = action[perm].clone()
            rewards_to_go = rewards_to_go[perm, :].clone()
            advantages = advantages[perm, :].clone()
            old_log_probs = old_log_probs[perm, :].clone()
            # Run batch optimization
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
                # Evaluate the actor
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
                actor_opt.step()
                critic_opt.step()
    return actor_loss.item(), critic_loss.item()
