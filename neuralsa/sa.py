# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


from typing import Dict

import torch
from omegaconf import DictConfig

from neuralsa.model import SAModel
from neuralsa.problem import Problem
from neuralsa.training.replay import Replay
from neuralsa.utils import extend_to


def p_accept(gain: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
    """
    Compute acceptance probability, at temperature temp, of a move leading
    to a change in the energy function of 'gain'.
    """
    return torch.minimum(torch.exp(gain / temp), torch.ones_like(gain))


def sa(
    actor: SAModel,
    problem: Problem,
    init_x: torch.Tensor,
    cfg: DictConfig,
    baseline: bool = False,
    random_std: float = 0.2,
    greedy: bool = False,
    record_state: bool = False,
    replay: Replay = None,
) -> Dict[str, torch.Tensor]:

    device = init_x.device

    # === Init SA cfg ===
    temp = torch.tensor([cfg.sa.init_temp], device=device)
    next_temp = temp
    alpha = cfg.sa.alpha

    # === Init archive ===
    best_x = x = init_x
    min_cost = problem.cost(best_x)
    primal = min_cost
    first_cost = cost = min_cost

    n_acc, n_rej = 0, 0
    distributions, states, actions = [], [], []
    acceptance = []
    costs = [min_cost]
    reward = None

    # ============================================================
    # Determine if we're using RLBSA (uses E, ΔE) or NSA (doesn't)
    # ============================================================
    # RLBSA models have 'RLBSA' in their class name
    use_rlbsa = 'RLBSA' in actor.__class__.__name__
    use_delta_e = use_rlbsa  # RLBSA uses delta_e and current_energy, NSA doesn't
    
    # Check if actor has LSTM (for recurrent hidden state management)
    use_lstm = hasattr(actor, "lstm")
    if use_lstm:
        B = init_x.shape[0]
        H = actor.embed_dim
        num_layers = actor.lstm.num_layers
        hidden_actor = (
            torch.zeros(num_layers, B, H, device=device),
            torch.zeros(num_layers, B, H, device=device),
        )
    else:
        hidden_actor = None

    # === Initial ΔE (energy change from previous step) and E (current energy) ===
    # Start with zero since there's no previous step
    # Only use delta_e and current_energy for RLBSA
    if use_delta_e:
        delta_e = torch.zeros((init_x.shape[0], 1), device=device)
        # For RLBSA: include current energy E in state [x, w, v, W, E, ΔE, T]
        current_energy = cost.unsqueeze(-1)  # [batch, 1]
        state = problem.to_state(x, temp, delta_e=delta_e, current_energy=current_energy).to(device)
    else:
        delta_e = None
        current_energy = None
        state = problem.to_state(x, temp).to(device)
    
    next_state = state

    # ============================================================
    # SA MAIN LOOP
    # ============================================================
    for _ in range(cfg.sa.outer_steps):
        for j in range(cfg.sa.inner_steps):

            if record_state:
                states.append(state)

            # ----------------------------------------------------
            # Sample action
            # ----------------------------------------------------
            if baseline:
                action, old_log_probs = actor.baseline_sample(
                    state, random_std=random_std, problem=problem
                )
            else:
                if hidden_actor is None:
                    sample_result = actor.sample(state, greedy=greedy)
                    if len(sample_result) == 4:
                        action, old_log_probs, _, _ = sample_result
                    elif len(sample_result) == 3:
                        action, old_log_probs, _ = sample_result
                    else:
                        action, old_log_probs = sample_result
                else:
                    action, old_log_probs, hidden_actor = actor.sample(
                        state, hidden_actor, greedy=greedy
                    )

            if record_state:
                logits = actor.get_logits(state, action)
                distributions.append(logits)
                actions.append(action)

            # ----------------------------------------------------
            # Proposal
            # ----------------------------------------------------
            x, _, _ = problem.from_state(state)
            proposal = problem.update(x, action)

            # ----------------------------------------------------
            # Energy difference
            # ----------------------------------------------------
            proposal_cost = problem.cost(proposal)
            gain = cost - proposal_cost   # = -ΔE (positive gain = good move)

            # ----------------------------------------------------
            # Metropolis acceptance
            # ----------------------------------------------------
            p_acceptance = p_accept(gain, temp)
            u = torch.rand(p_acceptance.shape, device=device)
            accept = (u < p_acceptance).float()
            realized_gain = gain * accept

            n_acc += accept
            n_rej += 1 - accept
            if record_state:
                acceptance.append(accept)

            # ----------------------------------------------------
            # State & cost update
            # ----------------------------------------------------
            prev_cost = cost  # Save for ΔE calculation
            cost = accept * proposal_cost + (1 - accept) * cost
            accept_x = extend_to(accept, x)
            next_x = accept_x * proposal + (1 - accept_x) * x

            # ----------------------------------------------------
            # Archive update (FIXED)
            # ----------------------------------------------------
            if record_state:
                costs.append(cost)

            new_best = (cost < min_cost).float()
            new_best_x = extend_to(new_best, next_x)
            best_x = new_best_x * next_x + (1 - new_best_x) * best_x

            min_cost = torch.minimum(cost, min_cost)
            primal = primal + min_cost

            # ----------------------------------------------------
            # Temperature schedule
            # ----------------------------------------------------
            if j == cfg.sa.inner_steps - 1:
                next_temp = temp * alpha
            else:
                next_temp = temp

            # ----------------------------------------------------
            # Next state (with E and ΔE if using RLBSA)
            # ----------------------------------------------------
            if use_delta_e:
                # For RLBSA: state includes current energy E and delta E
                # State format: [x, a, b, E, ΔE, T] (7 features for 2D Rosenbrock)
                # 
                # IMPORTANT: ΔE should be the ACTUAL energy change after acceptance,
                # not the proposed energy change. If move was rejected, ΔE = 0.
                actual_delta_e = (accept * (proposal_cost - prev_cost)).unsqueeze(-1)
                current_energy = cost.unsqueeze(-1)  # [batch, 1]
                next_state = problem.to_state(next_x, next_temp, delta_e=actual_delta_e, 
                                              current_energy=current_energy)
            else:
                next_state = problem.to_state(next_x, next_temp)

            # ----------------------------------------------------
            # PPO reward
            # ----------------------------------------------------
            if cfg.training.method == "ppo":
                if cfg.training.reward == "immediate":
                    reward = realized_gain.unsqueeze(1)
                elif cfg.training.reward == "min_cost":
                    reward = -min_cost.view(-1, 1)
                elif cfg.training.reward == "primal":
                    reward = -primal.view(-1, 1)
                else:
                    raise NotImplementedError

            # ----------------------------------------------------
            # Replay + TBPTT
            # ----------------------------------------------------
            if replay is not None:
                if hidden_actor is not None:
                    hidden_actor = tuple(h.detach() for h in hidden_actor)

                replay.push(
                    state,
                    action,
                    next_state,
                    reward,
                    old_log_probs,
                    cfg.training.gamma,
                )

        # Move forward
        state = next_state.clone()
        temp = next_temp

    # ============================================================
    # Final bookkeeping
    # ============================================================
    ngain = -(first_cost - cost)

    if replay is not None and len(replay) > 0:
        last_transition = replay.pop()
        replay.push(*(list(last_transition[:-1]) + [0.0]))

    return {
        "best_x": best_x,
        "min_cost": min_cost,
        "primal": primal,
        "ngain": ngain,
        "n_acc": n_acc,
        "n_rej": n_rej,
        "distributions": distributions,
        "states": states,
        "actions": actions,
        "acceptance": acceptance,
        "costs": costs,
    }
