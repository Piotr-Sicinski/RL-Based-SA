# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralsa.utils import extend_to, repeat_to


class SAModel(nn.Module):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.generator = torch.Generator(device=device)

    def manual_seed(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def get_logits(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, state: torch.Tensor, greedy: bool = False, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def baseline_sample(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


# ============================================================
# KNAPSACK - NSA (Original Neural SA without RL enhancements)
# ============================================================

class KnapsackActorNSA(SAModel):
    """Original NSA actor for Knapsack - simple MLP without LSTM or delta_e
    
    According to paper Section 4.1.1 (line 493-498), the original Neural SA
    uses a two-layer MLP: 5 → 16 → 1 with 112 learnable parameters.
    Input: [x_i, w_i, v_i, W, T]
    """
    def __init__(self, problem, embed_dim: int, device: str = "cpu") -> None:
        super().__init__(device)
        self.problem = problem
        self.embed_dim = embed_dim
        
        # State features per item: x(1) + weights(1) + values(1) + capacity(1) + temp(1)
        self.state_dim = 5
        
        # Simple MLP (using 'embed' to match original architecture)
        self.embed = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, device=device),
        )
        
        self.embed.apply(self.init_weights)
    
    def sample(
        self, state: torch.Tensor, greedy: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        state: [batch, problem_dim, 5] with format [x, w, v, W, T]
        returns: action [batch, problem_dim], log_probs [batch]
        """
        # Extract state components
        n_problems, problem_dim, _ = state.shape
        x = state[..., 0]
        weights = state[..., 1]
        capacity = state[..., 3]  # Raw capacity W (not normalized)
        
        # Compute mask to avoid exceeding the knapsack's capacity
        # by selecting too heavy items
        total_weight = torch.sum(weights * x, -1)
        free_space = capacity - extend_to(total_weight, capacity)
        oversized = (weights > free_space) * (x == 0)
        mask = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        
        # Compute logits and apply mask
        logits = self.embed(state)[..., 0] + mask
        probs = torch.softmax(logits, dim=-1)
        
        if greedy:
            smpl = torch.argmax(probs, dim=-1)
        else:
            smpl = torch.multinomial(probs, 1, generator=self.generator)[..., 0]
        
        action = F.one_hot(smpl, num_classes=problem_dim)
        log_probs = torch.log(probs[action.bool()])
        
        # Return [batch, dim, 1] to match vanilla behavior
        return action[..., None], log_probs
    
    def baseline_sample(self, state: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # uniform random action - sample one item to flip, respecting capacity constraints
        n_problems, problem_dim, _ = state.shape
        x = state[..., 0]
        weights = state[..., 1]
        capacity = state[..., 3]  # Raw capacity W (not normalized)
        
        # Compute mask to avoid exceeding the knapsack's capacity by selecting too heavy items
        total_weight = torch.sum(weights * x, -1)
        free_space = capacity - extend_to(total_weight, capacity)
        oversized = (weights > free_space) * (x == 0)
        mask = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        
        logits = torch.ones((state.shape[0], problem_dim), device=state.device) + mask
        probs = torch.softmax(logits, dim=-1)
        
        smpl = torch.multinomial(probs, 1, generator=self.generator)[..., 0]
        action = F.one_hot(smpl, num_classes=problem_dim)
        log_probs = torch.log(probs[action.bool()])
        
        # Return [batch, dim, 1] to match vanilla behavior
        return action[..., None], log_probs
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor, **kwargs):
        """
        state: [batch, problem_dim, 5]
        action: [batch, problem_dim] one-hot encoded
        """
        # Extract state components
        n_problems, problem_dim, _ = state.shape
        x = state[..., 0]
        weights = state[..., 1]
        capacity = state[..., 3]  # Raw capacity W (not normalized)
        
        # Compute mask to avoid exceeding the knapsack's capacity
        # by selecting too heavy items
        total_weight = torch.sum(weights * x, -1)
        free_space = capacity - extend_to(total_weight, capacity)
        oversized = (weights > free_space) * (x == 0)
        mask = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        
        # Get logits and compute log-probabilities of the action taken
        logits = self.embed(state)[..., 0] + mask
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs[action.bool()]


class KnapsackCriticNSA(nn.Module):
    """Original NSA critic for Knapsack"""
    def __init__(self, problem, embed_dim: int, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.problem = problem
        self.state_dim = 5  # x + weight + value + capacity + temp
        self.embed_dim = embed_dim
        
        self.embed = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, device=device),
        )
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.embed:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: [batch, problem_dim, 5]
        returns: value estimate [batch]
        """
        values = self.embed(state[..., :5])  # [batch, problem_dim, 1]
        return values.mean(dim=1).squeeze(-1)  # [batch]


# ============================================================
# KNAPSACK - RLBSA (RL-Based SA with LSTM and delta_e)
# ============================================================

class KnapsackActorRLBSA(SAModel):
    """RLBSA actor for Knapsack with LSTM and delta_e support
    
    According to paper Section 4.1.1 (line 504-510), the LSTM architecture
    uses 7 input features: [x_i, w_i, v_i, W, E, ΔE, T]
    """
    def __init__(self, problem, embed_dim: int, device: str = "cpu") -> None:
        super().__init__(device)
        self.problem = problem
        self.embed_dim = embed_dim

        # State features per item: x(1) + w(1) + v(1) + W(1) + E(1) + ΔE(1) + T(1)
        # Paper line 504-506: "three-layer network 7 → 16 → 16 → 1"
        self.state_dim_min = 5  # x + weight + value + capacity + temp (NSA)
        self.state_dim_max = 7  # + energy + delta_e (RLBSA)

        # Input projection before LSTM
        self.input_proj = nn.Linear(self.state_dim_max, embed_dim, device=device)
        # LSTM processes sequence of items
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, device=device)
        # Output layer produces logit per item
        self.output_layer = nn.Linear(embed_dim, 1, device=device)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.input_proj.bias is not None:
            self.input_proj.bias.data.fill_(0.01)
        if self.output_layer.bias is not None:
            self.output_layer.bias.data.fill_(0.01)

    def forward(
        self, state: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        state: [batch, problem_dim, features] where features can be 5, 6, or 7
               - 5: [x, w, v, W, T] (NSA)
               - 6: [x, w, v, W, T, ΔE] (intermediate)
               - 7: [x, w, v, W, E, ΔE, T] (RLBSA - paper format)
        hidden: LSTM hidden state (h, c)
        returns: logits [batch, problem_dim], hidden
        """
        batch_size, seq_len, features = state.shape
        
        # Pad to max features if necessary
        if features < self.state_dim_max:
            padding = torch.zeros(batch_size, seq_len, self.state_dim_max - features, 
                                 device=state.device, dtype=state.dtype)
            state = torch.cat([state, padding], dim=-1)
        
        x = self.input_proj(state)  # [batch, problem_dim, embed_dim]

        if hidden is None:
            h0 = torch.zeros(1, batch_size, self.embed_dim, device=self.device)
            c0 = torch.zeros(1, batch_size, self.embed_dim, device=self.device)
            hidden = (h0, c0)

        out, hidden = self.lstm(x, hidden)  # out: [batch, problem_dim, embed_dim]
        logits = self.output_layer(out).squeeze(-1)  # [batch, problem_dim]
        return logits, hidden

    def sample(
        self,
        state: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Extract state components for masking
        # State format: [x, w, v, W, ...] where W is at index 3
        n_problems, problem_dim, _ = state.shape
        x = state[..., 0]  # Current solution
        weights = state[..., 1]  # Item weights
        capacity = state[..., 3]  # Raw capacity W (no longer normalized)
        
        # Compute mask to avoid exceeding the knapsack's capacity
        # by selecting too heavy items
        total_weight = torch.sum(weights * x, -1)
        free_space = capacity - extend_to(total_weight, capacity)
        oversized = (weights > free_space) * (x == 0)
        mask = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        
        logits, hidden = self.forward(state, hidden)
        logits = logits + mask
        probs = torch.softmax(logits, dim=-1)

        if greedy:
            smpl = torch.argmax(probs, dim=-1)
        else:
            smpl = torch.multinomial(probs, 1, generator=self.generator)[..., 0]

        action = F.one_hot(smpl, num_classes=self.problem.dim)
        log_probs = torch.log(probs[action.bool()])

        return action, log_probs, hidden

    def baseline_sample(self, state: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # uniform random action - sample one item to flip, respecting capacity constraints
        n_problems, problem_dim, _ = state.shape
        x = state[..., 0]
        weights = state[..., 1]
        capacity = state[..., 3]  # Raw capacity W (no longer normalized)
        
        # Compute mask to avoid exceeding the knapsack's capacity by selecting too heavy items
        total_weight = torch.sum(weights * x, -1)
        free_space = capacity - extend_to(total_weight, capacity)
        oversized = (weights > free_space) * (x == 0)
        mask = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        
        logits = torch.ones((state.shape[0], problem_dim), device=state.device) + mask
        probs = torch.softmax(logits, dim=-1)
        
        smpl = torch.multinomial(probs, 1, generator=self.generator)[..., 0]
        action = F.one_hot(smpl, num_classes=problem_dim)
        log_probs = torch.log(probs[action.bool()])
        
        return action, log_probs

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, hidden=None):
        """
        state: [batch, problem_dim, features]
        action: [batch, problem_dim] one-hot encoded
        """
        # Extract state components for masking
        n_problems, problem_dim, _ = state.shape
        x = state[..., 0]
        weights = state[..., 1]
        capacity = state[..., 3]  # Raw capacity W (no longer normalized)
        
        # Compute mask to avoid exceeding the knapsack's capacity
        # by selecting too heavy items
        total_weight = torch.sum(weights * x, -1)
        free_space = capacity - extend_to(total_weight, capacity)
        oversized = (weights > free_space) * (x == 0)
        mask = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        
        logits, _ = self.forward(state, hidden)
        logits = logits + mask
        log_probs = torch.log_softmax(logits, dim=-1)
        # Get log prob of the selected action
        return log_probs[action.bool()]


class KnapsackCriticRLBSA(nn.Module):
    """RLBSA critic for Knapsack with delta_e support
    
    According to paper Section 4.1.1, should match actor's 7-feature input:
    [x, w, v, W, E, ΔE, T]
    """
    def __init__(self, problem, embed_dim: int, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.problem = problem
        self.state_dim_max = 7  # Match actor: x + weight + value + capacity + energy + delta_e + temp
        self.embed_dim = embed_dim

        # MLP critic processes per-item features
        self.embed = nn.Sequential(
            nn.Linear(self.state_dim_max, embed_dim, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, device=device),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.embed:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: [batch, problem_dim, features] where features can be 5, 6, or 7
               - 5: [x, w, v, W, T] (NSA)
               - 6: [x, w, v, W, T, ΔE] (intermediate)
               - 7: [x, w, v, W, E, ΔE, T] (RLBSA - paper format)
        returns: value estimate [batch]
        """
        batch_size, seq_len, features = state.shape
        
        # Pad to max features if necessary
        if features < self.state_dim_max:
            padding = torch.zeros(batch_size, seq_len, self.state_dim_max - features, 
                                 device=state.device, dtype=state.dtype)
            state = torch.cat([state, padding], dim=-1)
        
        # Process each item and aggregate
        values = self.embed(state)  # [batch, problem_dim, 1]
        return values.mean(dim=1).squeeze(-1)  # [batch]


# ============================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================
# Default to NSA for backward compatibility
KnapsackActor = KnapsackActorNSA
KnapsackCritic = KnapsackCriticNSA




# ============================================================
# BINPACKING - NSA (Original Neural SA)
# ============================================================

class BinPackingActorNSA(SAModel):
    def __init__(self, embed_dim: int, device: str) -> None:
        super().__init__(device)
        self.state_dim = 3

        # Mean and std computation
        self.embed_item = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=True, device=device),
        )
        # Mean and std computation
        self.embed_bin = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=True, device=device),
        )

        self.embed_item.apply(self.init_weights)
        self.embed_bin.apply(self.init_weights)

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_problems, problem_dim, _ = state.shape
        # Convert state -> x
        x = state[..., [0]]
        item_weights = state[..., [1]]
        free_capacity = state[..., [2]]
        temp = state[..., [3]]

        # Gather actions (item and bin)
        item, bin = action[:, 0], action[:, 1]

        # Get free capacity of each item's current bin
        fci = free_capacity.gather(1, x.long())

        # Build item state representation
        item_state = torch.cat([item_weights, fci, temp], -1)

        # Get item probabilities
        item_logits = self.embed_item(item_state)[..., 0]
        probs = torch.softmax(item_logits, dim=-1)
        log_probs_item = torch.log(probs.gather(1, item.view(-1, 1)))

        # Gather the weight of selected item
        item_weight = item_weights[..., 0].gather(1, item.view(-1, 1))
        item_weight = repeat_to(item_weight, free_capacity)

        # Build bin state representation
        bin_state = torch.cat([item_weight, free_capacity, temp], -1)

        # Get bin mask
        # too_big = 1 if weight > capacity
        oversized = (item_weight[..., 0] - free_capacity[..., 0]) > 0
        # Place -inf where oversized
        oversized = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        # Place -inf where item is (to avoid useless move)
        mask = torch.scatter(oversized, -1, item[..., None], torch.finfo(torch.float32).min)

        # Get bin probabilities
        bin_logits = self.embed_bin(bin_state)[..., 0]
        bin_logits = bin_logits + mask
        probs = torch.softmax(bin_logits, dim=-1)
        log_probs_bin = torch.log(probs.gather(1, bin.view(-1, 1)))

        return log_probs_item, log_probs_bin

    def sample_from_logits(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor = None,
        one_hot: bool = True,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_problems, problem_dim = logits.shape
        if mask is not None:
            logits = logits + mask

        probs = torch.softmax(logits, dim=-1)
        if greedy:
            smpl = torch.argmax(probs, -1, keepdim=False)
        else:
            smpl = torch.multinomial(probs, 1, generator=self.generator)[..., 0]

        if one_hot:
            smpl = F.one_hot(smpl, num_classes=problem_dim)[..., None]
        return smpl, torch.log(probs.gather(1, smpl.view(-1, 1)))

    def sample(
        self, state: torch.Tensor, greedy: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_problems, problem_dim, _ = state.shape
        # Convert state -> x
        x = state[..., [0]]
        item_weights = state[..., [1]]
        free_capacity = state[..., [2]]
        temp = state[..., [3]]

        # Get free capacity of each item's current bin
        fci = free_capacity.gather(1, x.long())

        # Build item state representation and get logits
        item_state = torch.cat([item_weights, fci, temp], -1)

        # Select an item: [batch, num items, 1]
        item_logits = self.embed_item(item_state)[..., 0]
        item, log_probs_item = self.sample_from_logits(item_logits, one_hot=False, greedy=greedy)

        # Gather the weight of selected item
        item_weight = item_weights[..., 0].gather(1, item.view(-1, 1))
        item_weight = repeat_to(item_weight, free_capacity)

        # Build bin state representation
        bin_state = torch.cat([item_weight, free_capacity, temp], -1)

        # Get bin mask
        # too_big = 1 if weight > capacity
        oversized = (item_weight[..., 0] - free_capacity[..., 0]) > 0
        # Place -inf where oversized
        oversized = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        # Place -inf where item is (to avoid useless move)
        mask = torch.scatter(oversized, -1, item[..., None], torch.finfo(torch.float32).min)

        # Select a bin: [batch, num bins, 1]
        bin_logits = self.embed_bin(bin_state)[..., 0]
        bin, log_probs_bin = self.sample_from_logits(
            bin_logits, mask=mask, one_hot=False, greedy=greedy
        )

        # Build action tensor
        action = torch.cat((item.view(-1, 1), bin.view(-1, 1)), dim=-1)

        return action, log_probs_item + log_probs_bin

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, **kwargs) -> torch.Tensor:
        n_problems, problem_dim, _ = state.shape
        # Convert state -> x
        x = state[..., [0]]
        item_weights = state[..., [1]]
        free_capacity = state[..., [2]]
        temp = state[..., [3]]

        # Gather actions (item and bin)
        item, bin = action[:, 0], action[:, 1]

        # Get free capacity of each item's current bin
        fci = free_capacity.gather(1, x.long())

        # Build item state representation
        item_state = torch.cat([item_weights, fci, temp], -1)

        # Get item probabilities
        item_logits = self.embed_item(item_state)[..., 0]
        probs = torch.softmax(item_logits, dim=-1)
        log_probs_item = torch.log(probs.gather(1, item.view(-1, 1)))

        # Gather the weight of selected item
        item_weight = item_weights[..., 0].gather(1, item.view(-1, 1))
        item_weight = repeat_to(item_weight, free_capacity)

        # Build bin state representation
        bin_state = torch.cat([item_weight, free_capacity, temp], -1)

        # Get bin mask
        # too_big = 1 if weight > capacity
        oversized = (item_weight[..., 0] - free_capacity[..., 0]) > 0
        # Place -inf where oversized
        oversized = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        # Place -inf where item is (to avoid useless move)
        mask = torch.scatter(oversized, -1, item[..., None], torch.finfo(torch.float32).min)

        # Get bin probabilities
        bin_logits = self.embed_bin(bin_state)[..., 0]
        bin_logits = bin_logits + mask
        probs = torch.softmax(bin_logits, dim=-1)
        log_probs_bin = torch.log(probs.gather(1, bin.view(-1, 1)))

        log_probs = log_probs_item + log_probs_bin

        return log_probs[..., 0]

    def baseline_sample(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        n_problems, problem_dim, _ = state.shape
        # Convert state -> x
        item_weights = state[..., [1]]
        free_capacity = state[..., [2]]

        # Sample item
        logits = torch.ones(state.shape[:2], device=state.device)
        item, _ = self.sample_from_logits(logits, mask=None, one_hot=False)

        # Gather the weight of selected item
        item_weight = item_weights[..., 0].gather(1, item.view(-1, 1))
        item_weight = repeat_to(item_weight, free_capacity)

        # Get bin mask
        # too_big = 1 if weight > capacity
        oversized = (item_weight[..., 0] - free_capacity[..., 0]) > 0
        # Place -inf where oversized
        oversized = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        # Place -inf where item is (to avoid useless move)
        mask = torch.scatter(oversized, -1, item[..., None], torch.finfo(torch.float32).min)

        # Sample bin
        logits = torch.ones(state.shape[:2], device=state.device)
        bin, _ = self.sample_from_logits(logits, mask=mask, one_hot=False)

        return torch.cat((item.view(-1, 1), bin.view(-1, 1)), dim=-1), None


class BinPackingCriticNSA(nn.Module):
    def __init__(self, embed_dim: int, device: str = "cpu") -> None:
        super().__init__()
        self.q_func = nn.Sequential(
            nn.Linear(3, embed_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, 1, device=device),
        )

        self.q_func.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size, problem_dim, _ = state.shape
        # Use features 1-3: weight, free_capacity, temp (skip x at 0, and delta_e at 4 if present)
        q_values = self.q_func(state[..., 1:4]).view(batch_size, problem_dim)
        return q_values.mean(dim=-1)


# ============================================================
# BINPACKING - RLBSA (Not yet implemented)
# ============================================================

class BinPackingActorRLBSA(SAModel):
    """RLBSA actor for BinPacking - NOT YET IMPLEMENTED"""
    def __init__(self, problem, embed_dim: int, device: str = "cpu") -> None:
        super().__init__(device)
        self.problem = problem
        self.embed_dim = embed_dim
        
        # State features per item/bin: [feature, E, ΔE, T]
        # Item network: [w_i, c_b(i), E, ΔE, T] = 5 features
        # Bin network: [w_i, c_j, E, ΔE, T] = 5 features
        self.state_dim = 5

        # Item selection network
        self.embed_item = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=True, device=device),
        )
        
        # Bin selection network
        self.embed_bin = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=True, device=device),
        )

        self.embed_item.apply(self.init_weights)
        self.embed_bin.apply(self.init_weights)

    def sample_from_logits(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor = None,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            logits = logits + mask

        probs = torch.softmax(logits, dim=-1)
        if greedy:
            smpl = torch.argmax(probs, dim=-1)
        else:
            smpl = torch.multinomial(probs, 1, generator=self.generator)[..., 0]

        log_probs = torch.log(probs.gather(1, smpl.view(-1, 1)))
        return smpl, log_probs

    def sample(
        self,
        state: torch.Tensor,
        hidden=None,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """
        Sample action from current state.
        
        Args:
            state: [batch, problem_dim, features] where features include:
                   [x (bin assignment), w (weight), free_capacity, E, ΔE, T]
            hidden: Not used for BinPacking (MLP), kept for compatibility
            greedy: whether to use greedy selection
            
        Returns:
            action: [batch, 2] (item_idx, bin_idx)
            log_probs: [batch, 1] combined log probabilities
            None: no hidden state (compatibility with Knapsack interface)
        """
        n_problems, problem_dim, n_features = state.shape
        
        # Extract features from state
        # state format: [x, w, free_capacity, E, ΔE, T]
        x = state[..., 0]  # [batch, problem_dim] - current bin assignments
        weights = state[..., 1]  # [batch, problem_dim] - item weights
        free_capacity = state[..., 2]  # [batch, problem_dim] - free capacity per bin
        
        # E, ΔE, T are constant across items in current timestep
        if n_features >= 6:
            E = state[..., 3]  # [batch, problem_dim] - current energy
            delta_E = state[..., 4]  # [batch, problem_dim] - energy change
            temp = state[..., 5]  # [batch, problem_dim] - temperature
        elif n_features >= 5:
            E = state[..., 3]  # [batch, problem_dim]
            delta_E = torch.zeros_like(E)
            temp = state[..., 4]
        else:
            # Fallback: no E, ΔE
            E = torch.zeros(n_problems, problem_dim, device=state.device)
            delta_E = torch.zeros_like(E)
            temp = state[..., 3] if n_features >= 4 else torch.ones_like(E)

        # === ITEM SELECTION ===
        # Get free capacity of each item's current bin
        fci = free_capacity.gather(1, x.long())
        
        # Build item state: [w_i, c_b(i), E, ΔE, T] for each item
        item_state = torch.stack([weights, fci, E, delta_E, temp], dim=-1)
        
        # Get item logits through MLP
        item_logits = self.embed_item(item_state)[..., 0]
        
        # Sample item
        item, log_probs_item = self.sample_from_logits(item_logits, greedy=greedy)

        # === BIN SELECTION ===
        # Gather the weight of selected item
        item_weight = weights.gather(1, item.view(-1, 1))
        item_weight = item_weight.expand(-1, problem_dim)

        # Build bin state: [w_i, c_j, E, ΔE, T] for each bin
        bin_state = torch.stack([item_weight, free_capacity, E, delta_E, temp], dim=-1)

        # Get bin mask - prevent oversized and same-bin moves
        oversized = (item_weight - free_capacity) > 0
        oversized = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        mask = torch.scatter(oversized, 1, item.unsqueeze(1), torch.finfo(torch.float32).min)

        # Get bin logits through MLP
        bin_logits = self.embed_bin(bin_state)[..., 0]  # [batch, problem_dim]

        # Sample bin
        bin, log_probs_bin = self.sample_from_logits(bin_logits, mask=mask, greedy=greedy)

        # Build action tensor
        action = torch.cat((item.view(-1, 1), bin.view(-1, 1)), dim=-1)

        return action, log_probs_item + log_probs_bin, None

    def baseline_sample(self, state: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uniform random baseline policy."""
        n_problems, problem_dim, _ = state.shape
        
        weights = state[..., 1]
        free_capacity = state[..., 2]

        # Sample item uniformly
        item = torch.randint(0, problem_dim, (n_problems,), device=state.device, generator=self.generator)
        item_weight = weights.gather(1, item.view(-1, 1)).expand(-1, problem_dim)

        # Get bin mask
        oversized = (item_weight - free_capacity) > 0
        oversized = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        mask = torch.scatter(oversized, 1, item.unsqueeze(1), torch.finfo(torch.float32).min)

        logits = torch.zeros(n_problems, problem_dim, device=state.device)
        logits = logits + mask
        probs = torch.softmax(logits, dim=-1)
        bin = torch.multinomial(probs, 1, generator=self.generator).squeeze(1)

        # Calculate log probabilities
        log_prob_item = torch.log(torch.ones(n_problems, device=state.device) / problem_dim)
        log_prob_bin = torch.log(probs.gather(1, bin.view(-1, 1))).squeeze(-1)

        action = torch.cat((item.view(-1, 1), bin.view(-1, 1)), dim=-1)
        return action, log_prob_item + log_prob_bin

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, hidden=None) -> torch.Tensor:
        """
        Evaluate log probability of given action under current policy.
        
        Args:
            state: [batch, problem_dim, features]
            action: [batch, 2] (item_idx, bin_idx)
            hidden: Not used, kept for compatibility
            
        Returns:
            log_probs: [batch] log probabilities of actions
        """
        n_problems, problem_dim, n_features = state.shape
        
        # Extract features
        x = state[..., 0]
        weights = state[..., 1]
        free_capacity = state[..., 2]
        
        if n_features >= 6:
            E = state[..., 3]
            delta_E = state[..., 4]
            temp = state[..., 5]
        elif n_features >= 5:
            E = state[..., 3]
            delta_E = torch.zeros_like(E)
            temp = state[..., 4]
        else:
            E = torch.zeros(n_problems, problem_dim, device=state.device)
            delta_E = torch.zeros_like(E)
            temp = state[..., 3] if n_features >= 4 else torch.ones_like(E)

        item_idx = action[:, 0].long()
        bin_idx = action[:, 1].long()

        # === ITEM LOG PROB ===
        fci = free_capacity.gather(1, x.long())
        item_state = torch.stack([weights, fci, E, delta_E, temp], dim=-1)
        
        item_logits = self.embed_item(item_state)[..., 0]
        item_log_probs = torch.log_softmax(item_logits, dim=-1)
        log_probs_item = item_log_probs.gather(1, item_idx.view(-1, 1))

        # === BIN LOG PROB ===
        item_weight = weights.gather(1, item_idx.view(-1, 1)).expand(-1, problem_dim)
        bin_state = torch.stack([item_weight, free_capacity, E, delta_E, temp], dim=-1)

        # Mask
        oversized = (item_weight - free_capacity) > 0
        oversized = torch.where(oversized, torch.finfo(torch.float32).min, 0.0)
        mask = torch.scatter(oversized, 1, item_idx.unsqueeze(1), torch.finfo(torch.float32).min)

        bin_logits = self.embed_bin(bin_state)[..., 0]
        bin_logits = bin_logits + mask
        bin_log_probs = torch.log_softmax(bin_logits, dim=-1)
        log_probs_bin = bin_log_probs.gather(1, bin_idx.view(-1, 1))

        return (log_probs_item + log_probs_bin).squeeze(-1)


class BinPackingCriticRLBSA(nn.Module):
    def __init__(self, problem, embed_dim: int, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.problem = problem
        self.embed_dim = embed_dim
        
        self.state_dim = 5

        # MLP critic processes per-bin features (as in paper)
        self.q_func = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, 1, device=device),
        )

        self.q_func.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch, problem_dim, features] where features = [x, w, free_capacity, E, ΔE, T]
            
        Returns:
            value: [batch] state value estimate
        """
        batch_size, problem_dim, n_features = state.shape
        
        if n_features >= 6:
            # [w, free_capacity, E, ΔE, T]
            features = state[..., 1:6]
        elif n_features >= 5:
            # [w, free_capacity, E/T, ΔE/0, T/0]
            features = state[..., 1:5]
            if features.shape[-1] < 5:
                padding = torch.zeros(batch_size, problem_dim, 5 - features.shape[-1],
                                     device=state.device)
                features = torch.cat([features, padding], dim=-1)
        else:
            # Minimal features [w, free_capacity, T], pad to 5
            features = state[..., 1:4]
            padding = torch.zeros(batch_size, problem_dim, 5 - features.shape[-1],
                                 device=state.device)
            features = torch.cat([features, padding], dim=-1)
        
        # Process features through MLP and aggregate
        q_values = self.q_func(features).view(batch_size, problem_dim)
        return q_values.mean(dim=-1)



# Backward compatibility - default to NSA
BinPackingActor = BinPackingActorNSA
BinPackingCritic = BinPackingCriticNSA


# ============================================================
# TSP - NSA (Original Neural SA)
# ============================================================

class TSPActorNSA(SAModel):
    def __init__(self, embed_dim: int, device: str) -> None:
        super().__init__(device)
        self.c1_state_dim = 7
        self.c2_state_dim = 13

        # Mean and std computation
        self.city1_net = nn.Sequential(
            nn.Linear(self.c1_state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=False, device=device),
        )
        # Mean and std computation
        self.city2_net = nn.Sequential(
            nn.Linear(self.c2_state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=False, device=device),
        )

        self.city1_net.apply(self.init_weights)
        self.city2_net.apply(self.init_weights)

    def sample_from_logits(
        self, logits: torch.Tensor, greedy: bool = False, one_hot: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_problems, problem_dim = logits.shape
        probs = torch.softmax(logits, dim=-1)

        if greedy:
            smpl = torch.argmax(probs, -1, keepdim=False)
        else:
            smpl = torch.multinomial(probs, 1, generator=self.generator)[..., 0]

        taken_probs = probs.gather(1, smpl.view(-1, 1))

        if one_hot:
            smpl = F.one_hot(smpl, num_classes=problem_dim)[..., None]

        return smpl, torch.log(taken_probs)

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_problems, problem_dim, _ = state.shape
        x, coords, temp = state[..., :1], state[..., 1:-1], state[..., [-1]]

        c1 = action[:, 0]
        # c2 = action[:, 1]

        # First city encoding
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)
        coords_next = torch.roll(coords, -1, 1)
        c1_state = torch.cat([coords, coords_prev, coords_next, temp], -1)

        # City 1 net
        logits = self.city1_net(c1_state)[..., 0]
        probs = torch.softmax(logits, dim=-1)
        log_probs_c1 = torch.log(probs)

        c1_prev = (c1 - 1) % problem_dim
        c1_next = (c1 + 1) % problem_dim

        # Second city encoding: [base.prev, base, base.next,
        #                        target.prev, target, target.next]
        arange = torch.arange(n_problems)
        c1_coords = coords[arange, c1]
        c1_prev_coords = coords[arange, c1_prev]
        c1_next_coords = coords[arange, c1_next]
        base = torch.cat([c1_coords, c1_prev_coords, c1_next_coords], -1)[:, None, :]
        base = repeat_to(base, c1_state)
        c2_state = torch.cat([base, c1_state], -1)

        # City 2 net
        logits = self.city2_net(c2_state)[..., 0]
        logits[arange, c1] = -float("inf")
        logits[arange, c1_prev] = -float("inf")
        logits[arange, c1_next] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        log_probs_c2 = torch.log(probs)

        return log_probs_c1, log_probs_c2

    def baseline_sample(self, state: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, None]:
        n_problems, problem_dim, _ = state.shape

        # Sample c1 at random
        logits = torch.ones((n_problems, problem_dim), device=self.device)
        c1, _ = self.sample_from_logits(logits, one_hot=False)

        # Compute mask and sample c2
        c1_prev = (c1 - 1) % problem_dim
        c1_next = (c1 + 1) % problem_dim
        arange = torch.arange(n_problems)
        logits[arange, c1] = -float("inf")
        logits[arange, c1_prev] = -float("inf")
        logits[arange, c1_next] = -float("inf")
        c2, _ = self.sample_from_logits(logits, one_hot=False)

        # Construct action tensor and return
        action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        return action, None

    def sample(
        self, state: torch.Tensor, greedy: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_problems, problem_dim, n_features = state.shape
        # Extract features: x(1), coords(2), temp(1), [delta_e(1)]
        x = state[..., :1]
        coords = state[..., 1:3]  # Always indices 1 and 2
        temp = state[..., [3]]  # Always index 3
        # delta_e at index 4 if present (ignored)

        # First city encoding
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)
        coords_next = torch.roll(coords, -1, 1)
        c1_state = torch.cat([coords, coords_prev, coords_next, temp], -1)

        # City 1 net
        logits = self.city1_net(c1_state)[..., 0]
        c1, log_probs_c1 = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

        c1_prev = (c1 - 1) % problem_dim
        c1_next = (c1 + 1) % problem_dim

        # Second city encoding: [base.prev, base, base.next,
        #                        target.prev, target, target.next]
        arange = torch.arange(n_problems)
        c1_coords = coords[arange, c1]
        c1_prev_coords = coords[arange, c1_prev]
        c1_next_coords = coords[arange, c1_next]
        base = torch.cat([c1_coords, c1_prev_coords, c1_next_coords], -1)[:, None, :]

        base = repeat_to(base, c1_state)
        c2_state = torch.cat([base, c1_state], -1)

        # City 2 net
        logits = self.city2_net(c2_state)[..., 0]
        logits[arange, c1] = -float("inf")
        logits[arange, c1_prev] = -float("inf")
        logits[arange, c1_next] = -float("inf")
        c2, log_probs_c2 = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

        # Construct action and log-probabilities
        action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        log_probs = log_probs_c1 + log_probs_c2
        return action, log_probs[..., 0]

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, **kwargs) -> torch.Tensor:
        n_problems, problem_dim, n_features = state.shape
        # Extract features: x(1), coords(2), temp(1), [delta_e(1)]
        x = state[..., :1]
        coords = state[..., 1:3]  # Always indices 1 and 2
        temp = state[..., [3]]  # Always index 3
        # delta_e at index 4 if present (ignored)

        c1 = action[:, 0]
        c2 = action[:, 1]

        # First city encoding
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)
        coords_next = torch.roll(coords, -1, 1)
        c1_state = torch.cat([coords, coords_prev, coords_next, temp], -1)

        # City 1 net
        logits = self.city1_net(c1_state)[..., 0]
        probs = torch.softmax(logits, dim=-1)
        log_probs_c1 = torch.log(probs.gather(1, c1.view(-1, 1)))

        c1_prev = (c1 - 1) % problem_dim
        c1_next = (c1 + 1) % problem_dim

        # Second city encoding: [base.prev, base, base.next,
        #                        target.prev, target, target.next]
        arange = torch.arange(n_problems)
        c1_coords = coords[arange, c1]
        c1_prev_coords = coords[arange, c1_prev]
        c1_next_coords = coords[arange, c1_next]
        base = torch.cat([c1_coords, c1_prev_coords, c1_next_coords], -1)[:, None, :]
        base = repeat_to(base, c1_state)
        c2_state = torch.cat([base, c1_state], -1)

        # City 2 net
        logits = self.city2_net(c2_state)[..., 0]
        logits[arange, c1] = -float("inf")
        logits[arange, c1_prev] = -float("inf")
        logits[arange, c1_next] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        log_probs_c2 = torch.log(probs.gather(1, c2.view(-1, 1)))

        # Construct log-probabilities and return
        log_probs = log_probs_c1 + log_probs_c2
        return log_probs[..., 0]


class TSPCriticNSA(nn.Module):
    def __init__(self, embed_dim: int, device: str = "cpu") -> None:
        super().__init__()
        self.q_func = nn.Sequential(
            nn.Linear(7, embed_dim, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, device=device),
        )

        self.q_func.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        n_problems, problem_dim, n_features = state.shape
        # Extract features: x(1), coords(2), temp(1), [delta_e(1)]
        x = state[..., :1]
        coords = state[..., 1:3]  # Always indices 1 and 2
        temp = state[..., [3]]  # Always index 3
        # delta_e at index 4 if present (ignored)
        
        # state encoding
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)
        coords_next = torch.roll(coords, -1, 1)
        state = torch.cat([coords, coords_prev, coords_next, temp], -1)
        q_values = self.q_func(state).view(n_problems, problem_dim)
        return q_values.mean(dim=-1)


# ============================================================
# TSP - RLBSA (Not yet implemented)
# ============================================================

class TSPActorRLBSA(SAModel):
    """RLBSA actor for TSP - NOT YET IMPLEMENTED"""
    def __init__(self, embed_dim: int, device: str = "cpu") -> None:
        super().__init__(device)
        raise NotImplementedError("RLBSA for TSP is not yet implemented. Use TSPActorNSA instead.")


class TSPCriticRLBSA(nn.Module):
    """RLBSA critic for TSP - NOT YET IMPLEMENTED"""
    def __init__(self, embed_dim: int, device: str = "cpu") -> None:
        super().__init__()
        raise NotImplementedError("RLBSA for TSP is not yet implemented. Use TSPCriticNSA instead.")


# Backward compatibility - default to NSA
TSPActor = TSPActorNSA
TSPCritic = TSPCriticNSA


# ============================================================
# ROSENBROCK - NSA (Original)
# ============================================================

class RosenbrockActorNSA(SAModel):
    """NSA actor for Rosenbrock - Gaussian policy"""
    def __init__(self, problem_dim: int, embed_dim: int, device: str = "cpu") -> None:
        super().__init__(device)
        self.problem_dim = problem_dim
        self.embed_dim = embed_dim
        # State: x (problem_dim) + a (1) + b (1) + temp (1) = problem_dim + 3
        state_dim = problem_dim + 3
        self.mu = torch.zeros(problem_dim, device=device)
        self.log_var = nn.Sequential(
            nn.Linear(state_dim, embed_dim, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, problem_dim, device=device),
            nn.Hardtanh(min_val=-6, max_val=2.0),
        )
        self.log_var.apply(self.init_weights)

    def sample(self, state: torch.Tensor, greedy: bool = False, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        state: [batch, problem_dim + 3] where last 3 are (a, b, temp)
        returns: action [batch, problem_dim], log_probs [batch]
        """
        batch_size = state.shape[0]
        mu = self.mu.repeat(batch_size, 1)
        log_var = self.log_var(state)
        
        if greedy:
            action = mu
        else:
            std = torch.exp(0.5 * log_var)
            action = mu + std * torch.randn(
                (batch_size, self.problem_dim), device=state.device, generator=self.generator
            )
        
        # Compute log probability with numerical stability
        # log N(x; μ, σ²) = -0.5 * (log(2π) + log(σ²) + (x-μ)²/σ²)
        var = torch.exp(log_var)
        log_prob = -0.5 * (
            np.log(2 * np.pi) + log_var + torch.pow(action - mu, 2) / (var + 1e-8)
        )
        return action, log_prob.sum(dim=1)

    def baseline_sample(self, state: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random Gaussian action"""
        batch_size = state.shape[0]
        action = torch.randn((batch_size, self.problem_dim), device=state.device, generator=self.generator)
        # Uniform log prob (not exact, but for baseline)
        log_probs = torch.zeros(batch_size, device=state.device)
        return action, log_probs

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluate log prob of action given state"""
        batch_size = state.shape[0]
        mu = self.mu.repeat(batch_size, 1)
        log_var = self.log_var(state)
        var = torch.exp(log_var)
        log_prob = -0.5 * (
            np.log(2 * np.pi) + log_var + torch.pow(action - mu, 2) / (var + 1e-8)
        )
        return log_prob.sum(dim=1)


class RosenbrockCriticNSA(nn.Module):
    """NSA critic for Rosenbrock"""
    def __init__(self, problem_dim: int, embed_dim: int, device: str = "cpu") -> None:
        super().__init__()
        state_dim = problem_dim + 3  # x + a + b + temp
        self.value_func = nn.Sequential(
            nn.Linear(state_dim, embed_dim, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, device=device),
        )
        self.value_func.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: [batch, problem_dim + 3]
        returns: value [batch]
        """
        return self.value_func(state).squeeze(-1)

# ============================================================
# ROSENBROCK - NSA (Original)
# ============================================================

# class RosenbrockActorNSA(SAModel):
#     """NSA actor for Rosenbrock - Gaussian policy"""
#     def __init__(self, problem_dim: int, embed_dim: int, device: str = "cpu") -> None:
#         super().__init__(device)
#         self.problem_dim = problem_dim
#         # State: x (problem_dim) + a (1) + b (1) + temp (1) = problem_dim + 3
#         state_dim = problem_dim + 3
#         self.mu = torch.zeros(problem_dim, device=device)
#         self.log_var = nn.Sequential(
#             nn.Linear(state_dim, embed_dim, device=device),
#             nn.ReLU(),
#             nn.Linear(embed_dim, problem_dim, device=device),
#             nn.Hardtanh(min_val=-6, max_val=2.0),
#         )
#         self.log_var.apply(self.init_weights)

#     def sample(self, state: torch.Tensor, greedy: bool = False, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         state: [batch, problem_dim + 3] where last 3 are (a, b, temp)
#         returns: action [batch, problem_dim], log_probs [batch]
#         """
#         batch_size = state.shape[0]
#         mu = self.mu.repeat(batch_size, 1)
#         log_var = self.log_var(state)
        
#         if greedy:
#             action = mu
#         else:
#             action = mu + torch.exp(0.5 * log_var) * torch.randn(
#                 (batch_size, self.problem_dim), device=state.device, generator=self.generator
#             )
        
#         # Compute log probability
#         log_prob = -0.5 * (
#             log_var + torch.log(torch.tensor(2 * np.pi, device=state.device)) + 
#             torch.pow(action - mu, 2) / log_var.exp()
#         )
#         return action, log_prob.sum(dim=1)

#     def baseline_sample(self, state: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Random Gaussian action"""
#         batch_size = state.shape[0]
#         action = torch.randn((batch_size, self.problem_dim), device=state.device, generator=self.generator)
#         # Uniform log prob (not exact, but for baseline)
#         log_probs = torch.zeros(batch_size, device=state.device)
#         return action, log_probs

#     def evaluate(self, state: torch.Tensor, action: torch.Tensor, **kwargs) -> torch.Tensor:
#         """Evaluate log prob of action given state"""
#         batch_size = state.shape[0]
#         mu = self.mu.repeat(batch_size, 1)
#         log_var = self.log_var(state)
#         log_prob = -0.5 * (
#             log_var + torch.log(torch.tensor(2 * np.pi, device=state.device)) + 
#             torch.pow(action - mu, 2) / log_var.exp()
#         )
#         return log_prob.sum(dim=1)


# class RosenbrockCriticNSA(nn.Module):
#     """NSA critic for Rosenbrock"""
#     def __init__(self, problem_dim: int, embed_dim: int, device: str = "cpu") -> None:
#         super().__init__()
#         state_dim = problem_dim + 3  # x + a + b + temp
#         self.value_func = nn.Sequential(
#             nn.Linear(state_dim, embed_dim, device=device),
#             nn.ReLU(),
#             nn.Linear(embed_dim, 1, device=device),
#         )
#         self.value_func.apply(self.init_weights)

#     @staticmethod
#     def init_weights(m: nn.Module) -> None:
#         if type(m) == nn.Linear:
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.01)

#     def forward(self, state: torch.Tensor) -> torch.Tensor:
#         """
#         state: [batch, problem_dim + 3]
#         returns: value [batch]
#         """
#         return self.value_func(state).squeeze(-1)


# ============================================================
# ROSENBROCK - RLBSA (RL-Based SA with extended state features)
# ============================================================

class RosenbrockActorRLBSA(SAModel):
    """RLBSA actor for Rosenbrock with extended state features (E, ΔE)
    
    According to paper section 4.2, the key innovation for continuous problems is
    adding current energy E and energy change ΔE to the state:
    S_t = (x_t, E_t, ΔE_t, T_t)
    
    For 2D Rosenbrock with problem parameters (a, b):
    Full state: [x(2), a(1), b(1), E(1), ΔE(1), T(1)] = 7 features
    
    Architecture: 2-layer MLP (7 → 16 → 16 → 2) as per paper Section 4.1.1
    Output: log variance for Gaussian action distribution (mean fixed at 0)
    
    Note: We use MLP instead of LSTM because PPO training shuffles data,
    which breaks LSTM's temporal dependencies. The state features (E, ΔE)
    already encode the relevant history.
    """
    def __init__(self, problem_dim: int, embed_dim: int, device: str = "cpu") -> None:
        super().__init__(device)
        self.problem_dim = problem_dim
        self.embed_dim = embed_dim
        
        # Full state: x(dim) + a(1) + b(1) + E(1) + ΔE(1) + T(1) = dim + 5
        # For 2D: 7 features
        state_dim = problem_dim + 5
        
        # Mean fixed at zero (perturbation-based actions)
        self.mu = torch.zeros(problem_dim, device=device)
        
        # 2-layer MLP: state_dim → embed_dim → embed_dim → problem_dim
        # Paper says 7 → 16 → 16 → 2 for 2D continuous problems
        self.log_var = nn.Sequential(
            nn.Linear(state_dim, embed_dim, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, problem_dim, device=device),
            nn.Hardtanh(min_val=-6, max_val=2.0),  # Constrain log variance
        )
        self.log_var.apply(self.init_weights)

    def sample(self, state: torch.Tensor, greedy: bool = False, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        state: [batch, problem_dim + 5] where format is [x, a, b, E, ΔE, T]
        returns: action [batch, problem_dim], log_probs [batch]
        """
        batch_size = state.shape[0]
        mu = self.mu.repeat(batch_size, 1)
        log_var = self.log_var(state)
        
        if greedy:
            action = mu
        else:
            std = torch.exp(0.5 * log_var)
            action = mu + std * torch.randn(
                (batch_size, self.problem_dim), device=state.device, generator=self.generator
            )
        
        # Compute log probability with numerical stability
        var = torch.exp(log_var)
        log_prob = -0.5 * (
            np.log(2 * np.pi) + log_var + torch.pow(action - mu, 2) / (var + 1e-8)
        )
        return action, log_prob.sum(dim=1)

    def baseline_sample(self, state: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random Gaussian action"""
        batch_size = state.shape[0]
        action = torch.randn((batch_size, self.problem_dim), device=state.device, generator=self.generator)
        log_probs = torch.zeros(batch_size, device=state.device)
        return action, log_probs

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluate log prob of action given state"""
        batch_size = state.shape[0]
        mu = self.mu.repeat(batch_size, 1)
        log_var = self.log_var(state)
        var = torch.exp(log_var)
        log_prob = -0.5 * (
            np.log(2 * np.pi) + log_var + torch.pow(action - mu, 2) / (var + 1e-8)
        )
        return log_prob.sum(dim=1)


class RosenbrockCriticRLBSA(nn.Module):
    """RLBSA critic for Rosenbrock with extended state features (E, ΔE)
    
    Full state: [x(dim), a, b, E, ΔE, T] = dim + 5 features
    Architecture: 2-layer MLP matching actor
    """
    def __init__(self, problem_dim: int, embed_dim: int, device: str = "cpu") -> None:
        super().__init__()
        state_dim = problem_dim + 5  # x + a + b + E + ΔE + T
        
        self.value_func = nn.Sequential(
            nn.Linear(state_dim, embed_dim, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, device=device),
        )
        self.value_func.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: [batch, problem_dim + 5]
        returns: value [batch]
        """
        return self.value_func(state).squeeze(-1)


# Backward compatibility - default to NSA
RosenbrockActor = RosenbrockActorNSA
RosenbrockCritic = RosenbrockCriticNSA


# ============================================================
# LEGACY ROSENBROCK (Original combined actor-critic)
# ============================================================

class RosenNet(nn.Module):
    """Legacy combined actor-critic for Rosenbrock - DEPRECATED
    
    Use RosenbrockActorNSA and RosenbrockCriticNSA instead.
    """
    def __init__(self, problem_dim: int, embed_dim: int) -> None:
        super().__init__()
        state_dim = problem_dim + 1
        self.mu = torch.zeros(problem_dim)
        self.log_var = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, problem_dim),
            nn.Hardtanh(min_val=-6, max_val=2.0),
        )
        self.value_func = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def actor_evaluate(
        self, state: torch.Tensor, action: torch.Tensor, temp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, problem_dim = state.shape
        s = torch.cat((state, temp.view(-1, 1)), dim=-1)
        mu = self.mu.repeat(batch_size, 1)
        log_var = self.log_var(s)
        # Compute logprob
        log_prob = -0.5 * (
            log_var + torch.log(torch.tensor(2 * np.pi)) + torch.pow(action - mu, 2) / log_var.exp()
        )
        entropy = 0.5 * torch.log((2 * np.e * np.pi) ** problem_dim * log_var.sum(dim=-1).exp())
        return log_prob.sum(dim=1), entropy

    def critic(self, state: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        s = torch.cat((state, temp.view(-1, 1)), dim=-1)
        return self.value_func(s)

    def actor_sample(
        self, state: torch.Tensor, temp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, problem_dim = state.shape
        s = torch.cat((state, temp.view(-1, 1)), dim=-1)
        mu = self.mu.repeat(batch_size, 1)
        log_var = self.log_var(s)
        action = mu + torch.exp(0.5 * log_var) * torch.randn(
            (batch_size, problem_dim), generator=self.generator
        )
        log_prob, _ = self.actor_evaluate(state, action, temp)
        return action.view(batch_size, problem_dim), log_prob, mu, torch.exp(0.5 * log_var)
