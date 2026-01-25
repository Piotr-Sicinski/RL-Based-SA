# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch

from neuralsa.utils import repeat_to


class Problem(ABC):
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.generator = torch.Generator(device=device)

    def gain(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.cost(s) - self.cost(self.update(s, a))

    def manual_seed(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    @abstractmethod
    def cost(self, s: torch.Tensor) -> torch.float:
        pass

    @abstractmethod
    def update(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def set_params(self, **kwargs) -> None:
        pass

    @abstractmethod
    def generate_params(self) -> Dict[str, torch.Tensor]:
        pass

    @property
    def state_encoding(self) -> torch.Tensor:
        pass

    @abstractmethod
    def generate_init_state(self) -> torch.Tensor:
        pass

    def to_state(self, x: torch.Tensor, temp: torch.Tensor, delta_e: torch.Tensor = None):
        """Construct state with optional ΔE (energy change from previous step)"""
        # Keep per-item structure: [batch, dim, features]
        state = torch.cat([x, self.state_encoding, repeat_to(temp, x)], -1)
        if delta_e is not None:
            state = torch.cat([state, repeat_to(delta_e, x)], -1)
        return state

    def from_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract x, spec, temp from state (ignoring ΔE if present)"""
        return state[..., : self.x_dim], state[..., self.x_dim : -1], state[..., -1:]





class Rosenbrock(Problem):
    x_dim = 2

    def __init__(self, dim=2, n_problems=256, device="cpu", params={}):
        """
        Initialize random Rosenbrock functions.

        Args:
            dim: int
            a: [n_problems, 1]
            b: [n_problems, 1]
        """
        super().__init__(device)
        self.dim = dim
        self.n_problems = n_problems
        self.set_params(**params)

    def set_params(self, a=None, b=None):
        self.a = a
        self.b = b

    def generate_params(self, mode="train"):
        if mode == "test":
            self.manual_seed(0)
        a = torch.rand(self.n_problems, 1, device=self.device, generator=self.generator)
        b = 100 * torch.rand(self.n_problems, 1, device=self.device, generator=self.generator)
        return {"a": a, "b": b}

    @property
    def state_encoding(self) -> torch.Tensor:
        """Return parameters to add to state vectors"""
        return torch.cat([self.a, self.b / 100], -1)

    def generate_init_x(self) -> torch.Tensor:
        x = torch.randn(self.n_problems, self.dim, device=self.device, generator=self.generator)
        return x

    def generate_init_state(self) -> torch.Tensor:
        x = torch.randn(self.n_problems, self.dim, device=self.device, generator=self.generator)
        return torch.cat([x, self.state_encoding], -1)

    def to_state(self, x: torch.Tensor, temp: torch.Tensor, delta_e: torch.Tensor = None,
                 current_energy: torch.Tensor = None) -> torch.Tensor:
        """Construct state for Rosenbrock (flat vector structure)
        
        According to paper (line 806), the state format for continuous functions should be:
        - NSA: [x, a, b, T] - dim + 3 features (2D: 5 features)
        - RLBSA: [x, a, b, E, ΔE, T] - dim + 5 features (2D: 7 features)
        
        Note: Current energy E and ΔE are both included for RLBSA.
        For continuous: E comes before ΔE, then T at end.
        
        Args:
            x: [batch, dim] - current solution
            temp: [batch, 1] or scalar - temperature
            delta_e: [batch, 1] or None - energy change ΔE (for RLBSA)
            current_energy: [batch, 1] or None - current energy E (for RLBSA)
            
        Returns:
            state: [batch, dim + features]
        """
        # x: [batch, dim], state_encoding: [batch, 2], temp: [batch, 1]
        if temp.dim() == 0 or (temp.dim() == 1 and temp.shape[0] == 1):
            # Scalar or single value, expand to [batch, 1]
            temp = temp.view(1, 1).expand(x.shape[0], 1)
        elif temp.dim() == 1:
            # [batch] -> [batch, 1]
            temp = temp.unsqueeze(-1)
        
        # Base state: [x, a, b] - always present
        state = torch.cat([x, self.state_encoding], -1)
        
        # For RLBSA: add E and ΔE before temperature
        # Order: [x(dim), a(1), b(1), E(1), ΔE(1), T(1)]
        if current_energy is not None and delta_e is not None:
            if current_energy.dim() == 1:
                current_energy = current_energy.unsqueeze(-1)
            if delta_e.dim() == 1:
                delta_e = delta_e.unsqueeze(-1)
            state = torch.cat([state, current_energy, delta_e], -1)
        
        # Add temperature at the end
        state = torch.cat([state, temp], -1)
        
        return state

    def from_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract x, spec, temp from state (ignoring E and ΔE if present)
        
        State can be in two formats:
        - NSA: [x(dim), a, b, T] - dim + 3 features (2D: 5)
        - RLBSA: [x(dim), a, b, E, ΔE, T] - dim + 5 features (2D: 7)
        
        Args:
            state: [batch, features] where features is dim+3 or dim+4
            
        Returns:
            x: [batch, dim]
            spec: [batch, 2] (a, b/100)
            temp: [batch, 1]
        """
        x = state[..., :self.dim]
        spec = state[..., self.dim:self.dim + 2]
        # Temperature is always at the end
        temp = state[..., -1:]
        return x, spec, temp

    def cost(self, s: torch.Tensor) -> torch.Tensor:
        """Evaluate Rosenbrock

        Args:
            s: [n_problems, self.dim]
        Returns:
            [n_problems] costs
        """
        return torch.sum(
            self.b * (s[:, 1:] - s[:, :-1] ** 2.0) ** 2.0 + (self.a - s[:, :-1]) ** 2.0, dim=-1
        )

    def update(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return s + a


class Knapsack(Problem):
    x_dim = 1  # Per-item dimension

    def __init__(self, dim: int = 50, n_problems: int = 256, device: str = "cpu", params: Dict = {}):
        super().__init__(device)
        self.dim = dim
        self.n_problems = n_problems
        params["capacity"] = params["capacity"] * torch.ones((self.n_problems, 1), device=device)
        self.set_params(**params)

    def set_params(
        self,
        weights: torch.Tensor = None,
        values: torch.Tensor = None,
        capacity: torch.Tensor = None,
    ):
        self.weights = weights
        self.values = values
        self.capacity = capacity

    def generate_params(self, mode: str = "train") -> Dict[str, torch.Tensor]:
        if mode == "test":
            self.manual_seed(0)
        v = torch.rand(self.n_problems, self.dim, device=self.device, generator=self.generator)
        w = torch.rand(self.n_problems, self.dim, device=self.device, generator=self.generator)
        if self.capacity is not None:
            c = self.capacity
        else:
            c = (
                self.dim
                * (
                    1
                    + torch.rand((self.n_problems, 1), device=self.device, generator=self.generator)
                )
                / 8
            )
        return {"values": v, "weights": w, "capacity": c}

    @property
    def state_encoding(self) -> torch.Tensor:
        """
        Returns:
            [batch, dim, 3] tensor with per-item (weight, value, capacity)
            
        Note: According to the paper (Section 4.1.1), the state should include
        raw capacity W, not normalized capacity. Each item gets the same W value.
        """
        ones = torch.ones((self.dim,), device=self.device)
        capacity = self.capacity * ones  # [batch, dim] - raw capacity W
        # Stack along last dimension to get [batch, dim, 3]
        return torch.stack([self.weights, self.values, capacity], -1)

    def generate_init_x(self) -> torch.Tensor:
        # [batch, dim, 1] - per-item structure
        return torch.zeros((self.n_problems, self.dim, 1), device=self.device)

    def generate_init_state(self) -> torch.Tensor:
        x = self.generate_init_x()
        return torch.cat([x, self.state_encoding], -1)

    def to_state(self, x: torch.Tensor, temp: torch.Tensor, delta_e: torch.Tensor = None, 
                 current_energy: torch.Tensor = None) -> torch.Tensor:
        """Construct state with optional ΔE and current energy E
        
        Args:
            x: Current solution [batch, dim, 1]
            temp: Temperature [batch, 1] or scalar
            delta_e: Energy change from previous step [batch, 1] or None
            current_energy: Current energy E [batch, 1] or None (for RLBSA LSTM models)
        
        Returns:
            state: [batch, dim, features] where features depends on what's included:
                - Basic (NSA): [x, w, v, W, T] - 5 features
                - With ΔE: [x, w, v, W, T, ΔE] - 6 features  
                - With E and ΔE (RLBSA): [x, w, v, W, E, ΔE, T] - 7 features
        """
        # Keep per-item structure: [batch, dim, features]
        # state_encoding gives us [batch, dim, 3] with [w, v, W]
        state = torch.cat([x, self.state_encoding, repeat_to(temp, x)], -1)
        
        # For RLBSA with LSTM: add current energy E before ΔE
        # Order: [x, w, v, W, E, ΔE, T] = 7 features
        if current_energy is not None:
            state = torch.cat([state, repeat_to(current_energy, x)], -1)
        
        if delta_e is not None:
            state = torch.cat([state, repeat_to(delta_e, x)], -1)
        
        return state

    # def cost(self, s: torch.Tensor) -> torch.Tensor:
    #     # s: [batch, dim, 1], binary solution
    #     v = torch.sum(self.values * s[..., 0], -1)
    #     w = torch.sum(self.weights * s[..., 0], -1)
    #     # penalize overweight solutions
    #     penalty = (w > self.capacity[..., 0]).float() * 1e6
    #     return -(v - penalty)  # negative because SA minimizes energy

    def cost(self, s: torch.Tensor) -> torch.Tensor:
        v = torch.sum(self.values * s[..., 0], -1)
        w = torch.sum(self.weights * s[..., 0], -1)
        return -v * (w < self.capacity[..., 0])

    def update(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Apply XOR with action to flip bits.
        s: current solution [batch, dim, 1]
        a: action [batch, dim] or [batch, dim, 1] one-hot indicating which item to flip
        """
        # Handle both [batch, dim] and [batch, dim, 1] formats
        if a.dim() == 2:
            # Expand action to match s shape: [batch, dim] -> [batch, dim, 1]
            a_expanded = a.unsqueeze(-1)
        else:
            # Already [batch, dim, 1]
            a_expanded = a
        # XOR: flip the bit where action is 1
        return ((s > 0.5) ^ (a_expanded > 0.5)).float()



class BinPacking(Problem):
    x_dim = 1

    def __init__(
        self,
        dim: int = 50,
        n_problems: int = 256,
        device: str = "cpu",
        params: Dict[str, torch.Tensor] = {},
    ):
        """Initialize BinPacking.

        Args:
            dim: num items
            n_problems: batch size
            device: device to use
            params: {'weights': torch.Tensor}
        """
        super().__init__(device)
        self.dim = dim
        self.n_problems = n_problems
        self.capacity = 1.0
        self.set_params(**params)

    def set_params(self, weights: torch.Tensor = None):
        """Set params.

        Args:
            weights: [batch_size, dim]
        """
        self.weights = weights

    def generate_params(self, mode: str = "train") -> Dict[str, torch.Tensor]:
        """Generate random weights in [0,1). Capacity taken to be 1.

        Returns:
            dict with 'weights' [n_problems, dim]
        """
        if mode == "test":
            self.manual_seed(0)
        w = torch.rand(self.n_problems, self.dim, device=self.device, generator=self.generator)
        return {"weights": w}

    @property
    def state_encoding(self) -> torch.Tensor:
        """
        Returns problem parameters ψ.
        
        Returns:
            [batch_size, dim, 1] tensor with per-item weights
        """
        return self.weights.unsqueeze(-1)

    def generate_init_x(self) -> torch.Tensor:
        """Generate initial bin assignments - each item in its own bin.
        
        Returns:
            [batch_size, dim, 1] tensor where each item i is assigned to bin i
        """
        x = torch.arange(self.dim, device=self.device).unsqueeze(0).expand(self.n_problems, -1)
        return x.unsqueeze(-1).float()

    def generate_init_state(self) -> torch.Tensor:
        """State encoding has dims [batch_size, dim, features]
        
        Features per item: [bin_assignment, weight]
        
        Returns:
            [batch_size, dim, 2] tensor
        """
        x = self.generate_init_x()
        return torch.cat([x, self.state_encoding], -1)

    def to_state(self, x: torch.Tensor, temp: torch.Tensor, energy: torch.Tensor = None, 
                 delta_energy: torch.Tensor = None) -> torch.Tensor:
        """Convert solution x to full state representation.
        
        According to RL-Based-SA paper (Section 3.5):
        State for discrete problems: S_i = (x, ψ, E, ΔE, T)
        
        Args:
            x: [batch_size, dim, 1] bin assignments
            temp: temperature values (scalar, [batch_size], or [batch_size, 1])
            energy: current energy E (optional, scalar, [batch_size], or [batch_size, 1])
            delta_energy: energy change ΔE (optional, scalar, [batch_size], or [batch_size, 1])
            
        Returns:
            [batch_size, dim, 6] tensor with features:
            [bin_assignment, weight, free_capacity, E, ΔE, T]
        """
        batch_size, problem_dim, _ = x.shape
        
        def normalize_to_batch(tensor, default_val=0.0):
            if tensor is None:
                return torch.full((batch_size, 1), default_val, device=self.device)
            if tensor.dim() == 0:
                # Scalar
                return tensor.view(1, 1).expand(batch_size, 1)
            elif tensor.dim() == 1:
                # [batch_size] or [1]
                if tensor.size(0) == 1:
                    return tensor.view(1, 1).expand(batch_size, 1)
                else:
                    return tensor.unsqueeze(-1)
            else:
                # [batch_size, 1] or [1, 1]
                if tensor.size(0) == 1:
                    return tensor.expand(batch_size, -1)
                else:
                    return tensor
        
        # Get weight of each item (ψ - problem parameters)
        w = self.weights.unsqueeze(-1)  # [batch_size, dim, 1]
        
        # Get the free capacity of each bin
        wb = self.get_bin_volume(x[..., 0])  # [batch_size, dim]
        free_capacity = (self.capacity - wb).unsqueeze(-1)  # [batch_size, dim, 1]
        
        # Energy E - broadcast to all items
        if energy is None:
            energy = self.cost(x)  # [batch_size]
        energy_norm = normalize_to_batch(energy)  # [batch_size, 1]
        E = energy_norm.unsqueeze(1).expand(-1, problem_dim, -1)  # [batch_size, dim, 1]
        
        # Energy change ΔE - broadcast to all items
        delta_energy_norm = normalize_to_batch(delta_energy, default_val=0.0)  # [batch_size, 1]
        delta_E = delta_energy_norm.unsqueeze(1).expand(-1, problem_dim, -1)  # [batch_size, dim, 1]
        
        # Temperature T - broadcast to all items
        temp_norm = normalize_to_batch(temp, default_val=1.0)  # [batch_size, 1]
        temp_expanded = temp_norm.unsqueeze(1).expand(-1, problem_dim, -1)  # [batch_size, dim, 1]
        
        # Concatenate all features: [x, ψ, E, ΔE, T]
        # where ψ includes both weight and free_capacity
        state = torch.cat((x, w, free_capacity, E, delta_E, temp_expanded), dim=-1)
        
        return state

    def cost(self, s: torch.Tensor) -> torch.Tensor:
        """Compute bin-packing objective (lower is better).
        
        Objective: minimize number of occupied bins
        Penalty: heavily penalize overflow
        
        Args:
            s: [batch_size, dim, 1] bin assignments
            
        Returns:
            [batch_size] costs (energy E)
        """
        x = s[..., 0].long()

        # Get volume in each bin
        volumes = self.get_bin_volume(x)
        occupied = (volumes > 0).float()
        K = torch.sum(occupied, -1)
        
        # Check for overflow (bins with volume > capacity)
        overflowed = (volumes > self.capacity).float()
        has_overflow = (torch.sum(overflowed, -1) > 0.5).float()

        # Penalize overflow heavily
        penalty = has_overflow * self.dim * 10  # Large penalty for overflow
        
        return K + penalty

    def update(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Update bin assignments based on action.
        
        Action specifies: move item_idx to bin_idx
        
        Args:
            s: [batch_size, dim, 1] current bin assignments
            a: [batch_size, 2] action (item_idx, bin_idx)
            
        Returns:
            [batch_size, dim, 1] updated bin assignments
        """
        item_idx = a[:, 0].long()  # [batch_size]
        bin_idx = a[:, 1].long()   # [batch_size]
        
        # Clone current state
        x_new = s[..., 0].clone()
        
        # Update: assign item_idx to bin_idx
        # Use scatter to update the selected items
        x_new.scatter_(1, item_idx.unsqueeze(1), bin_idx.unsqueeze(1).float())
        
        return x_new.unsqueeze(-1)

    def get_bin_volume(self, x: torch.Tensor) -> torch.Tensor:
        """Compute volume in each bin.
        
        Args:
            x: [batch_size, dim] bin assignments (each element indexes into a bin)
            
        Returns:
            [batch_size, dim] volumes per bin
        """
        volumes = torch.zeros_like(x, dtype=torch.float32, device=self.device)
        volumes.scatter_add_(1, x.long(), self.weights)
        return volumes

    def get_item_weight(self, item: torch.Tensor) -> torch.Tensor:
        """Get weight of specific items.
        
        Args:
            item: [batch_size] item indices
            
        Returns:
            [batch_size] weights
        """
        batch_indices = torch.arange(len(item), device=self.device)
        return self.weights[batch_indices, item.long()]

    def get_item_bin_volume(self, x: torch.Tensor) -> torch.Tensor:
        """Get volume of the bin that each item is currently in.
        
        Args:
            x: [batch_size, dim] bin assignments
            
        Returns:
            [batch_size, dim] volume of each item's bin
        """
        volumes = self.get_bin_volume(x)
        return torch.gather(volumes, 1, x.long())



class TSP(Problem):
    x_dim = 1

    def __init__(self, dim: int = 50, n_problems: int = 256, device: str = "cpu", params: str = {}):
        """Initialize BinPacking.

        Args:
            dim: num items
            n_problems: batch size
            params: {'weight': torch.Tensor}
        """
        super().__init__(device)
        self.dim = dim
        self.n_problems = n_problems
        self.set_params(**params)

    def set_params(self, coords: torch.Tensor = None) -> None:
        """Set params.

        Args:
            coords: [batch size, dim, 2]
        """
        self.coords = coords

    def generate_params(self, mode: str = "train") -> Dict[str, torch.Tensor]:
        """Generate random coordinates in the unit square.

        Returns:
            coords [batch size, num problems, 2]
        """
        if mode == "test":
            self.manual_seed(0)
        coords = torch.rand(
            self.n_problems, self.dim, 2, device=self.device, generator=self.generator
        )
        return {"coords": coords}

    def cost(self, s: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean tour lengths from city permutations

        Args:
            s: [batch size, dim]
        """
        # Edge lengths
        edge_lengths = self.get_edge_lengths_in_tour(s)
        return torch.sum(edge_lengths, -1)

    def update(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Silly city swap for now

        Args:
            s: perm vector [batch size, coords]
            a: cities to swap ([batch size], [batch size])
        """
        return self.two_opt(s[..., 0], a)[..., None]

    def two_opt(self, x: torch.Tensor, a: torch.Tensor):
        """Swap cities a[0] <-> a[1].

        Args:
            s: perm vector [batch size, coords]
            a: cities to swap ([batch size], [batch size])
        """
        # Two-opt moves invert a section of a tour. If we cut a tour into
        # segments a and b then we can choose to invert either a or b. Due
        # to the linear representation of a tour, we choose always to invert
        # the segment that is stored contiguously.
        l = torch.minimum(a[:, 0], a[:, 1])
        r = torch.maximum(a[:, 0], a[:, 1])
        ones = torch.ones((self.n_problems, 1), dtype=torch.long, device=self.device)
        fidx = torch.arange(self.dim, device=self.device) * ones
        # Reversed indices
        offset = l + r - 1
        ridx = torch.arange(0, -self.dim, -1, device=self.device) + offset[:, None]
        # Set flipped section to all True
        flip = torch.ge(fidx, l[:, None]) * torch.lt(fidx, r[:, None])
        # Set indices to replace flipped section with
        idx = (~flip) * fidx + flip * ridx
        # Perform 2-opt move
        return torch.gather(x, 1, idx)

    @property
    def state_encoding(self) -> torch.Tensor:
        return self.coords

    def get_coords(self, s: torch.Tensor) -> torch.Tensor:
        """Get coords from tour permutation."""
        permutation = s[..., None].expand_as(self.coords).long()
        return self.coords.gather(1, permutation)

    def generate_init_x(self) -> torch.Tensor:
        perm = torch.cat(
            [
                torch.randperm(self.dim, device=self.device, generator=self.generator).view(1, -1)
                for _ in range(self.n_problems)
            ],
            dim=0,
        ).to(self.device)
        return perm[..., None]

    def generate_init_state(self) -> torch.Tensor:
        """State encoding has dims

        [state enc] = [batch size, num items, concat]
        """
        perm = self.generate_init_x()
        return torch.cat([perm, self.state_encoding], -1)

    def get_edge_offsets_in_tour(self, s: torch.Tensor) -> torch.Tensor:
        """Compute vector to right city in tour

        Args:
            s: [batch size, dim]
        Returns:
            [batch size, dim, 2]
        """
        # Gather dataset in order of tour
        d = self.get_coords(s[..., 0])
        d_roll = torch.roll(d, -1, 1)
        # Edge lengths
        return d_roll - d

    def get_edge_lengths_in_tour(self, s: torch.Tensor) -> torch.Tensor:
        """Compute distance to right city in tour

        Args:
            s: [batch size, dim, 1]
        Returns:
            [batch size, dim]
        """
        # Edge offsets
        offset = self.get_edge_offsets_in_tour(s)
        # Edge lengths
        return torch.sqrt(torch.sum(offset**2, -1))

    def get_neighbors_in_tour(self, s: torch.Tensor) -> torch.Tensor:
        """Return distances to neighbors in tour.

        Args:
            s: [batch size, dim, 1] vector
        """
        right_distance = self.get_edge_lengths_in_tour(s)
        left_distance = torch.roll(right_distance, 1, 1)
        return torch.stack([right_distance, left_distance], -1)
