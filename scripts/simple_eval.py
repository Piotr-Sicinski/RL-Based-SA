# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from neuralsa.configs import NeuralSAExperiment, TrainingConfig, SAConfig
from neuralsa.model import (
    BinPackingActorNSA,
    BinPackingActorRLBSA,
    KnapsackActorNSA,
    KnapsackActorRLBSA,
    RosenbrockActorNSA,
    RosenbrockActorRLBSA,
)
from neuralsa.problem import BinPacking, Knapsack, Rosenbrock
from neuralsa.sa import sa

# For reproducibility on GPU
torch.backends.cudnn.deterministic = True


def create_folder(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_model_path(project_root: str, model_name: str) -> str:
    """Get model path from model name."""
    # Try multiple locations
    possible_paths = [
        os.path.join(project_root, "outputs", "models", f"{model_name}.pt"),
        os.path.join(project_root, "models", f"{model_name}.pt"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"Model not found: {model_name} (tried {possible_paths})")


def create_actor_and_problem(problem_name: str, method_type: str, problem_dim: int, device: str, embed_dim: int = 128):
    """Create actor and problem instances."""
    if problem_name == "knapsack":
        # Set capacity based on problem dimension
        if problem_dim == 50:
            capacity = 12.5
        elif problem_dim == 100:
            capacity = 25
        else:
            capacity = problem_dim / 8
        
        problem = Knapsack(problem_dim, n_problems=1, device=device, params={"capacity": capacity})
        
        if method_type == "rlbsa":
            actor = KnapsackActorRLBSA(problem, embed_dim, device=device)
        else:  # nsa
            actor = KnapsackActorNSA(problem, embed_dim, device=device)
            
    elif problem_name == "binpacking":
        problem = BinPacking(problem_dim, n_problems=1, device=device)
        
        if method_type == "rlbsa":
            actor = BinPackingActorRLBSA(problem, embed_dim, device=device)
        else:  # nsa
            actor = BinPackingActorNSA(embed_dim, device=device)
            
    elif problem_name == "rosenbrock":
        problem = Rosenbrock(problem_dim, n_problems=1, device=device)
        
        if method_type == "rlbsa":
            actor = RosenbrockActorRLBSA(problem_dim, embed_dim, device=device)
        else:  # nsa
            actor = RosenbrockActorNSA(problem_dim, embed_dim, device=device)
    else:
        raise ValueError(f"Invalid problem: {problem_name}")
    
    return actor, problem


def run_single_evaluation(
    actor,
    problem,
    problem_dim: int,
    K: int,
    init_temp: float,
    stop_temp: float,
    device: str,
    baseline: bool = False,
    greedy: bool = False,
    seed: int = 1,
) -> Tuple[float, float]:
    """Run a single evaluation and return (cost, time)."""
    torch.manual_seed(seed)
    actor.manual_seed(seed)
    
    # Set SA steps
    outer_steps = K * problem_dim
    
    # Define temperature decay parameter
    alpha = np.log(stop_temp) - np.log(init_temp)
    alpha = np.exp(alpha / outer_steps).item()
    
    # Create SA config
    sa_config = SAConfig(
        outer_steps=outer_steps,
        init_temp=init_temp,
        stop_temp=stop_temp,
        alpha=alpha,
    )
    
    # Create minimal config for SA
    cfg = NeuralSAExperiment(
        problem="dummy",
        problem_dim=problem_dim,
        device=device,
        sa=sa_config,
        training=TrainingConfig(),
    )
    
    # Generate initial solution
    init_x = problem.generate_init_x().to(device)
    
    # Generate problem parameters
    params = problem.generate_params(mode="test")
    params = {k: v.to(device) for k, v in params.items()}
    problem.set_params(**params)
    
    # Run SA
    torch.cuda.empty_cache()
    start_time = time.time()
    out = sa(actor, problem, init_x, cfg, replay=None, baseline=baseline, greedy=greedy)
    elapsed_time = time.time() - start_time
    
    min_cost = torch.mean(out["min_cost"]).item()
    
    return min_cost, elapsed_time


def evaluate_config(args):
    """Worker function for multiprocessing evaluation."""
    (problem_name, problem_dim, K, method_type, model_path, 
     init_temp, stop_temp, device, worker_id, n_runs, embed_dim) = args
    
    try:
        print(f"[Worker {worker_id}] Starting: {problem_name} dim={problem_dim} K={K} method={method_type} device={device}")
        
        # For baseline, use NSA actor but set baseline=True
        actor_type = "nsa" if method_type == "baseline" else method_type
        
        # Create actor and problem
        actor, problem = create_actor_and_problem(problem_name, actor_type, problem_dim, device, embed_dim)
        
        # Load model weights
        actor.load_state_dict(torch.load(model_path, map_location=device))
        actor.eval()
        
        # Determine evaluation mode
        baseline_mode = (method_type == "baseline")
        
        # Run evaluations
        results = []
        times = []
        
        for seed in range(1, n_runs + 1):
            cost, elapsed = run_single_evaluation(
                actor, problem, problem_dim, K, init_temp, stop_temp, device,
                baseline=baseline_mode, greedy=False, seed=seed
            )
            results.append(cost)
            times.append(elapsed)
        
        # Clean up
        del actor, problem
        torch.cuda.empty_cache()
        
        result = {
            "problem": problem_name,
            "problem_dim": problem_dim,
            "K": K,
            "method": method_type,
            "results": results,
            "times": times,
            "min": float(np.min(results)),
            "avg": float(np.mean(results)),
            "std": float(np.std(results)),
            "time_avg": float(np.mean(times)),
        }
        
        print(f"[Worker {worker_id}] Done: {problem_name} dim={problem_dim} K={K} method={method_type} "
              f"avg={result['avg']:.4f} time={result['time_avg']:.2f}s")
        
        return result
        
    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


@hydra.main(config_path="conf", config_name="simple_eval", version_base=None)
def main(cfg):
    # Get project root
    project_root = get_original_cwd()
    
    # Parse config
    experiments = OmegaConf.to_container(cfg.get("experiments", {}), resolve=True)
    num_workers = cfg.get("num_workers", 1)
    devices = cfg.get("devices", ["cpu"])
    n_runs = cfg.get("n_runs", 5)
    init_temp = cfg.get("init_temp", 1.0)
    stop_temp = cfg.get("stop_temp", 0.0005)
    embed_dim = cfg.get("embed_dim", 128)
    
    print(f"Configuration:")
    print(f"  Workers: {num_workers}")
    print(f"  Devices: {devices}")
    print(f"  Runs per config: {n_runs}")
    print(f"  Init temp: {init_temp}")
    print(f"  Stop temp: {stop_temp}")
    print()
    
    # Build task list
    tasks = []
    worker_id = 0
    
    for problem_name, problem_config in experiments.items():
        problem_sizes = problem_config.get("problem_sizes", [])
        K_values = problem_config.get("K", [])
        nsa_model = problem_config.get("nsa")
        rlbsa_model = problem_config.get("rlbsa")
        
        for problem_dim in problem_sizes:
            for K in K_values:
                # Baseline task (uses NSA model with baseline=True)
                if nsa_model:
                    nsa_model_path = get_model_path(project_root, nsa_model)
                    device = devices[worker_id % len(devices)]
                    tasks.append((
                        problem_name, problem_dim, K, "baseline", nsa_model_path,
                        init_temp, stop_temp, device, worker_id, n_runs, embed_dim
                    ))
                    worker_id += 1
                
                # NSA task
                if nsa_model:
                    nsa_model_path = get_model_path(project_root, nsa_model)
                    device = devices[worker_id % len(devices)]
                    tasks.append((
                        problem_name, problem_dim, K, "nsa", nsa_model_path,
                        init_temp, stop_temp, device, worker_id, n_runs, embed_dim
                    ))
                    worker_id += 1
                
                # RLBSA task
                if rlbsa_model:
                    rlbsa_model_path = get_model_path(project_root, rlbsa_model)
                    device = devices[worker_id % len(devices)]
                    tasks.append((
                        problem_name, problem_dim, K, "rlbsa", rlbsa_model_path,
                        init_temp, stop_temp, device, worker_id, n_runs, embed_dim
                    ))
                    worker_id += 1
    
    print(f"Total tasks: {len(tasks)}")
    print(f"Running with {num_workers} workers...")
    print()
    
    # Run evaluations in parallel
    if num_workers > 1:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(evaluate_config, tasks)
    else:
        results = [evaluate_config(task) for task in tasks]
    
    # Filter out failed tasks
    results = [r for r in results if r is not None]
    
    # Organize results
    organized_results = {}
    for result in results:
        problem = result["problem"]
        problem_dim = result["problem_dim"]
        K = result["K"]
        method = result["method"]
        
        key = f"{problem}_{problem_dim}_K{K}"
        if key not in organized_results:
            organized_results[key] = {}
        
        organized_results[key][method] = result
    
    # Print summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    for problem_name in experiments.keys():
        print(f"\n{problem_name.upper()}")
        print("-"*100)
        print(f"{'Size':<10} {'K':<10} {'Method':<10} {'Min':<12} {'Avg':<12} {'Std':<12} {'Time (s)':<12}")
        print("-"*100)
        
        problem_config = experiments[problem_name]
        problem_sizes = problem_config.get("problem_sizes", [])
        K_values = problem_config.get("K", [])
        
        for size in problem_sizes:
            for K in K_values:
                key = f"{problem_name}_{size}_K{K}"
                if key in organized_results:
                    # Print in order: baseline, nsa, rlbsa
                    for method in ["baseline", "nsa", "rlbsa"]:
                        if method in organized_results[key]:
                            r = organized_results[key][method]
                            print(f"{size:<10} {K:<10} {method:<10} {r['min']:<12.4f} {r['avg']:<12.4f} "
                                  f"{r['std']:<12.4f} {r['time_avg']:<12.2f}")
    
    # Save results
    output_dir = os.path.join(project_root, "outputs", "results")
    create_folder(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"simple_eval_{timestamp}.json")
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "results": organized_results,
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
