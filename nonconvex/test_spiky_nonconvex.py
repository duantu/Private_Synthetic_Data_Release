# #!/usr/bin/env python3
# """
# Minimal test driver for spiky nonconvex optimization.
# Prints only: parameters, threshold (0.5+eta), initial satisfaction,
# and iterations to target.
# """

# import numpy as np
# import math
# import secrets
# from spiky_nonconvex_copy import SpikyNonconvexCoordinateDescent

# def test_spiky_nonconvex_queries(
#     n=400, epsilon=5.0, delta=1e-5, beta=0.05, eta=0.01,
#     k=30, max_iterations=1000, seed=None
# ):
#     # Pick a seed if none
#     if seed is None:
#         seed = secrets.randbits(32)

#     # Seed NumPy (your class uses np.random.* inside generate_data)
#     np.random.seed(seed)

#     # Create optimizer (no unsupported kwargs)
#     opt = SpikyNonconvexCoordinateDescent(
#         epsilon=epsilon,
#         delta=delta,
#         beta=beta,
#         eta=eta,
#         n=n,
#         tau=0.01,
#         upper_bound=math.pi,
#         lower_bound=-math.pi
#     )

#     # Random queries (amplitudes & freqs) using this seed
#     # CHANGED ranges to Option 2: amplitudes in [0.2, 1.0], frequencies in [4.0, 7.0]
#     amplitudes_matrix = np.random.uniform(0.1, 1.0, size=(k, n))
#     frequencies_vector = np.random.uniform(1, 10, size=k)
#     weights = [1.0 / k] * k

#     opt.set_queries_and_amplitudes(amplitudes_matrix, frequencies_vector, weights)

#     # ADDED: print DP noise scale and lambda/2 for sanity
#     noise_scale = opt.rho * math.sqrt(2 * k * math.log(1 / delta)) / epsilon
#     print(f"noise_scale={noise_scale:.6f}, lambda/2={opt.lambda_val/2:.6f}")

#     opt.generate_data()
#     opt.compute_query_outputs()

#     # Initial satisfaction (before optimization)
#     initial_satisfaction = float(opt.compute_weighted_satisfaction_ratio())

#     # Run optimization (quiet)
#     res = opt.run_coordinate_descent(max_iterations=max_iterations, verbose=False)

#     # Pull results (support either key name)
#     iterations = res.get("iterations", res.get("num_iterations", 0))
#     reached = res.get("target_reached", False)
#     final_satisfaction = float(opt.compute_weighted_satisfaction_ratio())
#     threshold = 0.5 + eta

#     print(
#         f"seed={seed} | n={n} k={k} eps={epsilon} delta={delta} beta={beta} eta={eta} "
#         f"| target={threshold:.3f} | initial_satisfaction={initial_satisfaction:.3f} "
#         f"| iters={iterations} | reached={reached} | final_satisfaction={final_satisfaction:.3f}"
#     )

#     return {
#         "seed": seed,
#         "n": n,
#         "k": k,
#         "epsilon": epsilon,
#         "delta": delta,
#         "beta": beta,
#         "eta": eta,
#         "target": threshold,
#         "iterations": iterations,
#         "reached": reached,
#         "initial_satisfaction": initial_satisfaction,
#         "final_satisfaction": final_satisfaction,
#         "lambda": float(opt.lambda_val),
#         "noise_scale": float(noise_scale),
#     }

# def test_multiple_runs(num_runs=5, **kwargs):
#     results = []
#     for i in range(num_runs):
#         # vary seed per run unless user provided one
#         seed = kwargs.get("seed", None)
#         r = test_spiky_nonconvex_queries(seed=(i if seed is None else seed), **kwargs)
#         results.append(r)
#     return results

# def test_parameter_sweep():
#     # You can keep this as is, or adjust if you want to sweep around the new defaults.
#     # for n in [500, 1000, 1500]:
#     #     test_spiky_nonconvex_queries(n=n, k=50, epsilon=3.0, delta=1e-6, beta=0.05, eta=0.01, max_iterations=1000)

#     for epsilon in [2.0, 3.0, 5.0, 8.0]:
#         test_spiky_nonconvex_queries(n=400, k=30, epsilon=epsilon, delta=1e-5, beta=0.05, eta=0.01, max_iterations=1000)

#     for k in [20, 50, 80]:
#         test_spiky_nonconvex_queries(n=400, k=k, epsilon=3.0, delta=1e-5, beta=0.05, eta=0.01, max_iterations=1000)

# if __name__ == "__main__":
#     # Single run (now uses Option 2 defaults)
#     test_spiky_nonconvex_queries()

#     # Multiple runs with different initializations (Option 2 defaults)
#     test_multiple_runs(num_runs=3, n=400, epsilon=3.0, k=30, delta=1e-5, beta=0.05, eta=0.01)

#     # Optional sweep
#     # test_parameter_sweep()

# #!/usr/bin/env python3
# """
# Test file for spiky nonconvex optimization.
# Allows testing with different parameter initializations.
# """

# import numpy as np
# import math
# import secrets
# from spiky_nonconvex import SpikyNonconvexCoordinateDescent

# def test_spiky_nonconvex_queries(n=200, epsilon=3.0, delta=1e-5, beta=0.1, eta=0.01,
#                                 k=20, max_iterations=1000, verbose=True, seed=None
#                                 ):
#     if seed is None:
#         seed = secrets.randbits(32)
#     """
#     Test the spiky nonconvex query functionality with configurable parameters.
    
#     Args:
#         n: Number of data points
#         epsilon: DP privacy parameter
#         delta: DP privacy parameter
#         beta: Failure probability
#         eta: Edge for boosting
#         k: Number of queries
#         max_iterations: Maximum optimization iterations
#         verbose: Whether to print detailed output
#         seed: Random seed for reproducibility
    
#     Returns:
#         Dictionary with test results
#     """
#     print(f"=== Testing Spiky Nonconvex Queries ===")
#     print(f"Parameters: n={n}, ε={epsilon}, δ={delta}, β={beta}, η={eta}, k={k}")
    
#     # Set random seed for reproducibility
#     np.random.seed(seed)
    
#     # Create optimizer instance
#     optimizer = SpikyNonconvexCoordinateDescent(
#         epsilon=epsilon,
#         delta=delta,
#         beta=beta,
#         eta=eta,
#         n=n,
#         tau=0.01,
#         upper_bound=math.pi,
#         lower_bound=-math.pi
#     )
    
#     # Generate amplitude matrix and frequencies for each query
#     amplitudes_matrix = np.zeros((k, n))
#     frequencies_vector = np.zeros(k)
#     for i in range(k):
#         # Each query gets a different random amplitude vector
#         amplitudes_matrix[i] = np.random.uniform(0.1, 1.0, n)
#         # Each query gets a different random frequency
#         frequencies_vector[i] = np.random.uniform(1.0, 10.0)
    
#     # Equal weights for all queries
#     weights = [1.0/k] * k
    
#     optimizer.set_queries_and_amplitudes(amplitudes_matrix, frequencies_vector, weights)
    
#     if verbose:
#         print(f"Number of data points: {n}")
#         print(f"Number of queries: {k}")
#         print(f"Frequency parameters w_i: {frequencies_vector}")
#         print(f"Rho = (upper_bound^2 + 1) / n = {optimizer.rho:.6f}")
#         print(f"Lambda: {optimizer.lambda_val:.6f}")
#         print(f"Amplitudes range: [{np.min(amplitudes_matrix):.3f}, {np.max(amplitudes_matrix):.3f}]")
#         print(f"Sample amplitudes for query 0: {amplitudes_matrix[0, :5]}")
#         print(f"Sample amplitudes for query 1: {amplitudes_matrix[1, :5]}")
#         print(f"Sample amplitudes for query 2: {amplitudes_matrix[2, :5]}")
#         print(f"Sample frequencies: {frequencies_vector[:5]}")
    
#     # Generate data
#     optimizer.generate_data()
    
#     # Compute initial query outputs
#     optimizer.compute_query_outputs()
    
#     if verbose:
#         print(f"\nInitial Results:")
#         print(f"Real data query output: {optimizer.real_output[0]:.6f}")
#         print(f"Noisy real data output: {optimizer.real_data_noisy_output[0]:.6f}")
#         print(f"Fake data query output: {optimizer.fake_output[0]:.6f}")
#         print(f"Laplace noise added: {optimizer.lap_noise[0]:.6f}")
#         print(f"Initial error: {optimizer.error[0]:.6f}")
#         print(f"Lambda/2 threshold: {optimizer.lambda_val/2:.6f}")
    
#     # Run optimization
#     if verbose:
#         print(f"\nRunning coordinate descent...")
    
#     results = optimizer.run_coordinate_descent(max_iterations=max_iterations, verbose=verbose)
    
#     if verbose:
#         print(f"\nFinal Results:")
#         print(f"Final fake data query output: {optimizer.fake_output[0]:.6f}")
#         print(f"Final error: {optimizer.error[0]:.6f}")
#         print(f"Final satisfaction ratio: {optimizer.compute_weighted_satisfaction_ratio():.4f}")
        
#         # Verify the query computation manually
#         print(f"\nManual verification:")
#         for i in range(min(5, k)):  # Show first 5 queries
#             manual_query = np.sum(optimizer.fake_X**2 + amplitudes_matrix[i] * np.sin(frequencies_vector[i] * optimizer.fake_X)) / n
#             print(f"Query {i}: computed={optimizer.fake_output[i]:.6f}, manual={manual_query:.6f}, diff={abs(manual_query - optimizer.fake_output[i]):.10f}")
    
#     # Return results summary
#     return {
#         'n': n,
#         'epsilon': epsilon,
#         'delta': delta,
#         'beta': beta,
#         'eta': eta,
#         'k': k,
#         'lambda': optimizer.lambda_val,
#         'rho': optimizer.rho,
#         'initial_error': optimizer.error[0],
#         'final_error': optimizer.error[0],
#         'initial_satisfaction': results.get('initial_satisfaction', 0),
#         'final_satisfaction': optimizer.compute_weighted_satisfaction_ratio(),
#         'iterations': results.get('iterations', 0),
#         'converged': results.get('converged', False),
#         'frequencies': frequencies_vector,
#         'amplitudes_range': [np.min(amplitudes_matrix), np.max(amplitudes_matrix)]
#     }

# def test_multiple_runs(num_runs=5, **kwargs):
#     """
#     Run multiple tests with different random seeds.
    
#     Args:
#         num_runs: Number of test runs
#         **kwargs: Parameters to pass to test_spiky_nonconvex_queries
    
#     Returns:
#         List of result dictionaries
#     """
#     print(f"=== Running {num_runs} Tests with Different Initializations ===")
    
#     results = []
#     for i in range(num_runs):
#         print(f"\n--- Test Run {i+1}/{num_runs} ---")
#         # Remove verbose from kwargs to avoid conflict
#         kwargs_copy = kwargs.copy()
#         kwargs_copy.pop('verbose', None)
#         result = test_spiky_nonconvex_queries(seed=i, verbose=False, **kwargs_copy)
#         results.append(result)
        
#         print(f"Run {i+1}: Error={result['final_error']:.6f}, Satisfaction={result['final_satisfaction']:.4f}, Iterations={result['iterations']}")
    
#     # Summary statistics
#     print(f"\n=== Summary Statistics ===")
#     errors = [r['final_error'] for r in results]
#     satisfactions = [r['final_satisfaction'] for r in results]
#     iterations = [r['iterations'] for r in results]
    
#     print(f"Final Error: mean={np.mean(errors):.6f}, std={np.std(errors):.6f}, min={np.min(errors):.6f}, max={np.max(errors):.6f}")
#     print(f"Final Satisfaction: mean={np.mean(satisfactions):.4f}, std={np.std(satisfactions):.4f}")
#     print(f"Iterations: mean={np.mean(iterations):.1f}, std={np.std(iterations):.1f}")
    
#     return results

# def test_parameter_sweep():
#     """
#     Test different parameter combinations.
#     """
#     print("=== Parameter Sweep Tests ===")
    
#     # Test different n values
#     print("\n--- Testing different n values ---")
#     for n in [50, 100, 200, 500]:
#         result = test_spiky_nonconvex_queries(n=n, epsilon=3.0, k=10, verbose=False)
#         print(f"n={n}: Error={result['final_error']:.6f}, Satisfaction={result['final_satisfaction']:.4f}, Lambda={result['lambda']:.3f}")
    
#     # Test different epsilon values

#     print("\n--- Testing different epsilon values ---")
#     for epsilon in [0.5, 1.0, 3.0, 5.0]:
#         result = test_spiky_nonconvex_queries(n=200, epsilon=epsilon, k=10, verbose=False)
#         print(f"ε={epsilon}: Error={result['final_error']:.6f}, Satisfaction={result['final_satisfaction']:.4f}, Lambda={result['lambda']:.3f}")
    
#     # Test different k values
#     print("\n--- Testing different k values ---")
#     for k in [5, 10, 20, 50]:
#         result = test_spiky_nonconvex_queries(n=200, epsilon=3.0, k=k, verbose=False)
#         print(f"k={k}: Error={result['final_error']:.6f}, Satisfaction={result['final_satisfaction']:.4f}, Lambda={result['lambda']:.3f}")

# if __name__ == "__main__":
#     # Example usage
    
#     # Single test with default parameters
#     print("Single test with default parameters:")
#     result = test_spiky_nonconvex_queries()
    
#     print("\n" + "="*60)
    
#     # Multiple runs with different seeds
#     print("Multiple runs with different initializations:")
#     results = test_multiple_runs(num_runs=3, n=200, epsilon=3.0, k=20)
    
#     print("\n" + "="*60)
    
#     # Parameter sweep
#     test_parameter_sweep()
#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Minimal test driver for spiky nonconvex optimization.
Prints only: parameters, threshold (0.5+eta), initial satisfaction,
and iterations to target.
"""
#!/usr/bin/env python3
"""
Minimal test driver for spiky nonconvex optimization.
Prints only: parameters, threshold (0.5+eta), initial satisfaction,
loss before/after, and iterations to target.
"""

#!/usr/bin/env python3
"""
Minimal test driver for spiky nonconvex optimization.
Now with an easier regime: n=150, eps=2.5, delta=1e-6, eta=0.005, k=25,
narrower amplitudes/frequencies, and modest iteration cap.
Also prints loss before/after.
"""
#!/usr/bin/env python3
"""
Minimal test driver for spiky nonconvex optimization.
Hard-but-tractable regime so optimization actually runs:
  n=180, eps=2.0, delta=1e-6, eta=0.05, k=40
  amplitudes in [0.4, 0.9], frequencies in [4.5, 7.5]
Prints loss before/after and satisfaction progress summary.
"""
import numpy as np
import math
import secrets
from spiky_nonconvex_copy import SpikyNonconvexCoordinateDescent

def _total_loss(fake_out, noisy_real_out, lam):
    return float(
        np.sum(np.exp((fake_out - noisy_real_out) / lam - 1.0)) +
        np.sum(np.exp((noisy_real_out - fake_out) / lam - 1.0))
    )

def test_spiky_nonconvex_queries(
    n=320, k=36, epsilon=3.0, delta=1e-6, beta=0.1, eta=0.01, max_iterations=5000, seed=None
):
    if seed is None:
        seed = secrets.randbits(32)
    np.random.seed(seed)  # drives amplitudes/frequencies

    opt = SpikyNonconvexCoordinateDescent(
        epsilon=epsilon,
        delta=delta,
        beta=beta,
        eta=eta,
        n=n,
        tau=0.01,
        upper_bound=math.pi,
        lower_bound=-math.pi,
        m_inner=6,
        shortlist_size=24,
        seed=seed,                 # <-- ensure real_X/fake_X use same seed family
    )

    amplitudes_matrix = np.random.uniform(0.2, 0.7, size=(k, n))
    frequencies_vector = np.random.uniform(4.5, 7.5, size=k)
    weights = [1.0 / k] * k

    opt.set_queries_and_amplitudes(amplitudes_matrix, frequencies_vector, weights)

    noise_scale = opt.rho * math.sqrt(2 * k * math.log(1 / delta)) / epsilon
    print(f"noise_scale={noise_scale:.6f}, lambda/2={opt.lambda_val/2:.6f}")

    opt.generate_data()
    opt.compute_query_outputs()  # one DP-noise draw fixed for this run

    initial_satisfaction = float(opt.compute_weighted_satisfaction_ratio())
    loss_before = _total_loss(opt.fake_output, opt.real_data_noisy_output, opt.lambda_val)
    target = 0.5 + eta

    # Do NOT resample noise during optimization
    res = opt.run_coordinate_descent(max_iterations=max_iterations, verbose=False, resample_noise=False)

    iterations = res.get("iterations", res.get("num_iterations", 0))
    reached = res.get("target_reached", False)
    final_satisfaction = float(opt.compute_weighted_satisfaction_ratio())
    loss_after = _total_loss(opt.fake_output, opt.real_data_noisy_output, opt.lambda_val)
    loss_delta = loss_before - loss_after

    print(f"loss_before={loss_before:.6f} | loss_after={loss_after:.6f} | loss_drop={loss_delta:.6f}")
    print(
        f"seed={seed} | n={n} k={k} eps={epsilon} delta={delta} beta={beta} eta={eta} "
        f"| target={target:.3f} | initial_satisfaction={initial_satisfaction:.3f} "
        f"| iters={iterations} | reached={reached} | final_satisfaction={final_satisfaction:.3f}"
    )

    return {
        "seed": seed,
        "n": n,
        "k": k,
        "epsilon": epsilon,
        "delta": delta,
        "beta": beta,
        "eta": eta,
        "target": target,
        "iterations": iterations,
        "reached": reached,
        "initial_satisfaction": initial_satisfaction,
        "final_satisfaction": final_satisfaction,
        "lambda": float(opt.lambda_val),
        "noise_scale": float(noise_scale),
        "loss_before": float(loss_before),
        "loss_after": float(loss_after),
        "loss_drop": float(loss_delta),
    }

def test_multiple_runs(num_runs=3, **kwargs):
    results = []
    for i in range(num_runs):
        seed = kwargs.get("seed", None)
        r = test_spiky_nonconvex_queries(seed=(i if seed is None else seed), **kwargs)
        results.append(r)
    return results

if __name__ == "__main__":
    # Single run
    test_spiky_nonconvex_queries()

    # Multiple runs (distinct seeds 0..9 if seed not supplied in kwargs)
    test_multiple_runs(num_runs=10, n=320, k=36, epsilon=3.0, delta=1e-6, beta=0.1, eta=0.01, max_iterations=5000)
