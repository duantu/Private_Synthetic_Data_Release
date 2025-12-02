#!/usr/bin/env python3
"""
Focused test to demonstrate nonconvex behavior of cubic polynomial queries.
"""

import numpy as np
from deg3_polynomial import LPNormsCoordinateDescent

def test_extreme_nonconvex_scenarios():
    """Test extreme scenarios that should show clear nonconvex behavior."""
    
    print("=== EXTREME NONCONVEX CUBIC POLYNOMIAL QUERY TESTING ===")
    print("Testing scenarios designed to show nonconvex behavior...")
    print()
    
    # Very challenging parameters
    n = 100
    epsilon = 0.5  # Extremely low epsilon (very high noise)
    delta = 1e-4
    beta = 0.3
    eta = 0.15  # High satisfaction requirement
    
    scenarios = [
        {
            'name': 'extreme_chaotic_50',
            'k': 50,
            'coeff_pattern': 'chaotic',
            'weights': [1.0/50] * 50
        },
        {
            'name': 'extreme_bimodal_100', 
            'k': 100,
            'coeff_pattern': 'extreme_bimodal',
            'weights': [1.0/100] * 100
        },
        {
            'name': 'weighted_chaotic_30',
            'k': 30,
            'coeff_pattern': 'chaotic',
            'weights': [0.7] + [0.3/29] * 29  # One very dominant query
        }
    ]
    
    # Generate extreme coefficient patterns
    np.random.seed(999)  # Different seed for extreme patterns
    
    coeff_patterns = {
        'chaotic': np.random.uniform(0.001, 1.0, n),  # Extremely wide range
        'extreme_bimodal': np.concatenate([
            np.random.uniform(0.001, 0.01, n//2),   # Extremely low
            np.random.uniform(0.99, 1.0, n//2)      # Extremely high
        ])
    }
    
    results = {}
    
    for scenario in scenarios:
        print(f"Testing: {scenario['name']}")
        print(f"  Queries: {scenario['k']}, Data points: {n}")
        print(f"  Epsilon: {epsilon} (very high noise)")
        print(f"  Eta: {eta} (high satisfaction requirement)")
        
        # Generate coefficient matrix
        k = scenario['k']
        weights = scenario['weights']
        base_coeffs = coeff_patterns[scenario['coeff_pattern']]
        
        coeffs_matrix = np.zeros((k, n))
        for i in range(k):
            np.random.seed(999 + i)
            noise = np.random.normal(0, 0.3, n)  # Large noise for variation
            coeffs_matrix[i] = np.clip(base_coeffs + noise, 0.001, 1.0)
        
        print(f"  Coefficient range: [{np.min(coeffs_matrix):.3f}, {np.max(coeffs_matrix):.3f}]")
        
        try:
            # Create optimizer with tau=0 (no tau convergence)
            optimizer = LPNormsCoordinateDescent(
                epsilon=epsilon,
                delta=delta,
                beta=beta,
                eta=eta,
                n=n,
                tau=0.0,  # No tau convergence for nonconvex
                upper_bound=1.0,
                lower_bound=-1.0
            )
            
            optimizer.set_queries_and_coefficients(coeffs_matrix, weights)
            optimizer.generate_data()
            
            # Run with many iterations
            result = optimizer.run_coordinate_descent(
                max_iterations=2000,
                verbose=False
            )
            
            results[scenario['name']] = result
            
            print(f"  Result: {result['num_iterations']} iterations")
            print(f"  Satisfaction: {result['final_weighted_satisfaction']:.3f} (target: {0.5 + eta:.3f})")
            print(f"  Converged (tau): {result['converged']}")
            print(f"  Mean error: {result['error_stats']['mean_error']:.6f}")
            print(f"  Max error: {result['error_stats']['max_error']:.6f}")
            print(f"  Queries above lambda/2: {result['error_stats']['queries_above_lambda_half']}/{k}")
            print()
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            print()
            results[scenario['name']] = {'error': str(e)}
    
    # Analysis
    print("="*60)
    print("NONCONVEX BEHAVIOR ANALYSIS")
    print("="*60)
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        satisfactions = [r['final_weighted_satisfaction'] for r in valid_results.values()]
        iterations = [r['num_iterations'] for r in valid_results.values()]
        mean_errors = [r['error_stats']['mean_error'] for r in valid_results.values()]
        
        print(f"Scenarios tested: {len(results)}")
        print(f"Successful scenarios: {len(valid_results)}")
        print()
        print(f"SATISFACTION ANALYSIS:")
        print(f"  Average satisfaction: {np.mean(satisfactions):.3f} ± {np.std(satisfactions):.3f}")
        print(f"  Satisfaction range: [{np.min(satisfactions):.3f}, {np.max(satisfactions):.3f}]")
        print(f"  Target satisfaction: {0.5 + eta:.3f}")
        print(f"  Scenarios meeting target: {sum(1 for s in satisfactions if s >= 0.5 + eta)}/{len(satisfactions)}")
        print()
        print(f"CONVERGENCE ANALYSIS:")
        print(f"  Average iterations: {np.mean(iterations):.1f} ± {np.std(iterations):.1f}")
        print(f"  Iteration range: [{np.min(iterations)}, {np.max(iterations)}]")
        print()
        print(f"ERROR ANALYSIS:")
        print(f"  Average mean error: {np.mean(mean_errors):.6f} ± {np.std(mean_errors):.6f}")
        print(f"  Error range: [{np.min(mean_errors):.6f}, {np.max(mean_errors):.6f}]")
        print()
        
        print("DETAILED RESULTS:")
        for name, result in results.items():
            if 'error' in result:
                print(f"  {name}: ERROR - {result['error']}")
            else:
                target_met = "✓" if result['final_weighted_satisfaction'] >= 0.5 + eta else "✗"
                print(f"  {name}: {result['num_iterations']} iterations, "
                      f"satisfaction={result['final_weighted_satisfaction']:.3f} {target_met}, "
                      f"error={result['error_stats']['mean_error']:.6f}")
    
    return results

if __name__ == "__main__":
    results = test_extreme_nonconvex_scenarios()
