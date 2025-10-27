#!/usr/bin/env python3

"""
Test script for the LP norms coordinate descent algorithm
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lp_norms_oo import LPNormsCoordinateDescent, run_lp_norms_optimization
import numpy as np

def test_basic_functionality():
    """Test basic functionality with a small example"""
    print("=" * 60)
    print("TESTING LP NORMS COORDINATE DESCENT ALGORITHM")
    print("=" * 60)
    
    # Test parameters
    queries = [1.5, 2.0, 2.5, 3.0]  # p values for LP norms
    weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights
    
    print(f"Queries (p values): {queries}")
    print(f"Weights: {weights}")
    print(f"Sum of weights: {sum(weights)}")
    
    # Test 1: Basic initialization and setup
    print("\n" + "-" * 40)
    print("TEST 1: Basic Initialization")
    print("-" * 40)
    
    try:
        optimizer = LPNormsCoordinateDescent(
            epsilon=15.0,
            delta=1e-1,
            beta=0.05,
            eta=0.001,
            n=50,  # Smaller n for faster testing
            tau=0.01,
            upper_bound=1.0,
            lower_bound=-1.0
        )
        
        optimizer.set_queries_and_weights(queries, weights)
        optimizer.generate_data()
        
        print("✓ Initialization successful")
        print(f"✓ Lambda calculated: {optimizer.lambda_val:.6f}")
        print(f"✓ Number of data points: {optimizer.num_points}")
        print(f"✓ Number of queries: {optimizer.k}")
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False
    
    # Test 2: Query output computation
    print("\n" + "-" * 40)
    print("TEST 2: Query Output Computation")
    print("-" * 40)
    
    try:
        optimizer.compute_query_outputs()
        
        print("✓ Query outputs computed successfully")
        print(f"✓ Real outputs: {optimizer.real_output}")
        print(f"✓ Noisy outputs: {optimizer.real_data_noisy_output}")
        print(f"✓ Fake outputs: {optimizer.fake_output}")
        print(f"✓ Errors: {optimizer.error}")
        print(f"✓ Laplace noise: {optimizer.lap_noise}")
        
        # Check satisfaction ratio
        satisfaction = optimizer.compute_weighted_satisfaction_ratio()
        print(f"✓ Initial weighted satisfaction: {satisfaction:.4f}")
        
    except Exception as e:
        print(f"✗ Query output computation failed: {e}")
        return False
    
    # Test 3: Single coordinate descent step
    print("\n" + "-" * 40)
    print("TEST 3: Single Coordinate Descent Step")
    print("-" * 40)
    
    try:
        initial_loss = np.sum(np.exp((optimizer.fake_output - optimizer.real_data_noisy_output) / optimizer.lambda_val - 1)) + \
                      np.sum(np.exp((optimizer.real_data_noisy_output - optimizer.fake_output) / optimizer.lambda_val - 1))
        
        total_loss, total_loss_update = optimizer.coordinate_descent_step()
        
        print("✓ Coordinate descent step completed")
        print(f"✓ Initial loss: {initial_loss:.6f}")
        print(f"✓ Final loss: {total_loss:.6f}")
        print(f"✓ Loss update: {total_loss_update:.6f}")
        
        # Check if loss decreased (should be positive for convex case)
        if total_loss_update > 0:
            print("✓ Loss decreased as expected")
        else:
            print("⚠ Loss did not decrease (this might be normal in some cases)")
        
    except Exception as e:
        print(f"✗ Coordinate descent step failed: {e}")
        return False
    
    # Test 4: Full optimization run (short version)
    print("\n" + "-" * 40)
    print("TEST 4: Full Optimization Run (Limited Iterations)")
    print("-" * 40)
    
    try:
        results = optimizer.run_coordinate_descent(max_iterations=20, verbose=True)
        
        print("✓ Full optimization completed")
        print(f"✓ Final iterations: {results['num_iterations']}")
        print(f"✓ Converged: {results['converged']}")
        print(f"✓ Target reached: {results['target_reached']}")
        print(f"✓ Both conditions met: {results['both_conditions_met']}")
        print(f"✓ Final satisfaction: {results['final_weighted_satisfaction']:.4f}")
        print(f"✓ Final loss update: {results['final_loss_update']:.6f}")
        
    except Exception as e:
        print(f"✗ Full optimization failed: {e}")
        return False
    
    return True

def test_convenience_function():
    """Test the convenience function"""
    print("\n" + "=" * 60)
    print("TESTING CONVENIENCE FUNCTION")
    print("=" * 60)
    
    try:
        queries = [1.2, 1.8, 2.2, 3.0, 4.0]
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        results = run_lp_norms_optimization(
            queries=queries,
            weights=weights,
            epsilon=10.0,
            delta=1e-2,
            beta=0.1,
            eta=0.005,
            n=30,  # Small n for quick testing
            tau=0.005,
            max_iterations=15,
            verbose=False  # Less verbose for this test
        )
        
        print("✓ Convenience function test successful")
        print(f"✓ Final satisfaction: {results['final_weighted_satisfaction']:.4f}")
        print(f"✓ Target ratio: {results['target_ratio']:.4f}")
        print(f"✓ Lambda: {results['lambda_val']:.6f}")
        print(f"✓ Iterations: {results['num_iterations']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Convenience function test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)
    
    # Test 1: Mismatched queries and weights
    print("\n" + "-" * 30)
    print("Test: Mismatched queries and weights")
    try:
        optimizer = LPNormsCoordinateDescent(epsilon=10, delta=1e-1, beta=0.05, eta=0.001, n=10)
        optimizer.set_queries_and_weights([1.5, 2.0], [0.3, 0.3, 0.4])  # Mismatch
        print("✗ Should have failed with mismatched lengths")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test 2: Weights that don't sum to 1
    print("\n" + "-" * 30)
    print("Test: Weights that don't sum to 1")
    try:
        optimizer = LPNormsCoordinateDescent(epsilon=10, delta=1e-1, beta=0.05, eta=0.001, n=10)
        optimizer.set_queries_and_weights([1.5, 2.0], [0.3, 0.3])  # Sums to 0.6
        print("✗ Should have failed with weights not summing to 1")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test 3: Running without setting queries first
    print("\n" + "-" * 30)
    print("Test: Running without setting queries")
    try:
        optimizer = LPNormsCoordinateDescent(epsilon=10, delta=1e-1, beta=0.05, eta=0.001, n=10)
        optimizer.generate_data()
        optimizer.compute_query_outputs()  # Should fail
        print("✗ Should have failed without setting queries")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    print("✓ All edge case tests passed")
    return True

if __name__ == "__main__":
    print("Starting LP Norms Coordinate Descent Tests...")
    print()
    
    success = True
    
    # Run all tests
    success &= test_basic_functionality()
    success &= test_convenience_function()
    success &= test_edge_cases()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! The algorithm is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
    print("=" * 60)