import numpy as np
import math
import copy
from typing import List, Tuple, Optional
from deg3_polynomial import LPNormsCoordinateDescent

class CubicPolynomialQueries:
    """
    Comprehensive testing and analysis for cubic polynomial queries.
    
    This class provides various test scenarios and analysis tools for the 
    cubic polynomial query optimization: q(x) = (1/n) * sum_i c_i * x_i^3
    where c_i ∈ (0,1] and x_i ∈ [-1,1].
    """
    
    def __init__(self, n: int = 100, epsilon: float = 10.0, delta: float = 1e-1, 
                 beta: float = 0.05, eta: float = 0.01):
        """
        Initialize the cubic polynomial query analyzer.
        
        Args:
            n: Number of data points
            epsilon: DP parameter
            delta: DP parameter
            beta: Failure probability
            eta: Edge for boosting
        """
        self.n = n
        self.epsilon = epsilon
        self.delta = delta
        self.beta = beta
        self.eta = eta
        
        # Common parameters
        self.upper_bound = 1.0
        self.lower_bound = -1.0
        self.tau = 0.01
        
    def generate_coefficient_scenarios(self) -> dict:
        """Generate different coefficient scenarios for testing."""
        scenarios = {}
        
        # Scenario 1: Uniform coefficients
        scenarios['uniform'] = np.ones(self.n)
        
        # Scenario 2: Random coefficients
        np.random.seed(42)
        scenarios['random'] = np.random.uniform(0.1, 1.0, self.n)
        
        # Scenario 3: Decaying coefficients (higher weight on first half)
        scenarios['decaying'] = np.concatenate([
            np.random.uniform(0.8, 1.0, self.n//2),
            np.random.uniform(0.1, 0.3, self.n//2)
        ])
        
        # Scenario 4: Alternating high/low coefficients
        scenarios['alternating'] = np.array([
            0.9 if i % 2 == 0 else 0.2 for i in range(self.n)
        ])
        
        # Scenario 5: Sparse coefficients (many small, few large)
        scenarios['sparse'] = np.random.uniform(0.05, 0.15, self.n)
        # Make 10% of coefficients large
        large_indices = np.random.choice(self.n, size=self.n//10, replace=False)
        scenarios['sparse'][large_indices] = np.random.uniform(0.8, 1.0, len(large_indices))
        
        return scenarios
    
    def generate_query_scenarios(self) -> dict:
        """Generate different query weight scenarios."""
        scenarios = {}
        
        # Scenario 1: Single query
        scenarios['single'] = {
            'k': 1,
            'weights': [1.0]
        }
        
        # Scenario 2: Multiple queries with equal weights
        scenarios['equal_weights'] = {
            'k': 5,
            'weights': [0.2, 0.2, 0.2, 0.2, 0.2]
        }
        
        # Scenario 3: Weighted queries (emphasize some queries)
        scenarios['weighted'] = {
            'k': 5,
            'weights': [0.4, 0.3, 0.15, 0.1, 0.05]
        }
        
        # Scenario 4: Many queries with uniform weights
        scenarios['many_queries'] = {
            'k': 20,
            'weights': [0.05] * 20
        }
        
        # Scenario 5: Few queries with concentrated weights
        scenarios['concentrated'] = {
            'k': 3,
            'weights': [0.6, 0.3, 0.1]
        }
        
        return scenarios
    
    def run_single_experiment(self, coeffs_matrix: np.ndarray, weights: List[float], 
                            max_iterations: int = 1000, verbose: bool = False) -> dict:
        """Run a single experiment with given parameters."""
        
        # Create optimizer
        optimizer = LPNormsCoordinateDescent(
            epsilon=self.epsilon,
            delta=self.delta,
            beta=self.beta,
            eta=self.eta,
            n=self.n,
            tau=self.tau,
            upper_bound=self.upper_bound,
            lower_bound=self.lower_bound
        )
        
        # Set coefficient matrix and weights
        optimizer.set_queries_and_coefficients(coeffs_matrix, weights)
        
        # Generate data
        optimizer.generate_data()
        
        # Run optimization
        results = optimizer.run_coordinate_descent(
            max_iterations=max_iterations, 
            verbose=verbose
        )
        
        # Add additional metrics
        results['coeff_stats'] = {
            'mean': np.mean(coeffs_matrix),
            'std': np.std(coeffs_matrix),
            'min': np.min(coeffs_matrix),
            'max': np.max(coeffs_matrix),
            'shape': coeffs_matrix.shape
        }
        
        results['query_output_stats'] = {
            'real_outputs': optimizer.real_output.copy() if len(optimizer.real_output) > 0 else [],
            'fake_outputs': optimizer.fake_output.copy() if len(optimizer.fake_output) > 0 else [],
            'noise_magnitudes': np.abs(optimizer.lap_noise).copy() if len(optimizer.lap_noise) > 0 else []
        }
        
        return results
    
    def run_comprehensive_test(self, max_iterations: int = 500, verbose: bool = True) -> dict:
        """Run comprehensive tests across different scenarios."""
        
        if verbose:
            print("=== Comprehensive Cubic Polynomial Query Testing ===")
            print(f"Parameters: n={self.n}, epsilon={self.epsilon}, delta={self.delta}")
            print(f"Rho = 2/n = {2.0/self.n:.6f}")
            print()
        
        # Generate scenarios
        coeff_scenarios = self.generate_coefficient_scenarios()
        query_scenarios = self.generate_query_scenarios()
        
        all_results = {}
        
        # Test each combination
        for coeff_name, base_coeffs in coeff_scenarios.items():
            for query_name, query_config in query_scenarios.items():
                scenario_name = f"{coeff_name}_{query_name}"
                
                # Generate coefficient matrix for this scenario
                k = query_config['k']
                weights = query_config['weights']
                
                # Create coefficient matrix where each query gets a variation of the base coefficients
                coeffs_matrix = np.zeros((k, self.n))
                for i in range(k):
                    if coeff_name == 'uniform':
                        # All queries get the same uniform coefficients
                        coeffs_matrix[i] = base_coeffs.copy()
                    else:
                        # Each query gets a slightly different version of the base coefficients
                        np.random.seed(42 + i)  # Different seed for each query
                        noise = np.random.normal(0, 0.1, self.n)
                        coeffs_matrix[i] = np.clip(base_coeffs + noise, 0.01, 1.0)
                
                if verbose:
                    print(f"Testing scenario: {scenario_name}")
                    print(f"  Coefficients: mean={np.mean(coeffs_matrix):.3f}, std={np.std(coeffs_matrix):.3f}")
                    print(f"  Queries: {k} queries, weights sum={sum(weights):.3f}")
                    print(f"  Coefficient matrix shape: {coeffs_matrix.shape}")
                
                try:
                    results = self.run_single_experiment(
                        coeffs_matrix=coeffs_matrix,
                        weights=weights,
                        max_iterations=max_iterations,
                        verbose=False
                    )
                    
                    all_results[scenario_name] = results
                    
                    if verbose:
                        print(f"  Result: {results['num_iterations']} iterations, "
                              f"satisfaction={results['final_weighted_satisfaction']:.3f}, "
                              f"converged={results['converged']}")
                        print()
                        
                except Exception as e:
                    if verbose:
                        print(f"  Error: {str(e)}")
                        print()
                    all_results[scenario_name] = {'error': str(e)}
        
        return all_results
    
    def analyze_results(self, results: dict) -> dict:
        """Analyze and summarize the results."""
        
        analysis = {
            'summary_stats': {},
            'best_scenarios': {},
            'worst_scenarios': {},
            'convergence_analysis': {}
        }
        
        # Filter out error results
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return analysis
        
        # Summary statistics
        iterations = [r['num_iterations'] for r in valid_results.values()]
        satisfactions = [r['final_weighted_satisfaction'] for r in valid_results.values()]
        converged_count = sum(1 for r in valid_results.values() if r['converged'])
        
        analysis['summary_stats'] = {
            'total_scenarios': len(results),
            'successful_scenarios': len(valid_results),
            'converged_scenarios': converged_count,
            'avg_iterations': np.mean(iterations),
            'std_iterations': np.std(iterations),
            'avg_satisfaction': np.mean(satisfactions),
            'std_satisfaction': np.std(satisfactions),
            'min_satisfaction': np.min(satisfactions),
            'max_satisfaction': np.max(satisfactions)
        }
        
        # Best and worst scenarios
        sorted_by_satisfaction = sorted(valid_results.items(), 
                                      key=lambda x: x[1]['final_weighted_satisfaction'], 
                                      reverse=True)
        
        analysis['best_scenarios'] = {
            'top_3': sorted_by_satisfaction[:3]
        }
        
        analysis['worst_scenarios'] = {
            'bottom_3': sorted_by_satisfaction[-3:]
        }
        
        # Convergence analysis
        fast_convergence = [k for k, v in valid_results.items() 
                          if v['num_iterations'] <= 50]
        slow_convergence = [k for k, v in valid_results.items() 
                          if v['num_iterations'] >= 200]
        
        analysis['convergence_analysis'] = {
            'fast_convergence': fast_convergence,
            'slow_convergence': slow_convergence,
            'convergence_rate': converged_count / len(valid_results) if valid_results else 0
        }
        
        return analysis
    
    def print_analysis_report(self, results: dict, analysis: dict):
        """Print a comprehensive analysis report."""
        
        print("\n" + "="*60)
        print("COMPREHENSIVE CUBIC POLYNOMIAL QUERY ANALYSIS REPORT")
        print("="*60)
        
        # Summary statistics
        stats = analysis['summary_stats']
        print(f"\nSUMMARY STATISTICS:")
        print(f"  Total scenarios tested: {stats['total_scenarios']}")
        print(f"  Successful scenarios: {stats['successful_scenarios']}")
        print(f"  Converged scenarios: {stats['converged_scenarios']}")
        print(f"  Convergence rate: {stats['converged_scenarios']/stats['successful_scenarios']*100:.1f}%")
        print(f"  Average iterations: {stats['avg_iterations']:.1f} ± {stats['std_iterations']:.1f}")
        print(f"  Average satisfaction: {stats['avg_satisfaction']:.3f} ± {stats['std_satisfaction']:.3f}")
        print(f"  Satisfaction range: [{stats['min_satisfaction']:.3f}, {stats['max_satisfaction']:.3f}]")
        
        # Best scenarios
        print(f"\nTOP 3 SCENARIOS (by satisfaction):")
        for i, (name, result) in enumerate(analysis['best_scenarios']['top_3'], 1):
            print(f"  {i}. {name}: satisfaction={result['final_weighted_satisfaction']:.3f}, "
                  f"iterations={result['num_iterations']}, converged={result['converged']}")
        
        # Worst scenarios
        print(f"\nBOTTOM 3 SCENARIOS (by satisfaction):")
        for i, (name, result) in enumerate(analysis['worst_scenarios']['bottom_3'], 1):
            print(f"  {i}. {name}: satisfaction={result['final_weighted_satisfaction']:.3f}, "
                  f"iterations={result['num_iterations']}, converged={result['converged']}")
        
        # Convergence analysis
        conv_analysis = analysis['convergence_analysis']
        print(f"\nCONVERGENCE ANALYSIS:")
        print(f"  Fast convergence (≤50 iterations): {len(conv_analysis['fast_convergence'])} scenarios")
        if conv_analysis['fast_convergence']:
            print(f"    {', '.join(conv_analysis['fast_convergence'])}")
        print(f"  Slow convergence (≥200 iterations): {len(conv_analysis['slow_convergence'])} scenarios")
        if conv_analysis['slow_convergence']:
            print(f"    {', '.join(conv_analysis['slow_convergence'])}")
        
        # Detailed results for each scenario
        print(f"\nDETAILED RESULTS:")
        for name, result in results.items():
            if 'error' in result:
                print(f"  {name}: ERROR - {result['error']}")
            else:
                print(f"  {name}: {result['num_iterations']} iterations, "
                      f"satisfaction={result['final_weighted_satisfaction']:.3f}, "
                      f"converged={result['converged']}, "
                      f"error={result['error_stats']['mean_error']:.6f}")

def test_nonconvex_behavior():
    """Test nonconvex behavior with larger query sets and no tau convergence."""
    print("=== Testing Nonconvex Cubic Polynomial Queries ===")
    print("NOTE: Cubic queries are NONCONVEX - expect imperfect convergence!")
    print()
    
    # Create analyzer with more challenging parameters
    analyzer = CubicPolynomialQueries(
        n=200,           # Larger number of data points
        epsilon=1.0,     # Very low epsilon (much more noise)
        delta=1e-3,      # Lower delta
        beta=0.2,        # Higher failure probability
        eta=0.1          # Higher edge requirement
    )
    
    # Test scenarios with larger query sets
    test_scenarios = {
        'many_queries_50': {
            'k': 50,
            'weights': [1.0/50] * 50  # Equal weights
        },
        'many_queries_100': {
            'k': 100,
            'weights': [1.0/100] * 100  # Equal weights
        },
        'weighted_many_50': {
            'k': 50,
            'weights': [0.5] + [0.5/49] * 49  # One dominant query
        },
        'sparse_queries_30': {
            'k': 30,
            'weights': [0.1] * 10 + [0.9/20] * 20  # 10 important, 20 less important
        },
        'extreme_queries_200': {
            'k': 200,
            'weights': [1.0/200] * 200  # Very many queries
        }
    }
    
    # Generate challenging coefficient patterns
    np.random.seed(123)  # Different seed for more challenging patterns
    n = analyzer.n
    
    coeff_patterns = {
        'highly_varying': np.random.uniform(0.01, 1.0, n),  # Very wide range
        'bimodal': np.concatenate([
            np.random.uniform(0.05, 0.2, n//2),   # Low coefficients
            np.random.uniform(0.8, 1.0, n//2)     # High coefficients
        ]),
        'exponential_decay': np.exp(-np.linspace(0, 3, n)) * 0.9 + 0.1,  # Exponential decay
        'random_sparse': np.random.choice([0.1, 0.9], n, p=[0.8, 0.2]),   # Mostly small, some large
        'extreme_bimodal': np.concatenate([
            np.random.uniform(0.01, 0.05, n//2),   # Very low coefficients
            np.random.uniform(0.95, 1.0, n//2)     # Very high coefficients
        ]),
        'chaotic': np.random.uniform(0.001, 1.0, n)  # Extremely wide range
    }
    
    all_results = {}
    
    for coeff_name, base_coeffs in coeff_patterns.items():
        for scenario_name, config in test_scenarios.items():
            full_name = f"{coeff_name}_{scenario_name}"
            
            # Generate coefficient matrix
            k = config['k']
            weights = config['weights']
            coeffs_matrix = np.zeros((k, n))
            
            for i in range(k):
                # Each query gets a different variation
                np.random.seed(123 + i)
                noise = np.random.normal(0, 0.2, n)  # Larger noise for more variation
                coeffs_matrix[i] = np.clip(base_coeffs + noise, 0.01, 1.0)
            
            print(f"Testing: {full_name}")
            print(f"  Queries: {k}, Data points: {n}")
            print(f"  Coefficient range: [{np.min(coeffs_matrix):.3f}, {np.max(coeffs_matrix):.3f}]")
            print(f"  Epsilon: {analyzer.epsilon}, Lambda will be calculated...")
            
            try:
                # Create optimizer with tau=0 (no tau convergence for nonconvex)
                optimizer = LPNormsCoordinateDescent(
                    epsilon=analyzer.epsilon,
                    delta=analyzer.delta,
                    beta=analyzer.beta,
                    eta=analyzer.eta,
                    n=analyzer.n,
                    tau=0.0,  # No tau convergence for nonconvex
                    upper_bound=analyzer.upper_bound,
                    lower_bound=analyzer.lower_bound
                )
                
                optimizer.set_queries_and_coefficients(coeffs_matrix, weights)
                optimizer.generate_data()
                
                # Run with more iterations to see nonconvex behavior
                results = optimizer.run_coordinate_descent(
                    max_iterations=1000,
                    verbose=False
                )
                
                all_results[full_name] = results
                
                print(f"  Result: {results['num_iterations']} iterations, "
                      f"satisfaction={results['final_weighted_satisfaction']:.3f}, "
                      f"converged={results['converged']}")
                print(f"  Final error: {results['error_stats']['mean_error']:.6f}")
                print()
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                print()
                all_results[full_name] = {'error': str(e)}
    
    # Analyze nonconvex results
    print("="*60)
    print("NONCONVEX BEHAVIOR ANALYSIS")
    print("="*60)
    
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    
    if valid_results:
        satisfactions = [r['final_weighted_satisfaction'] for r in valid_results.values()]
        iterations = [r['num_iterations'] for r in valid_results.values()]
        converged_count = sum(1 for r in valid_results.values() if r['converged'])
        
        print(f"Total scenarios: {len(all_results)}")
        print(f"Successful scenarios: {len(valid_results)}")
        print(f"Average satisfaction: {np.mean(satisfactions):.3f} ± {np.std(satisfactions):.3f}")
        print(f"Satisfaction range: [{np.min(satisfactions):.3f}, {np.max(satisfactions):.3f}]")
        print(f"Average iterations: {np.mean(iterations):.1f} ± {np.std(iterations):.1f}")
        print(f"Scenarios with tau convergence: {converged_count}/{len(valid_results)}")
        print()
        
        print("DETAILED RESULTS:")
        for name, result in all_results.items():
            if 'error' in result:
                print(f"  {name}: ERROR - {result['error']}")
            else:
                print(f"  {name}: {result['num_iterations']} iterations, "
                      f"satisfaction={result['final_weighted_satisfaction']:.3f}, "
                      f"converged={result['converged']}, "
                      f"error={result['error_stats']['mean_error']:.6f}")
    
    return all_results

def main():
    """Main function to run comprehensive cubic polynomial query tests."""
    
    # Run nonconvex behavior test
    print("Running nonconvex behavior test...")
    nonconvex_results = test_nonconvex_behavior()
    
    print("\n" + "="*60)
    print("RUNNING ORIGINAL COMPREHENSIVE TEST")
    print("="*60)
    
    # Also run original comprehensive test for comparison
    analyzer = CubicPolynomialQueries(
        n=100,           # Number of data points
        epsilon=10.0,    # DP parameter
        delta=1e-1,      # DP parameter
        beta=0.05,       # Failure probability
        eta=0.01         # Edge for boosting
    )
    
    # Run comprehensive tests
    results = analyzer.run_comprehensive_test(
        max_iterations=500,
        verbose=True
    )
    
    # Analyze results
    analysis = analyzer.analyze_results(results)
    
    # Print report
    analyzer.print_analysis_report(results, analysis)
    
    return results, analysis, nonconvex_results

if __name__ == "__main__":
    results, analysis, nonconvex_results = main()
