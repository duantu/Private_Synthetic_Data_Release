import numpy as np
import math
import copy
from typing import List, Tuple, Optional

class LPNormsCoordinateDescent:
    """
    Object-oriented version of the LP norms coordinate descent algorithm.
    
    This class implements a coordinate descent algorithm for optimizing synthetic data
    to match noisy query outputs while maintaining differential privacy guarantees.
    
    NOTE: This version is designed for CONVEX LP norms. For nonconvex functions,
    you'll want to remove the tau stopping criterion and only use satisfaction ratio.
    """
    
    def __init__(self, 
                 epsilon: float,
                 delta: float, 
                 beta: float,
                 eta: float,
                 n: int,
                 tau: float = 0.01,
                 upper_bound: float = 1.0,
                 lower_bound: float = -1.0,
                 data_precision: int = 4,
                 ub_on_p: int = 20):
        """
        Initialize the coordinate descent algorithm with DP parameters.
        
        Args:
            epsilon: DP parameter
            delta: DP parameter  
            beta: Failure probability of the base synopsis
            eta: Edge for boosting
            n: Number of data points
            tau: Loss update stopping criterion (USED for convex LP norms)
            upper_bound: Data upper bound
            lower_bound: Data lower bound
            data_precision: Data precision
            ub_on_p: Upper bound on p values for LP norms
        """
        self.epsilon = epsilon
        self.delta = delta
        self.beta = beta
        self.eta = eta
        self.n = n
        self.tau = tau
        self.num_points = 2 * n
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.data_precision = data_precision
        self.ub_on_p = ub_on_p
        
        # Calculate derived parameters
        # For cubic queries over n points, we will use n points directly and rho = (upper - lower) / n
        self.num_points = n
        self.m = self.num_points * math.ceil(math.log2(((upper_bound - lower_bound)/10**(-data_precision)) + 1))
        self.rho = (upper_bound - lower_bound) / self.n
        
        # Initialize data structures
        self.real_X = None
        self.fake_X = None
        self.fake_X_original = None
        self.sampled_queries = None
        self.query_weights = None
        self.real_output = None
        self.real_data_noisy_output = None
        self.lap_noise = None
        self.fake_output = None
        self.error = None
        self.lambda_val = None  # Will be calculated when queries are set
        self.coeffs_matrix = None  # Matrix of c_i vectors for each query: shape (k, n)
        
    def set_queries_and_coefficients(self, coeffs_matrix: np.ndarray, weights: List[float]):
        """
        Set the coefficient matrix and weights for cubic polynomial queries.
        
        Args:
            coeffs_matrix: Matrix of coefficient vectors for each query, shape (k, n)
                          Each row represents c_i coefficients for one query
            weights: List of weights for each query (must sum to 1)
        """
        coeffs_matrix = np.asarray(coeffs_matrix, dtype=float)
        k, n_coeffs = coeffs_matrix.shape
        
        if len(weights) != k:
            raise ValueError("Number of coefficient vectors must match number of weights")
        
        if not np.isclose(sum(weights), 1.0, atol=1e-6):
            raise ValueError("Weights must sum to 1")
        
        if n_coeffs != self.num_points:
            raise ValueError(f"Coefficient vectors length {n_coeffs} must equal num_points {self.num_points}")
        
        if np.any(coeffs_matrix <= 0) or np.any(coeffs_matrix > 1):
            raise ValueError("All coefficients must be in (0, 1]")
        
        self.coeffs_matrix = coeffs_matrix.copy()
        self.query_weights = np.array(weights)
        self.k = k
        
        # For compatibility, create dummy queries array (not used in computation)
        self.sampled_queries = np.array([3.0] * k)  # All cubic queries
        
        # Calculate lambda according to your formula: lambda = ln(k/beta) * 2 * rho * sqrt(2k * ln(1/delta)) / epsilon
        self.lambda_val = (math.log(self.k / self.beta) * 2 * self.rho * 
                          math.sqrt(2 * self.k * math.log(1 / self.delta))) / self.epsilon
        
        print(f"Calculated lambda: {self.lambda_val:.6f}")
        print(f"Set {k} cubic polynomial queries with different coefficient vectors")
        
    def generate_data(self, real_data: Optional[np.ndarray] = None, 
                     initial_fake_data: Optional[np.ndarray] = None):
        """
        Generate or set the real and fake data.
        
        Args:
            real_data: Real data array (if None, generates random data)
            initial_fake_data: Initial fake data array (if None, generates random data)
        """
        if real_data is not None:
            self.real_X = real_data.copy()
        else:
            self.real_X = np.random.choice(
                np.arange(self.lower_bound, self.upper_bound + 0.01, 0.01), 
                self.num_points, replace=False
            )
        
        if initial_fake_data is not None:
            self.fake_X = initial_fake_data.copy()
        else:
            self.fake_X = np.random.randn(self.num_points)
        
        self.fake_X_original = self.fake_X.copy()
        
    def compute_query_outputs(self):
        """Compute query outputs on real and fake data, including noise."""
        if self.coeffs_matrix is None:
            raise ValueError("Must set coefficient matrix first")
        
        k = self.k
        
        # Initialize arrays
        self.real_output = np.zeros(k)
        self.real_data_noisy_output = np.zeros(k)
        self.lap_noise = np.zeros(k)
        self.fake_output = np.zeros(k)
        self.error = np.zeros(k)
        
        # Compute outputs for each query
        for index in range(k):
            # Get coefficient vector for this query
            coeffs = self.coeffs_matrix[index]
            
            # Real data output: (1/n) * sum_i c_i * x_i^3
            sum_real = np.sum(coeffs * (self.real_X ** 3))
            self.real_output[index] = sum_real / self.n
            
            # Add Laplace noise: Laplace(rho * sqrt(2k * ln(1/delta)) / epsilon), with rho = (upper-lower)/n
            noise_scale = self.rho * math.sqrt(2 * self.k * math.log(1 / self.delta)) / self.epsilon
            self.lap_noise[index] = np.random.laplace(loc=0.0, scale=noise_scale)
            self.real_data_noisy_output[index] = self.real_output[index] + self.lap_noise[index]
            
            # Fake data output: (1/n) * sum_i c_i * x_i^3
            sum_fake = np.sum(coeffs * (self.fake_X ** 3))
            self.fake_output[index] = sum_fake / self.n
            
            # Calculate error
            self.error[index] = abs(self.fake_output[index] - self.real_data_noisy_output[index])
    
    def compute_weighted_satisfaction_ratio(self) -> float:
        """
        Compute the weighted ratio of queries satisfying error < lambda/2.
        
        Returns:
            Weighted ratio of satisfied queries
        """
        if self.query_weights is None:
            raise ValueError("Must set queries and weights first")
        
        # Satisfaction criterion: error < lambda/2
        satisfied_mask = self.error < self.lambda_val / 2
        weighted_satisfied = np.sum(self.query_weights[satisfied_mask])
        
        return weighted_satisfied
    
    def coordinate_descent_step(self) -> Tuple[float, float]:
        """
        Perform one step of coordinate descent.
        
        Returns:
            Tuple of (total_loss, total_loss_update)
        """
        if self.coeffs_matrix is None:
            raise ValueError("Must set coefficient matrix first")
        
        k = self.k
        
        # Calculate current total loss
        total_loss_plus = np.sum(np.exp((self.fake_output - self.real_data_noisy_output) / self.lambda_val - 1))
        total_loss_minus = np.sum(np.exp((self.real_data_noisy_output - self.fake_output) / self.lambda_val - 1))
        total_loss = total_loss_plus + total_loss_minus
        
        # Compute gradients
        loss_gradient = np.zeros(self.num_points)
        
        for i in range(self.num_points):
            xi = self.fake_X[i]
            
            # Calculate gradient for each query individually
            gradient_sum = 0.0
            for j in range(k):
                # Get coefficient for this query and coordinate
                c_i = self.coeffs_matrix[j, i]
                
                # dq/dx_i = (3/n) * c_i * x_i^2
                query_grad_wrt_xi = (3.0 / self.n) * c_i * (xi ** 2)
                
                # Calculate loss gradient components for this query
                loss_gradient_plus = (query_grad_wrt_xi * 
                                    np.exp((self.fake_output[j] - self.real_data_noisy_output[j]) / self.lambda_val - 1) / 
                                    self.lambda_val)
                loss_gradient_minus = (-query_grad_wrt_xi * 
                                     np.exp(-(self.fake_output[j] - self.real_data_noisy_output[j]) / self.lambda_val - 1) / 
                                     self.lambda_val)
                
                gradient_sum += loss_gradient_plus + loss_gradient_minus
            
            loss_gradient[i] = gradient_sum
        
        # Find coordinate with maximum gradient
        x_coord_descent = np.argmax(np.abs(loss_gradient))
        
        # Update coordinate using Newton's method with backtracking line search
        xi = self.fake_X[x_coord_descent]
        
        # Calculate Hessian for the chosen coordinate
        hessian_sum = 0.0
        for j in range(k):
            # Get coefficient for this query and coordinate
            c_i = self.coeffs_matrix[j, x_coord_descent]
            
            # dq/dx_i = (3/n) * c_i * x_i^2, d2q/dx_i2 = (6/n) * c_i * x_i
            query_grad_wrt_xi = (3.0 / self.n) * c_i * (xi ** 2)
            query_hessian_wrt_xi = (6.0 / self.n) * c_i * xi
            
            # Calculate loss Hessian components for this query
            loss_hessian_plus = (np.exp((self.fake_output[j] - self.real_data_noisy_output[j]) / self.lambda_val - 1) * 
                               (query_hessian_wrt_xi + query_grad_wrt_xi**2 / self.lambda_val) / 
                               self.lambda_val)
            loss_hessian_minus = (-np.exp(-(self.fake_output[j] - self.real_data_noisy_output[j]) / self.lambda_val - 1) * 
                                (query_hessian_wrt_xi - query_grad_wrt_xi**2 / self.lambda_val) / 
                                self.lambda_val)
            
            hessian_sum += loss_hessian_plus + loss_hessian_minus
        
        loss_hessian_wrt_chosen_x_coord = hessian_sum
        
        # Backtracking line search
        t = 1
        alpha1 = 0.05
        alpha2 = 0.5
        
        fake_X_copy_linesearch = self.fake_X.copy()
        
        while True:
            # Calculate f(x + t*Delta_x)
            fake_X_copy_linesearch[x_coord_descent] = xi - t * loss_gradient[x_coord_descent] / loss_hessian_wrt_chosen_x_coord
            
            # Update fake output
            fake_output_linesearch = np.zeros(k)
            for index in range(k):
                coeffs = self.coeffs_matrix[index]
                sum_fake_linesearch = np.sum(coeffs * (fake_X_copy_linesearch ** 3))
                fake_output_linesearch[index] = sum_fake_linesearch / self.n
            
            loss_current_stepsize_plus = np.sum(np.exp((fake_output_linesearch - self.real_data_noisy_output) / self.lambda_val - 1))
            loss_current_stepsize_minus = np.sum(np.exp((self.real_data_noisy_output - fake_output_linesearch) / self.lambda_val - 1))
            loss_current_stepsize = loss_current_stepsize_plus + loss_current_stepsize_minus
            
            # Calculate f(x) + alpha1*t*grad_f*Delta_x
            loss_damped_stepsize = (total_loss + alpha1 * t * loss_gradient[x_coord_descent] * 
                                  (-loss_gradient[x_coord_descent] / loss_hessian_wrt_chosen_x_coord))
            
            if loss_current_stepsize > loss_damped_stepsize:
                t = t * alpha2
            else:
                break
        
        # Update coordinate
        xi_update = -t * loss_gradient[x_coord_descent] / loss_hessian_wrt_chosen_x_coord
        self.fake_X[x_coord_descent] = xi + xi_update
        
        # Update fake output
        for index in range(k):
            coeffs = self.coeffs_matrix[index]
            sum_fake = np.sum(coeffs * (self.fake_X ** 3))
            self.fake_output[index] = sum_fake / self.n
        
        # Update error
        self.error = abs(self.fake_output - self.real_data_noisy_output)
        
        # Calculate new total loss
        total_loss_plus_after = np.sum(np.exp((self.fake_output - self.real_data_noisy_output) / self.lambda_val - 1))
        total_loss_minus_after = np.sum(np.exp((self.real_data_noisy_output - self.fake_output) / self.lambda_val - 1))
        total_loss_after = total_loss_plus_after + total_loss_minus_after
        
        total_loss_update = total_loss - total_loss_after
        
        return total_loss_after, total_loss_update
    
    def run_coordinate_descent(self, max_iterations: int = 1000, verbose: bool = True) -> dict:
        """
        Run the coordinate descent algorithm until convergence.
        
        For CONVEX LP norms: Stops when BOTH conditions are met:
        1. Satisfaction ratio reaches 0.5+eta AND
        2. Total loss update < tau
        
        Args:
            max_iterations: Maximum number of iterations
            verbose: Whether to print progress
            
        Returns:
            Dictionary with results and statistics
        """
        if self.coeffs_matrix is None or self.query_weights is None:
            raise ValueError("Must set coefficient matrix and weights first")
        
        # Compute initial query outputs
        self.compute_query_outputs()
        
        # Calculate target satisfaction ratio
        target_ratio = 0.5 + self.eta
        
        # Initialize tracking variables
        num_iterations = 0
        total_loss_update = float('inf')
        
        # Store initial statistics
        initial_weighted_satisfaction = self.compute_weighted_satisfaction_ratio()
        
        if verbose:
            print(f"Initial weighted satisfaction ratio: {initial_weighted_satisfaction:.4f}")
            print(f"Target weighted satisfaction ratio: {target_ratio:.4f}")
            print(f"Lambda: {self.lambda_val:.6f}")
            print(f"Lambda/2: {self.lambda_val/2:.6f}")
            print(f"Tau: {self.tau}")
            print(f"Number of queries: {self.k}")
            print(f"Number of queries above lambda/2 error: {np.sum(self.error > self.lambda_val/2)}")
            print("NOTE: For CONVEX LP norms - algorithm stops when BOTH conditions are met:")
            print("  1. Satisfaction ratio >= 0.5 + eta")
            print("  2. Total loss update < tau")
        
        # Main coordinate descent loop - stops when BOTH conditions are met for convex case
        while (num_iterations < max_iterations and 
               (self.compute_weighted_satisfaction_ratio() < target_ratio or 
                total_loss_update > self.tau)):
            
            total_loss, total_loss_update = self.coordinate_descent_step()
            num_iterations += 1
            
            if verbose and num_iterations % 10 == 0:
                current_satisfaction = self.compute_weighted_satisfaction_ratio()
                queries_above_threshold = np.sum(self.error > self.lambda_val/2)
                print(f"Iteration {num_iterations}: "
                      f"Loss update = {total_loss_update:.6f}, "
                      f"Weighted satisfaction = {current_satisfaction:.4f}, "
                      f"Queries above lambda/2 = {queries_above_threshold}")
        
        # Final statistics
        final_weighted_satisfaction = self.compute_weighted_satisfaction_ratio()
        final_error_stats = {
            'mean_error': np.mean(self.error),
            'max_error': np.max(self.error),
            'queries_above_lambda_half': np.sum(self.error > self.lambda_val / 2)
        }
        
        results = {
            'num_iterations': num_iterations,
            'converged': total_loss_update <= self.tau,
            'target_reached': final_weighted_satisfaction >= target_ratio,
            'both_conditions_met': (final_weighted_satisfaction >= target_ratio and total_loss_update <= self.tau),
            'initial_weighted_satisfaction': initial_weighted_satisfaction,
            'final_weighted_satisfaction': final_weighted_satisfaction,
            'target_ratio': target_ratio,
            'final_loss': total_loss,
            'final_loss_update': total_loss_update,
            'lambda_val': self.lambda_val,
            'tau': self.tau,
            'error_stats': final_error_stats,
            'fake_X': self.fake_X.copy(),
            'real_X': self.real_X.copy(),
            'queries': self.sampled_queries.copy(),
            'weights': self.query_weights.copy(),
            'errors': self.error.copy(),
            'lap_noise': self.lap_noise.copy()
        }
        
        if verbose:
            print(f"\nAlgorithm completed:")
            print(f"  Iterations: {num_iterations}")
            print(f"  Converged (loss update <= tau): {results['converged']}")
            print(f"  Target reached (weighted satisfaction >= 0.5 + eta): {results['target_reached']}")
            print(f"  Both conditions met: {results['both_conditions_met']}")
            print(f"  Final weighted satisfaction: {final_weighted_satisfaction:.4f}")
            print(f"  Final loss update: {total_loss_update:.6f}")
            print(f"  Queries above lambda/2: {final_error_stats['queries_above_lambda_half']}")
        
        return results

# Example usage function
def run_lp_norms_optimization(queries: List[float], 
                             weights: List[float],
                             epsilon: float = 15.0,
                             delta: float = 1e-1,
                             beta: float = 0.05,
                             eta: float = 0.001,
                             n: int = 100,
                             tau: float = 0.01,
                             upper_bound: float = 1.0,
                             lower_bound: float = -1.0,
                             max_iterations: int = 1000,
                             verbose: bool = True) -> dict:
    """
    Convenience function to run the LP norms coordinate descent optimization.
    
    Args:
        queries: List of p values for LP norms
        weights: List of weights for each query
        epsilon: DP parameter
        delta: DP parameter
        beta: Failure probability
        eta: Edge for boosting
        n: Number of data points
        tau: Loss update stopping criterion (USED for convex LP norms)
        upper_bound: Data upper bound
        lower_bound: Data lower bound
        max_iterations: Maximum iterations
        verbose: Whether to print progress
        
    Returns:
        Dictionary with optimization results
    """
    # Create optimizer instance
    optimizer = LPNormsCoordinateDescent(
        epsilon=epsilon,
        delta=delta,
        beta=beta,
        eta=eta,
        n=n,
        tau=tau,
        upper_bound=upper_bound,
        lower_bound=lower_bound
    )
    
    # Set coefficient matrix and weights (for compatibility, create dummy coefficient matrix)
    # This is a legacy function - use set_queries_and_coefficients for proper functionality
    coeffs_matrix = np.ones((len(queries), n))  # All ones for compatibility
    optimizer.set_queries_and_coefficients(coeffs_matrix, weights)
    
    # Generate data
    optimizer.generate_data()
    
    # Run optimization
    results = optimizer.run_coordinate_descent(max_iterations=max_iterations, verbose=verbose)
    
    return results

# Test function for cubic polynomial queries
def test_cubic_polynomial_queries():
    """Test the updated cubic polynomial query functionality."""
    print("=== Testing Cubic Polynomial Queries ===")
    
    # Parameters
    n = 50  # Number of data points
    k = 3   # Number of queries (all use same cubic form, different weights)
    
    # Create optimizer instance
    optimizer = LPNormsCoordinateDescent(
        epsilon=10.0,
        delta=1e-1,
        beta=0.05,
        eta=0.01,
        n=n,
        tau=0.01,
        upper_bound=1.0,
        lower_bound=-1.0
    )
    
    # Set coefficient matrix and weights (k queries, each with different coefficient vectors)
    k = 3  # Number of queries
    weights = [0.4, 0.3, 0.3]  # Different weights for aggregation
    
    # Generate different coefficient vectors for each query
    np.random.seed(42)  # For reproducibility
    coeffs_matrix = np.zeros((k, n))
    for i in range(k):
        # Each query gets a different random coefficient vector
        coeffs_matrix[i] = np.random.uniform(0.1, 1.0, n)
    
    optimizer.set_queries_and_coefficients(coeffs_matrix, weights)
    
    print(f"Number of data points: {n}")
    print(f"Number of queries: {k}")
    print(f"Rho = (upper - lower) / n = {2.0 / n:.6f}")
    print(f"Lambda: {optimizer.lambda_val:.6f}")
    print(f"Coefficients range: [{np.min(coeffs_matrix):.3f}, {np.max(coeffs_matrix):.3f}]")
    print(f"Sample coefficients for query 0: {coeffs_matrix[0, :5]}")
    print(f"Sample coefficients for query 1: {coeffs_matrix[1, :5]}")
    print(f"Sample coefficients for query 2: {coeffs_matrix[2, :5]}")
    
    # Generate data
    optimizer.generate_data()
    
    # Compute initial query outputs
    optimizer.compute_query_outputs()
    
    print(f"\nInitial Results:")
    print(f"Real data query output: {optimizer.real_output[0]:.6f}")
    print(f"Noisy real data output: {optimizer.real_data_noisy_output[0]:.6f}")
    print(f"Fake data query output: {optimizer.fake_output[0]:.6f}")
    print(f"Laplace noise added: {optimizer.lap_noise[0]:.6f}")
    print(f"Initial error: {optimizer.error[0]:.6f}")
    print(f"Lambda/2 threshold: {optimizer.lambda_val/2:.6f}")
    
    # Run a few coordinate descent steps
    print(f"\nRunning coordinate descent...")
    for i in range(5):
        total_loss, loss_update = optimizer.coordinate_descent_step()
        satisfaction = optimizer.compute_weighted_satisfaction_ratio()
        print(f"Step {i+1}: Loss = {total_loss:.6f}, Loss update = {loss_update:.6f}, "
              f"Satisfaction = {satisfaction:.4f}, Error = {optimizer.error[0]:.6f}")
    
    print(f"\nFinal Results:")
    print(f"Final fake data query output: {optimizer.fake_output[0]:.6f}")
    print(f"Final error: {optimizer.error[0]:.6f}")
    print(f"Final satisfaction ratio: {optimizer.compute_weighted_satisfaction_ratio():.4f}")
    
    # Verify the query computation manually
    for i in range(k):
        manual_query = np.sum(coeffs_matrix[i] * (optimizer.fake_X ** 3)) / n
        print(f"Manual verification - query {i} output: {manual_query:.6f}")
        print(f"Difference from computed: {abs(manual_query - optimizer.fake_output[i]):.10f}")
    
    return optimizer

# Example usage
if __name__ == "__main__":
    # Test the new cubic polynomial functionality
    test_cubic_polynomial_queries()

"""
REMINDER FOR NONCONVEX CASE:

When you work on nonconvex functions later, remember to:

1. Change the stopping condition to only use satisfaction ratio:
   while (num_iterations < max_iterations and 
          self.compute_weighted_satisfaction_ratio() < target_ratio):

2. Remove tau from convergence check:
   'converged': False,  # Not applicable for nonconvex case

3. Update verbose output to reflect nonconvex behavior

The key changes for the convex LP norms case:
- Algorithm stops when BOTH conditions are met: satisfaction ratio ≥ 0.5+eta AND total loss update < tau
- Added 'both_conditions_met' field in results to track when both stopping criteria are satisfied
- Clear documentation about the dual stopping condition for convex functions
- Updated verbose output to show both conditions

This makes sense because for convex functions, you want to ensure both that you've reached 
the satisfaction target AND that the optimization has converged (small loss updates), but for 
nonconvex functions, you only care about the satisfaction target since convergence might not 
be achievable due to local minima.
"""