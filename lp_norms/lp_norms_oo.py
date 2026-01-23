import numpy as np
import math
from typing import List, Tuple, Optional

# ----------------------------------------------------------------------
# Global defaults: change these ONCE and everything follows
# ----------------------------------------------------------------------
DEFAULT_LOWER_BOUND = -1.0
DEFAULT_UPPER_BOUND = 1.0
DEFAULT_DATA_PRECISION = 2  # step = 10^{-2} = 0.01


class LPNormsCoordinateDescent:
    """
    Object-oriented version of the LP norms coordinate descent algorithm.

    This class implements a coordinate descent algorithm for optimizing synthetic data
    to match noisy query outputs while maintaining differential privacy guarantees.

    NOTE: This version is designed for CONVEX LP norms. For nonconvex functions,
    you'll want to remove the tau stopping criterion and only use satisfaction ratio.

    Randomness:
        All randomness is handled by a local np.random.Generator (self.rng),
        so experiments are reproducible when you pass a fixed seed.
    """

    def __init__(self,
                 epsilon: float,
                 delta: float,
                 beta: float,
                 eta: float,
                 n: int,
                 tau: float = 0.01,
                 upper_bound: float = DEFAULT_UPPER_BOUND,
                 lower_bound: float = DEFAULT_LOWER_BOUND,
                 data_precision: int = DEFAULT_DATA_PRECISION,
                 ub_on_p: int = 20,
                 seed: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize the coordinate descent algorithm with DP parameters.

        Args:
            epsilon: DP parameter
            delta: DP parameter
            beta: Failure probability of the base synopsis
            eta: Edge for boosting
            n: Number of data points (num_points = 2n)
            tau: Loss update stopping criterion (USED for convex LP norms)
            upper_bound: Data upper bound
            lower_bound: Data lower bound
            data_precision: Number of decimal digits for the real_X grid
            ub_on_p: Upper bound on p values for LP norms
            seed: Optional seed for a local RNG (ignored if rng is provided)
            rng: Optional np.random.Generator to use (for full external control)
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

        # Local RNG (Option 3): all randomness flows through this
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng(seed)
        self.seed = seed  # just for debugging / introspection if you want it

        # Grid step used for generating real_X from a discrete grid
        self.grid_step = 10 ** (-self.data_precision)

        # Calculate derived parameters (kept consistent with your original expression)
        self.m = self.num_points * math.ceil(
            math.log2(((upper_bound - lower_bound) / self.grid_step) + 1)
        )
        self.rho = (upper_bound - lower_bound) / self.num_points

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

    def set_queries_and_weights(self, queries: List[float], weights: List[float]):
        """
        Set the queries and their corresponding weights.

        Args:
            queries: List of p values for LP norms (1 < p < ub_on_p)
            weights: List of weights for each query (must sum to 1)
        """
        if len(queries) != len(weights):
            raise ValueError("Number of queries must match number of weights")

        if not np.isclose(sum(weights), 1.0, atol=1e-6):
            raise ValueError("Weights must sum to 1")

        self.sampled_queries = np.array(queries, dtype=float)
        self.query_weights = np.array(weights, dtype=float)
        self.k = len(queries)

        # lambda = ln(k/beta) * 2 * rho * sqrt(2k * ln(1/delta)) / epsilon
        self.lambda_val = (math.log(self.k / self.beta) * 2 * self.rho *
                           math.sqrt(2 * self.k * math.log(1 / self.delta))) / self.epsilon

        print(f"Calculated lambda: {self.lambda_val:.6f}")

    def generate_data(self, real_data: Optional[np.ndarray] = None,
                      initial_fake_data: Optional[np.ndarray] = None):
        """
        Generate or set the real and fake data.

        Args:
            real_data: Real data array (if None, generates random data from a discrete grid)
            initial_fake_data: Initial fake data array (if None, generates N(0,1) data)
        """
        # Real data
        if real_data is not None:
            self.real_X = np.array(real_data, dtype=float).copy()
        else:
            # Discrete grid based on bounds + precision
            values = np.arange(
                self.lower_bound,
                self.upper_bound + self.grid_step,  # include upper bound
                self.grid_step
            )

            if self.num_points > len(values):
                raise ValueError(
                    f"Cannot sample num_points={self.num_points} distinct values "
                    f"from grid of size={len(values)} with replace=False. "
                    f"Either increase the range, decrease n, or increase data_precision."
                )

            # Use local RNG instead of global np.random
            self.real_X = self.rng.choice(values, size=self.num_points, replace=False)

        # Fake data
        if initial_fake_data is not None:
            self.fake_X = np.array(initial_fake_data, dtype=float).copy()
        else:
            # Use local RNG instead of global np.random.randn
            self.fake_X = self.rng.standard_normal(self.num_points)

        self.fake_X_original = self.fake_X.copy()

    def compute_query_outputs(self):
        """Compute query outputs on real and fake data, including noise."""
        if self.sampled_queries is None:
            raise ValueError("Must set queries and weights first")

        k = len(self.sampled_queries)

        # Initialize arrays
        self.real_output = np.zeros(k)
        self.real_data_noisy_output = np.zeros(k)
        self.lap_noise = np.zeros(k)
        self.fake_output = np.zeros(k)
        self.error = np.zeros(k)

        # Compute outputs for each query
        for index, p in enumerate(self.sampled_queries):
            # Real data output
            sum_real_xi_p = np.sum(np.abs(self.real_X) ** p)
            self.real_output[index] = sum_real_xi_p ** (1 / p) / self.num_points

            # Add Laplace noise: Laplace(rho * sqrt(2k * ln(1/delta)) / epsilon)
            noise_scale = self.rho * math.sqrt(2 * self.k * math.log(1 / self.delta)) / self.epsilon
            # Use local RNG instead of np.random.laplace
            self.lap_noise[index] = self.rng.laplace(loc=0.0, scale=noise_scale)
            self.real_data_noisy_output[index] = self.real_output[index] + self.lap_noise[index]

            # Fake data output
            sum_fake_xi_p = np.sum(np.abs(self.fake_X) ** p)
            self.fake_output[index] = sum_fake_xi_p ** (1 / p) / self.num_points

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

        satisfied_mask = self.error < self.lambda_val / 2
        weighted_satisfied = np.sum(self.query_weights[satisfied_mask])
        return float(weighted_satisfied)

    def coordinate_descent_step(self) -> Tuple[float, float]:
        """
        Perform one step of coordinate descent.

        Returns:
            Tuple of (total_loss, total_loss_update)
        """
        if self.sampled_queries is None:
            raise ValueError("Must set queries and weights first")

        k = len(self.sampled_queries)

        # Calculate current total loss
        total_loss_plus = np.sum(
            np.exp((self.fake_output - self.real_data_noisy_output) / self.lambda_val - 1)
        )
        total_loss_minus = np.sum(
            np.exp((self.real_data_noisy_output - self.fake_output) / self.lambda_val - 1)
        )
        total_loss = total_loss_plus + total_loss_minus

        # Compute gradients
        loss_gradient = np.zeros(self.num_points)

        for i in range(self.num_points):
            xi = self.fake_X[i]

            gradient_sum = 0.0
            for j, p in enumerate(self.sampled_queries):
                sum_fake_xi_p = np.sum(np.abs(self.fake_X) ** p)

                if sum_fake_xi_p > 0:
                    query_grad_wrt_xi = (
                        np.sign(xi)
                        * (np.abs(xi) / sum_fake_xi_p ** (1 / p)) ** (p - 1)
                        / self.num_points
                    )
                else:
                    query_grad_wrt_xi = 0.0

                loss_gradient_plus = (
                    query_grad_wrt_xi
                    * np.exp((self.fake_output[j] - self.real_data_noisy_output[j]) / self.lambda_val - 1)
                    / self.lambda_val
                )
                loss_gradient_minus = (
                    -query_grad_wrt_xi
                    * np.exp(-(self.fake_output[j] - self.real_data_noisy_output[j]) / self.lambda_val - 1)
                    / self.lambda_val
                )

                gradient_sum += loss_gradient_plus + loss_gradient_minus

            loss_gradient[i] = gradient_sum

        # Find coordinate with maximum gradient magnitude
        x_coord_descent = int(np.argmax(np.abs(loss_gradient)))
        xi = self.fake_X[x_coord_descent]

        # Calculate Hessian for the chosen coordinate
        hessian_sum = 0.0
        for j, p in enumerate(self.sampled_queries):
            sum_fake_xi_p = np.sum(np.abs(self.fake_X) ** p)

            if sum_fake_xi_p > 0:
                query_grad_wrt_xi = (
                    np.sign(xi)
                    * (np.abs(xi) / sum_fake_xi_p ** (1 / p)) ** (p - 1)
                    / self.num_points
                )

                query_hessian_wrt_xi = (
                    (
                        (1 / p - 1) * p * sum_fake_xi_p ** (1 / p - 2) * xi ** 2 * np.abs(xi) ** (2 * p - 4)
                        + sum_fake_xi_p ** (1 / p - 1) * (np.abs(xi) ** (p - 2) * xi ** 2 * np.abs(xi) ** (p - 4))
                    )
                    / self.num_points
                )

                loss_hessian_plus = (
                    np.exp((self.fake_output[j] - self.real_data_noisy_output[j]) / self.lambda_val - 1)
                    * (query_hessian_wrt_xi + query_grad_wrt_xi ** 2 / self.lambda_val)
                    / self.lambda_val
                )
                loss_hessian_minus = (
                    -np.exp(-(self.fake_output[j] - self.real_data_noisy_output[j]) / self.lambda_val - 1)
                    * (query_hessian_wrt_xi - query_grad_wrt_xi ** 2 / self.lambda_val)
                    / self.lambda_val
                )

                hessian_sum += loss_hessian_plus + loss_hessian_minus

        loss_hessian_wrt_chosen_x_coord = hessian_sum

        # Backtracking line search
        t = 1.0
        alpha1 = 0.05
        alpha2 = 0.5

        fake_X_copy_linesearch = self.fake_X.copy()

        while True:
            # Candidate update for chosen coordinate
            fake_X_copy_linesearch[x_coord_descent] = (
                xi - t * loss_gradient[x_coord_descent] / loss_hessian_wrt_chosen_x_coord
            )

            # Recompute fake outputs with candidate
            fake_output_linesearch = np.zeros(k)
            for index, p in enumerate(self.sampled_queries):
                sum_fake_xi_p_linesearch = np.sum(np.abs(fake_X_copy_linesearch) ** p)
                fake_output_linesearch[index] = (
                    sum_fake_xi_p_linesearch ** (1 / p) / self.num_points
                )

            loss_current_stepsize_plus = np.sum(
                np.exp((fake_output_linesearch - self.real_data_noisy_output) / self.lambda_val - 1)
            )
            loss_current_stepsize_minus = np.sum(
                np.exp((self.real_data_noisy_output - fake_output_linesearch) / self.lambda_val - 1)
            )
            loss_current_stepsize = loss_current_stepsize_plus + loss_current_stepsize_minus

            loss_damped_stepsize = (
                total_loss
                + alpha1 * t * loss_gradient[x_coord_descent]
                * (-loss_gradient[x_coord_descent] / loss_hessian_wrt_chosen_x_coord)
            )

            if loss_current_stepsize > loss_damped_stepsize:
                t *= alpha2
            else:
                break

        # Final update
        xi_update = -t * loss_gradient[x_coord_descent] / loss_hessian_wrt_chosen_x_coord
        self.fake_X[x_coord_descent] = xi + xi_update

        # Update fake outputs with the final updated fake_X
        for index, p in enumerate(self.sampled_queries):
            sum_fake_xi_p = np.sum(np.abs(self.fake_X) ** p)
            self.fake_output[index] = sum_fake_xi_p ** (1 / p) / self.num_points

        # Update error
        self.error = abs(self.fake_output - self.real_data_noisy_output)

        # New total loss
        total_loss_plus_after = np.sum(
            np.exp((self.fake_output - self.real_data_noisy_output) / self.lambda_val - 1)
        )
        total_loss_minus_after = np.sum(
            np.exp((self.real_data_noisy_output - self.fake_output) / self.lambda_val - 1)
        )
        total_loss_after = total_loss_plus_after + total_loss_minus_after

        total_loss_update = total_loss - total_loss_after
        return total_loss_after, total_loss_update

    def run_coordinate_descent(self, max_iterations: int = 1000, verbose: bool = True) -> dict:
        """
        Run the coordinate descent algorithm until convergence.

        For CONVEX LP norms: Stops when BOTH conditions are met:
          1. Satisfaction ratio reaches 0.5 + eta
          2. Total loss update < tau
        """
        if self.sampled_queries is None or self.query_weights is None:
            raise ValueError("Must set queries and weights first")

        self.compute_query_outputs()
        target_ratio = 0.5 + self.eta

        num_iterations = 0
        total_loss_update = float('inf')

        initial_weighted_satisfaction = self.compute_weighted_satisfaction_ratio()

        if verbose:
            print(f"Initial weighted satisfaction ratio: {initial_weighted_satisfaction:.4f}")
            print(f"Target weighted satisfaction ratio: {target_ratio:.4f}")
            print(f"Lambda: {self.lambda_val:.6f}")
            print(f"Lambda/2: {self.lambda_val / 2:.6f}")
            print(f"Tau: {self.tau}")
            print(f"Number of queries: {self.k}")
            print(f"Number of queries above lambda/2 error: {np.sum(self.error > self.lambda_val / 2)}")
            print("NOTE: For CONVEX LP norms - algorithm stops when BOTH conditions are met:")
            print("  1. Satisfaction ratio >= 0.5 + eta")
            print("  2. Total loss update < tau")

        while (num_iterations < max_iterations and
               (self.compute_weighted_satisfaction_ratio() < target_ratio or
                total_loss_update > self.tau)):

            total_loss, total_loss_update = self.coordinate_descent_step()
            num_iterations += 1

            if verbose and num_iterations % 10 == 0:
                current_satisfaction = self.compute_weighted_satisfaction_ratio()
                queries_above_threshold = np.sum(self.error > self.lambda_val / 2)
                print(f"Iteration {num_iterations}: "
                      f"Loss update = {total_loss_update:.6f}, "
                      f"Weighted satisfaction = {current_satisfaction:.4f}, "
                      f"Queries above lambda/2 = {queries_above_threshold}")

        final_weighted_satisfaction = self.compute_weighted_satisfaction_ratio()
        final_error_stats = {
            'mean_error': float(np.mean(self.error)),
            'max_error': float(np.max(self.error)),
            'queries_above_lambda_half': int(np.sum(self.error > self.lambda_val / 2)),
        }

        results = {
            'num_iterations': num_iterations,
            'converged': total_loss_update <= self.tau,
            'target_reached': final_weighted_satisfaction >= target_ratio,
            'both_conditions_met': (final_weighted_satisfaction >= target_ratio and
                                    total_loss_update <= self.tau),
            'initial_weighted_satisfaction': initial_weighted_satisfaction,
            'final_weighted_satisfaction': final_weighted_satisfaction,
            'target_ratio': target_ratio,
            'final_loss': total_loss,
            'final_loss_update': total_loss_update,
            'lambda_val': self.lambda_val,
            'tau': self.tau,
            'error_stats': final_error_stats,
            'fake_X': self.fake_X.copy(),
            'fake_X_original': self.fake_X_original.copy(),
            'real_X': self.real_X.copy(),
            'queries': self.sampled_queries.copy(),
            'weights': self.query_weights.copy(),
            'errors': self.error.copy(),
            'lap_noise': self.lap_noise.copy(),

            ### expose per-query outputs so your plotting script can use them
            'real_output': self.real_output.copy(),
            'real_data_noisy_output': self.real_data_noisy_output.copy(),
            'fake_output': self.fake_output.copy(),

            # useful debug info
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'data_precision': self.data_precision,
            'grid_step': self.grid_step,
            'seed': self.seed,
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


def run_lp_norms_optimization(queries: List[float],
                              weights: List[float],
                              epsilon: float = 15.0,
                              delta: float = 1e-1,
                              beta: float = 0.05,
                              eta: float = 0.001,
                              n: int = 100,
                              tau: float = 0.01,
                              upper_bound: float = DEFAULT_UPPER_BOUND,
                              lower_bound: float = DEFAULT_LOWER_BOUND,
                              data_precision: int = DEFAULT_DATA_PRECISION,
                              max_iterations: int = 1000,
                              verbose: bool = True,
                              seed: Optional[int] = None,
                              rng: Optional[np.random.Generator] = None) -> dict:
    """
    Convenience function to run the LP norms coordinate descent optimization.

    NOTE: Defaults are tied to the module-level DEFAULT_* constants,
    so you can change bounds/precision in one place.

    For reproducibility:
        - Pass a fixed `seed` (e.g., seed=12345), OR
        - Pass your own `rng=np.random.default_rng(12345)` if you want full control.
    """
    optimizer = LPNormsCoordinateDescent(
        epsilon=epsilon,
        delta=delta,
        beta=beta,
        eta=eta,
        n=n,
        tau=tau,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        data_precision=data_precision,
        seed=seed,
        rng=rng,
    )

    optimizer.set_queries_and_weights(queries, weights)
    optimizer.generate_data()
    results = optimizer.run_coordinate_descent(max_iterations=max_iterations, verbose=verbose)
    return results


if __name__ == "__main__":
    # Simple example usage
    queries = [1.5, 2.0, 2.5, 3.0, 4.0]
    weights = [0.2] * len(queries)

    results = run_lp_norms_optimization(
        queries=queries,
        weights=weights,
        epsilon=15.0,
        delta=1e-1,
        beta=0.05,
        eta=0.001,
        n=100,
        tau=0.01,
        max_iterations=1000,
        verbose=True,
        seed=12345,  # this makes the run reproducible
    )

    print(f"\nFinal Results:")
    print(f"Final weighted satisfaction: {results['final_weighted_satisfaction']:.4f}")
    print(f"Target was: {results['target_ratio']:.4f}")
    print(f"Both conditions met: {results['both_conditions_met']}")
    print(f"Lambda value: {results['lambda_val']:.6f}")
    print(f"Tau: {results['tau']}")
