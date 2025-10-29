import numpy as np
import math
from typing import List, Tuple, Optional

class SpikyNonconvexCoordinateDescent:
    """
    Coordinate descent algorithm for spiky nonconvex queries.

    This class implements a coordinate descent algorithm for optimizing synthetic data
    to match noisy query outputs while maintaining differential privacy guarantees.

    Query function: f(X) = (1/n) * sum_i (x_i^2 + a_i * sin(w_i*x_i))
    where a_i ∈ (0,1], x_i ∈ [lower_bound, upper_bound], and w_i are frequency parameters for each query.

    Supports two update modes (per outer iteration):
      - "newton": top-1 coordinate, repeated m times; Newton step with Armijo backtracking
      - "gd":     top-1 coordinate, repeated m times; gradient step with Armijo backtracking
    """

    def __init__(self,
                 epsilon: float,
                 delta: float,
                 beta: float,
                 eta: float,
                 n: int,
                 tau: float = 0.01,
                 upper_bound: float = math.pi,
                 lower_bound: float = -math.pi,
                 frequency: float = 5.0,
                 data_precision: int = 4,
                 seed: Optional[int] = None,
                 # --- controls for the inner updates ---
                 update_mode: str = "newton",         # "newton" or "gd"
                 m_inner: int = 20,                   # times to apply top-1 update per outer iteration
                 shortlist_size: int = 64,            # number of candidates to consider for Newton top-1
                 selection_rule: str = "newton_step", # "newton_step" or "grad_mag"
                 newton_damping: float = 1e-6):       # kept for reference (commented out below)
        """
        Initialize the coordinate descent algorithm with DP parameters.

        Args:
            epsilon, delta, beta, eta, n, tau: DP/boosting/stopping params
            upper_bound, lower_bound: data bounds for x_i
            frequency, data_precision: not used directly here beyond initialization
            seed: seed for local RNG (None -> true randomness)
            update_mode: "newton" or "gd"
            m_inner: number of top-1 coordinate updates per outer iteration
            shortlist_size: L candidates for Newton shortlist
            selection_rule:
                - "newton_step": pick coord with largest |g_i/H_i| among shortlist
                - "grad_mag":    pick coord with largest |g_i|
            newton_damping: (kept for reference) positivity floor for Hessian (commented out)
        """
        # Core params
        self.epsilon = epsilon
        self.delta = delta
        self.beta = beta
        self.eta = eta
        self.n = n
        self.tau = tau
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.frequency = frequency
        self.data_precision = data_precision

        # Local RNG (isolated from global np.random)
        self.rng = np.random.default_rng(seed)

        # Derived params
        self.num_points = n
        self.m = self.num_points * math.ceil(math.log2(((upper_bound - lower_bound)/10**(-data_precision)) + 1))
        self.rho = (self.upper_bound**2 + 1) / self.n

        # Data / state
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
        self.lambda_val = None
        self.amplitudes_matrix = None
        self.frequencies_vector = None

        # Update controls
        self.update_mode = update_mode
        self.m_inner = int(m_inner)
        self.shortlist_size = int(shortlist_size)
        self.selection_rule = selection_rule
        self.newton_damping = newton_damping  # NOTE: we will NOT use it (commented out below)

    # -------------------------------
    # Setup & data generation methods
    # -------------------------------
    def set_queries_and_amplitudes(self,
                                   amplitudes_matrix: np.ndarray,
                                   frequencies_vector: np.ndarray,
                                   weights: List[float]):
        """
        Set the amplitude matrix, frequency vector, and weights for spiky nonconvex queries.

        Args:
            amplitudes_matrix: (k, n) amplitudes a_ij for j-th coordinate in i-th query
            frequencies_vector: (k,) frequencies w_i for each query
            weights: list of k nonnegative weights summing to 1
        """
        amplitudes_matrix = np.asarray(amplitudes_matrix, dtype=float)
        frequencies_vector = np.asarray(frequencies_vector, dtype=float)
        k, n_amps = amplitudes_matrix.shape

        if len(weights) != k:
            raise ValueError("Number of amplitude vectors must match number of weights")
        if len(frequencies_vector) != k:
            raise ValueError("Number of frequency parameters must match number of queries")
        if not np.isclose(sum(weights), 1.0, atol=1e-6):
            raise ValueError("Weights must sum to 1")
        if n_amps != self.num_points:
            raise ValueError(f"Amplitude vectors length {n_amps} must equal num_points {self.num_points}")
        if np.any(amplitudes_matrix <= 0) or np.any(amplitudes_matrix > 1):
            raise ValueError("All amplitudes must be in (0, 1]")
        if np.any(frequencies_vector <= 0):
            raise ValueError("All frequencies must be positive")

        self.amplitudes_matrix = amplitudes_matrix.copy()
        self.frequencies_vector = frequencies_vector.copy()
        self.query_weights = np.array(weights, dtype=float)
        self.k = k

        # For compatibility, dummy query types not used in computation
        self.sampled_queries = np.array([2.0] * k)

        # lambda = ln(k/beta) * 2 * rho * sqrt(2k * ln(1/delta)) / epsilon
        self.lambda_val = (math.log(self.k / self.beta) * 2 * self.rho *
                           math.sqrt(2 * self.k * math.log(1 / self.delta))) / self.epsilon

        print(f"Calculated lambda: {self.lambda_val:.6f}")
        print(f"Set {k} spiky nonconvex queries with different amplitude vectors and frequencies")
        print(f"Frequencies: {frequencies_vector}")

    def generate_data(self,
                      real_data: Optional[np.ndarray] = None,
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
            # Continuous uniform on [lower_bound, upper_bound]
            self.real_X = self.rng.uniform(self.lower_bound, self.upper_bound, size=self.num_points)

        if initial_fake_data is not None:
            self.fake_X = initial_fake_data.copy()
        else:
            # Gaussian initialization
            self.fake_X = self.rng.normal(loc=0.0, scale=1.0, size=self.num_points)

        self.fake_X_original = self.fake_X.copy()

    # -------------------------------
    # Core computations
    # -------------------------------
    def compute_query_outputs(self):
        """Compute query outputs on real and fake data, including DP noise on the real outputs."""
        if self.amplitudes_matrix is None or self.frequencies_vector is None:
            raise ValueError("Must set amplitude matrix and frequency vector first")

        k = self.k

        self.real_output = np.zeros(k)
        self.real_data_noisy_output = np.zeros(k)
        self.lap_noise = np.zeros(k)
        self.fake_output = np.zeros(k)
        self.error = np.zeros(k)

        # Noise scale for DP
        noise_scale = self.rho * math.sqrt(2 * k * math.log(1 / self.delta)) / self.epsilon

        for index in range(k):
            amps = self.amplitudes_matrix[index]
            freq = self.frequencies_vector[index]

            # Real output
            sum_real = np.sum(self.real_X**2 + amps * np.sin(freq * self.real_X))
            self.real_output[index] = sum_real / self.n

            # Add Laplace noise (use local RNG)
            self.lap_noise[index] = self.rng.laplace(loc=0.0, scale=noise_scale)
            self.real_data_noisy_output[index] = self.real_output[index] + self.lap_noise[index]

            # Fake output
            sum_fake = np.sum(self.fake_X**2 + amps * np.sin(freq * self.fake_X))
            self.fake_output[index] = sum_fake / self.n

            # Error
            self.error[index] = abs(self.fake_output[index] - self.real_data_noisy_output[index])

    def compute_weighted_satisfaction_ratio(self) -> float:
        """
        Weighted ratio of queries satisfying |error| < lambda/2.
        """
        if self.query_weights is None:
            raise ValueError("Must set queries and weights first")
        satisfied_mask = self.error < self.lambda_val / 2
        return float(np.sum(self.query_weights[satisfied_mask]))

    # -------------------------------
    # Derivatives for a single coordinate
    # -------------------------------
    def _coord_grad(self, i: int) -> float:
        """
        dL/dx_i at current self.fake_X, where
          L = sum_j exp( (f_j(fake)-y_j)/lambda - 1 ) + exp( (y_j-f_j(fake))/lambda - 1 )
        """
        xi = self.fake_X[i]
        grad_sum = 0.0
        for j in range(self.k):
            a_i = self.amplitudes_matrix[j, i]
            w_i = self.frequencies_vector[j]

            # dq_j/dx_i = (1/n) * (2*xi + w_i * a_i * cos(w_i*xi))
            dq_dxi = (1.0 / self.n) * (2 * xi + w_i * a_i * np.cos(w_i * xi))

            # loss grads
            diff = self.fake_output[j] - self.real_data_noisy_output[j]
            gplus  =  (dq_dxi * np.exp( diff / self.lambda_val - 1) / self.lambda_val)
            gminus = (-dq_dxi * np.exp(-diff / self.lambda_val - 1) / self.lambda_val)
            grad_sum += gplus + gminus
        return float(grad_sum)

    def _coord_hessian(self, i: int) -> float:
        """
        d^2 L / dx_i^2 at current self.fake_X.
        This computes the exact 1D second derivative along coordinate i.
        """
        xi = self.fake_X[i]
        H_sum = 0.0
        for j in range(self.k):
            a_i = self.amplitudes_matrix[j, i]
            w_i = self.frequencies_vector[j]

            # dq_j/dx_i and d2q_j/dx_i^2
            dq_dxi   = (1.0 / self.n) * (2 * xi + w_i * a_i * np.cos(w_i * xi))
            d2q_dxi2 = (1.0 / self.n) * (2 - (w_i**2) * a_i * np.sin(w_i * xi))

            diff = self.fake_output[j] - self.real_data_noisy_output[j]
            epos = np.exp( diff / self.lambda_val - 1)
            eneg = np.exp(-diff / self.lambda_val - 1)

            # Second derivative contribution for each loss term
            # Using derivative of exp(u) where u = +/- (f - y)/lambda - 1
            H_pos = epos * (d2q_dxi2 / self.lambda_val + (dq_dxi**2) / (self.lambda_val**2))
            H_neg = eneg * (d2q_dxi2 / self.lambda_val + (dq_dxi**2) / (self.lambda_val**2))
            H_sum += H_pos + H_neg
        return float(H_sum)

    # -------------------------------
    # Loss & line search helpers
    # -------------------------------
    def _loss_and_fake_output_for_x(self, x_new: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute total loss and fake_output for proposed x_new.
        """
        k = self.k
        fake_output_new = np.zeros(k)
        for j in range(k):
            amps = self.amplitudes_matrix[j]
            freq = self.frequencies_vector[j]
            s = np.sum(x_new**2 + amps * np.sin(freq * x_new))
            fake_output_new[j] = s / self.n
        loss_plus  = np.sum(np.exp((fake_output_new - self.real_data_noisy_output) / self.lambda_val - 1))
        loss_minus = np.sum(np.exp((self.real_data_noisy_output - fake_output_new) / self.lambda_val - 1))
        return (float(loss_plus + loss_minus), fake_output_new)

    def _directional_step(self, idx: int, step: float, direction: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Apply x[idx] <- x[idx] + step * direction (clipped), return new (loss, fake_output, x_new).
        """
        x_new = self.fake_X.copy()
        x_new[idx] = np.clip(x_new[idx] + step * direction, self.lower_bound, self.upper_bound)
        loss_new, f_out_new = self._loss_and_fake_output_for_x(x_new)
        return loss_new, f_out_new, x_new

    # -------------------------------
    # One outer step (does m_inner top-1 updates)
    # -------------------------------
    def coordinate_descent_step(self) -> Tuple[float, float]:
        """
        Perform one outer step: within it, do m_inner sequential top-1 coordinate updates
        using either Newton ("newton") or gradient ("gd") updates with Armijo backtracking.

        Returns:
            (total_loss_after, total_loss_update)
        """
        # Current loss
        total_loss_plus = np.sum(np.exp((self.fake_output - self.real_data_noisy_output) / self.lambda_val - 1))
        total_loss_minus = np.sum(np.exp((self.real_data_noisy_output - self.fake_output) / self.lambda_val - 1))
        total_loss = float(total_loss_plus + total_loss_minus)
        initial_total_loss = total_loss

        # Line search parameters
        alpha = 1e-3   # Armijo
        beta  = 0.7    # shrink factor
        t_min = 1e-6

        for _ in range(self.m_inner):
            # Full gradient (needed for both modes)
            grad = np.zeros(self.num_points)
            for i in range(self.num_points):
                grad[i] = self._coord_grad(i)

            # Select index to update
            if self.update_mode == "newton":
                # shortlist by |grad|
                L = min(self.shortlist_size, self.num_points)
                idxs_top = np.argpartition(-np.abs(grad), L-1)[:L]

                if self.selection_rule == "newton_step":
                    # compute |g_i / H_i| and pick largest
                    scores = np.zeros_like(idxs_top, dtype=float)
                    for t, idx in enumerate(idxs_top):
                        g_i = grad[idx]
                        H_i = self._coord_hessian(idx)

                        # --- NO DAMPING (as requested) ---
                        # if H_i < self.newton_damping:
                        #     H_i = self.newton_damping

                        if H_i == 0.0:
                            scores[t] = 0.0
                        else:
                            scores[t] = abs(g_i / H_i)

                    idx = int(idxs_top[np.argmax(scores)])
                else:
                    # fallback: just largest |grad|
                    idx = int(np.argmax(np.abs(grad)))

                # Newton direction on that coordinate
                g = grad[idx]
                H = self._coord_hessian(idx)

                # --- NO DAMPING (as requested) ---
                # if H < self.newton_damping:
                #     H = self.newton_damping

                # If H is zero, skip (no meaningful Newton step)
                if H == 0.0:
                    # Try the next-best by pure gradient if needed
                    # (or simply break the inner loop)
                    continue

                direction = -g / H
                dir_deriv = g * direction  # should be <= 0 near descent; if >0, Armijo rejects

            else:
                # Pure gradient top-1
                idx = int(np.argmax(np.abs(grad)))
                g = grad[idx]
                if abs(g) < 1e-12:
                    break
                direction = -g
                dir_deriv = g * direction  # = -g^2 <= 0

            # Armijo backtracking on this single coordinate
            t = 1.0
            while True:
                loss_try, fake_output_try, x_try = self._directional_step(idx, t, direction)
                if loss_try <= total_loss + alpha * t * dir_deriv:
                    # accept
                    self.fake_X = x_try
                    self.fake_output = fake_output_try
                    self.error = np.abs(self.fake_output - self.real_data_noisy_output)
                    total_loss = loss_try
                    break
                t *= beta
                if t < t_min:
                    # take a tiny step anyway
                    loss_try, fake_output_try, x_try = self._directional_step(idx, t_min, direction)
                    self.fake_X = x_try
                    self.fake_output = fake_output_try
                    self.error = np.abs(self.fake_output - self.real_data_noisy_output)
                    total_loss = loss_try
                    break

        total_loss_after = total_loss
        total_loss_update = initial_total_loss - total_loss_after
        return total_loss_after, total_loss_update

    # -------------------------------
    # Outer loop
    # -------------------------------
    def run_coordinate_descent(self, max_iterations: int = 1000, verbose: bool = True) -> dict:
        """
        Run the coordinate descent algorithm until the satisfaction ratio target is met
        (0.5 + eta) or max_iterations is reached.

        For your current (convex-LP-style) reporting, we still print tau-based convergence info,
        but the stopping criterion uses only the satisfaction ratio target.
        """
        if self.amplitudes_matrix is None or self.frequencies_vector is None or self.query_weights is None:
            raise ValueError("Must set amplitude matrix, frequency vector, and weights first")

        # Initial outputs & stats
        self.compute_query_outputs()
        target_ratio = 0.5 + self.eta

        num_iterations = 0
        total_loss_update = float('inf')
        total_loss = float('inf')

        initial_weighted_satisfaction = self.compute_weighted_satisfaction_ratio()

        if verbose:
            print(f"Initial weighted satisfaction ratio: {initial_weighted_satisfaction:.4f}")
            print(f"Target weighted satisfaction ratio: {target_ratio:.4f}")
            print(f"Lambda: {self.lambda_val:.6f}")
            print(f"Lambda/2: {self.lambda_val/2:.6f}")
            print(f"Tau: {self.tau}")
            print(f"Number of queries: {self.k}")
            print(f"Number of queries above lambda/2 error: {np.sum(self.error > self.lambda_val/2)}")
            print("NOTE: Stopping when satisfaction ratio >= 0.5 + eta (nonconvex setting).")

        # Main loop: stop when target ratio achieved or max iters
        while (num_iterations < max_iterations and
               self.compute_weighted_satisfaction_ratio() < target_ratio):
            total_loss, total_loss_update = self.coordinate_descent_step()
            num_iterations += 1

            if verbose and num_iterations % 10 == 0:
                current_satisfaction = self.compute_weighted_satisfaction_ratio()
                queries_above_threshold = np.sum(self.error > self.lambda_val / 2)
                print(
                    f"Iteration {num_iterations}: "
                    f"Loss update = {total_loss_update:.6f}, "
                    f"Weighted satisfaction = {current_satisfaction:.4f}, "
                    f"Queries above lambda/2 = {queries_above_threshold}"
                )

        # Final stats
        final_weighted_satisfaction = self.compute_weighted_satisfaction_ratio()
        final_error_stats = {
            'mean_error': float(np.mean(self.error)),
            'max_error': float(np.max(self.error)),
            'queries_above_lambda_half': int(np.sum(self.error > self.lambda_val / 2))
        }

        results = {
            'num_iterations': num_iterations,
            'iterations': num_iterations,  # for compatibility with external callers
            'converged': (total_loss_update <= self.tau),  # retained for reporting
            'target_reached': (final_weighted_satisfaction >= target_ratio),
            'both_conditions_met': (final_weighted_satisfaction >= target_ratio and total_loss_update <= self.tau),
            'initial_weighted_satisfaction': float(initial_weighted_satisfaction),
            'final_weighted_satisfaction': float(final_weighted_satisfaction),
            'target_ratio': float(target_ratio),
            'final_loss': float(total_loss),
            'final_loss_update': float(total_loss_update),
            'lambda_val': float(self.lambda_val),
            'tau': float(self.tau),
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


# -------------------------------
# Legacy convenience wrapper
# -------------------------------
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
    Legacy convenience function to run the optimization.
    Note: This sets trivial amplitudes/frequencies just to keep the interface.
    """
    optimizer = SpikyNonconvexCoordinateDescent(
        epsilon=epsilon,
        delta=delta,
        beta=beta,
        eta=eta,
        n=n,
        tau=tau,
        upper_bound=upper_bound,
        lower_bound=lower_bound
    )

    # Dummy amplitudes/frequencies for compatibility
    amplitudes_matrix = np.ones((len(queries), n), dtype=float)
    frequencies_vector = np.ones(len(queries), dtype=float) * 5.0
    optimizer.set_queries_and_amplitudes(amplitudes_matrix, frequencies_vector, weights)

    optimizer.generate_data()
    results = optimizer.run_coordinate_descent(max_iterations=max_iterations, verbose=verbose)
    return results


if __name__ == "__main__":
    print("SpikyNonconvexCoordinateDescent class loaded successfully.")
    print("Use your test harness to set amplitudes/frequencies and run experiments.")
