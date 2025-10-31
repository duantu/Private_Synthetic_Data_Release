# # import numpy as np
# # import math
# # from typing import List, Tuple, Optional

# # class SpikyNonconvexCoordinateDescent:
# #     """
# #     Coordinate descent algorithm for spiky nonconvex queries.

# #     This class implements a coordinate descent algorithm for optimizing synthetic data
# #     to match noisy query outputs while maintaining differential privacy guarantees.

# #     Query function: f(X) = (1/n) * sum_i (x_i^2 + a_i * sin(w_i*x_i))
# #     where a_i ∈ (0,1], x_i ∈ [lower_bound, upper_bound], and w_i are frequency parameters for each query.

# #     Supports two update modes (per outer iteration):
# #       - "newton": top-1 coordinate, repeated m times; Newton step with Armijo backtracking
# #       - "gd":     top-1 coordinate, repeated m times; gradient step with Armijo backtracking
# #     """

# #     def __init__(self,
# #                  epsilon: float,
# #                  delta: float,
# #                  beta: float,
# #                  eta: float,
# #                  n: int,
# #                  tau: float = 0.01,
# #                  upper_bound: float = math.pi,
# #                  lower_bound: float = -math.pi,
# #                  frequency: float = 5.0,
# #                  data_precision: int = 4,
# #                  seed: Optional[int] = None,
# #                  # --- controls for the inner updates ---
# #                  update_mode: str = "newton",         # "newton" or "gd"
# #                  m_inner: int = 20,                   # times to apply top-1 update per outer iteration
# #                  shortlist_size: int = 64,            # number of candidates to consider for Newton top-1
# #                  selection_rule: str = "newton_step", # "newton_step" or "grad_mag"
# #                  newton_damping: float = 1e-6):       # kept for reference (commented out below)
# #         """
# #         Initialize the coordinate descent algorithm with DP parameters.

# #         Args:
# #             epsilon, delta, beta, eta, n, tau: DP/boosting/stopping params
# #             upper_bound, lower_bound: data bounds for x_i
# #             frequency, data_precision: not used directly here beyond initialization
# #             seed: seed for local RNG (None -> true randomness)
# #             update_mode: "newton" or "gd"
# #             m_inner: number of top-1 coordinate updates per outer iteration
# #             shortlist_size: L candidates for Newton shortlist
# #             selection_rule:
# #                 - "newton_step": pick coord with largest |g_i/H_i| among shortlist
# #                 - "grad_mag":    pick coord with largest |g_i|
# #             newton_damping: (kept for reference) positivity floor for Hessian (commented out)
# #         """
# #         # Core params
# #         self.epsilon = epsilon
# #         self.delta = delta
# #         self.beta = beta
# #         self.eta = eta
# #         self.n = n
# #         self.tau = tau
# #         self.upper_bound = upper_bound
# #         self.lower_bound = lower_bound
# #         self.frequency = frequency
# #         self.data_precision = data_precision

# #         # Local RNG (isolated from global np.random)
# #         self.rng = np.random.default_rng(seed)

# #         # Derived params
# #         self.num_points = n
# #         self.m = self.num_points * math.ceil(math.log2(((upper_bound - lower_bound)/10**(-data_precision)) + 1))
# #         self.rho = (self.upper_bound**2 + 1) / self.n

# #         # Data / state
# #         self.real_X = None
# #         self.fake_X = None
# #         self.fake_X_original = None
# #         self.sampled_queries = None
# #         self.query_weights = None
# #         self.real_output = None
# #         self.real_data_noisy_output = None
# #         self.lap_noise = None
# #         self.fake_output = None
# #         self.error = None
# #         self.lambda_val = None
# #         self.amplitudes_matrix = None
# #         self.frequencies_vector = None

# #         # Update controls
# #         self.update_mode = update_mode
# #         self.m_inner = int(m_inner)
# #         self.shortlist_size = int(shortlist_size)
# #         self.selection_rule = selection_rule
# #         self.newton_damping = newton_damping  # NOTE: we will NOT use it (commented out below)

# #     # -------------------------------
# #     # Setup & data generation methods
# #     # -------------------------------
# #     def set_queries_and_amplitudes(self,
# #                                    amplitudes_matrix: np.ndarray,
# #                                    frequencies_vector: np.ndarray,
# #                                    weights: List[float]):
# #         """
# #         Set the amplitude matrix, frequency vector, and weights for spiky nonconvex queries.

# #         Args:
# #             amplitudes_matrix: (k, n) amplitudes a_ij for j-th coordinate in i-th query
# #             frequencies_vector: (k,) frequencies w_i for each query
# #             weights: list of k nonnegative weights summing to 1
# #         """
# #         amplitudes_matrix = np.asarray(amplitudes_matrix, dtype=float)
# #         frequencies_vector = np.asarray(frequencies_vector, dtype=float)
# #         k, n_amps = amplitudes_matrix.shape

# #         if len(weights) != k:
# #             raise ValueError("Number of amplitude vectors must match number of weights")
# #         if len(frequencies_vector) != k:
# #             raise ValueError("Number of frequency parameters must match number of queries")
# #         if not np.isclose(sum(weights), 1.0, atol=1e-6):
# #             raise ValueError("Weights must sum to 1")
# #         if n_amps != self.num_points:
# #             raise ValueError(f"Amplitude vectors length {n_amps} must equal num_points {self.num_points}")
# #         if np.any(amplitudes_matrix <= 0) or np.any(amplitudes_matrix > 1):
# #             raise ValueError("All amplitudes must be in (0, 1]")
# #         if np.any(frequencies_vector <= 0):
# #             raise ValueError("All frequencies must be positive")

# #         self.amplitudes_matrix = amplitudes_matrix.copy()
# #         self.frequencies_vector = frequencies_vector.copy()
# #         self.query_weights = np.array(weights, dtype=float)
# #         self.k = k

# #         # For compatibility, dummy query types not used in computation
# #         self.sampled_queries = np.array([2.0] * k)

# #         # lambda = ln(k/beta) * 2 * rho * sqrt(2k * ln(1/delta)) / epsilon
# #         self.lambda_val = (math.log(self.k / self.beta) * 2 * self.rho *
# #                            math.sqrt(2 * self.k * math.log(1 / self.delta))) / self.epsilon

# #         print(f"Calculated lambda: {self.lambda_val:.6f}")
# #         print(f"Set {k} spiky nonconvex queries with different amplitude vectors and frequencies")
# #         # print(f"Frequencies: {frequencies_vector}")

# #     def generate_data(self,
# #                       real_data: Optional[np.ndarray] = None,
# #                       initial_fake_data: Optional[np.ndarray] = None):
# #         """
# #         Generate or set the real and fake data.

# #         Args:
# #             real_data: Real data array (if None, generates random data)
# #             initial_fake_data: Initial fake data array (if None, generates random data)
# #         """
# #         if real_data is not None:
# #             self.real_X = real_data.copy()
# #         else:
# #             # Continuous uniform on [lower_bound, upper_bound]
# #             self.real_X = self.rng.uniform(self.lower_bound, self.upper_bound, size=self.num_points)

# #         if initial_fake_data is not None:
# #             self.fake_X = initial_fake_data.copy()
# #         else:
# #             # Gaussian initialization
# #             self.fake_X = self.rng.normal(loc=0.0, scale=1.0, size=self.num_points)

# #         self.fake_X_original = self.fake_X.copy()

# #     # -------------------------------
# #     # Core computations
# #     # -------------------------------
# #     def compute_query_outputs(self):
# #         """Compute query outputs on real and fake data, including DP noise on the real outputs."""
# #         if self.amplitudes_matrix is None or self.frequencies_vector is None:
# #             raise ValueError("Must set amplitude matrix and frequency vector first")

# #         k = self.k

# #         self.real_output = np.zeros(k)
# #         self.real_data_noisy_output = np.zeros(k)
# #         self.lap_noise = np.zeros(k)
# #         self.fake_output = np.zeros(k)
# #         self.error = np.zeros(k)

# #         # Noise scale for DP
# #         noise_scale = self.rho * math.sqrt(2 * k * math.log(1 / self.delta)) / self.epsilon

# #         for index in range(k):
# #             amps = self.amplitudes_matrix[index]
# #             freq = self.frequencies_vector[index]

# #             # Real output
# #             sum_real = np.sum(self.real_X**2 + amps * np.sin(freq * self.real_X))
# #             self.real_output[index] = sum_real / self.n

# #             # Add Laplace noise (use local RNG)
# #             self.lap_noise[index] = self.rng.laplace(loc=0.0, scale=noise_scale)
# #             self.real_data_noisy_output[index] = self.real_output[index] + self.lap_noise[index]

# #             # Fake output
# #             sum_fake = np.sum(self.fake_X**2 + amps * np.sin(freq * self.fake_X))
# #             self.fake_output[index] = sum_fake / self.n

# #             # Error
# #             self.error[index] = abs(self.fake_output[index] - self.real_data_noisy_output[index])

# #     def compute_weighted_satisfaction_ratio(self) -> float:
# #         """
# #         Weighted ratio of queries satisfying |error| < lambda/2.
# #         """
# #         if self.query_weights is None:
# #             raise ValueError("Must set queries and weights first")
# #         satisfied_mask = self.error < self.lambda_val / 2
# #         return float(np.sum(self.query_weights[satisfied_mask]))

# #     # -------------------------------
# #     # Derivatives for a single coordinate
# #     # -------------------------------
# #     def _coord_grad(self, i: int) -> float:
# #         """
# #         dL/dx_i at current self.fake_X, where
# #           L = sum_j exp( (f_j(fake)-y_j)/lambda - 1 ) + exp( (y_j-f_j(fake))/lambda - 1 )
# #         """
# #         xi = self.fake_X[i]
# #         grad_sum = 0.0
# #         for j in range(self.k):
# #             a_i = self.amplitudes_matrix[j, i]
# #             w_i = self.frequencies_vector[j]

# #             # dq_j/dx_i = (1/n) * (2*xi + w_i * a_i * cos(w_i*xi))
# #             dq_dxi = (1.0 / self.n) * (2 * xi + w_i * a_i * np.cos(w_i * xi))

# #             # loss grads
# #             diff = self.fake_output[j] - self.real_data_noisy_output[j]
# #             gplus  =  (dq_dxi * np.exp( diff / self.lambda_val - 1) / self.lambda_val)
# #             gminus = (-dq_dxi * np.exp(-diff / self.lambda_val - 1) / self.lambda_val)
# #             grad_sum += gplus + gminus
# #         return float(grad_sum)

# #     def _coord_hessian(self, i: int) -> float:
# #         """
# #         d^2 L / dx_i^2 at current self.fake_X.
# #         This computes the exact 1D second derivative along coordinate i.
# #         """
# #         xi = self.fake_X[i]
# #         H_sum = 0.0
# #         for j in range(self.k):
# #             a_i = self.amplitudes_matrix[j, i]
# #             w_i = self.frequencies_vector[j]

# #             # dq_j/dx_i and d2q_j/dx_i^2
# #             dq_dxi   = (1.0 / self.n) * (2 * xi + w_i * a_i * np.cos(w_i * xi))
# #             d2q_dxi2 = (1.0 / self.n) * (2 - (w_i**2) * a_i * np.sin(w_i * xi))

# #             diff = self.fake_output[j] - self.real_data_noisy_output[j]
# #             epos = np.exp( diff / self.lambda_val - 1)
# #             eneg = np.exp(-diff / self.lambda_val - 1)

# #             # Second derivative contribution for each loss term
# #             # Using derivative of exp(u) where u = +/- (f - y)/lambda - 1
# #             H_pos = epos * (d2q_dxi2 / self.lambda_val + (dq_dxi**2) / (self.lambda_val**2))
# #             H_neg = eneg * (d2q_dxi2 / self.lambda_val + (dq_dxi**2) / (self.lambda_val**2))
# #             H_sum += H_pos + H_neg
# #         return float(H_sum)

# #     # -------------------------------
# #     # Loss & line search helpers
# #     # -------------------------------
# #     def _loss_and_fake_output_for_x(self, x_new: np.ndarray) -> Tuple[float, np.ndarray]:
# #         """
# #         Compute total loss and fake_output for proposed x_new.
# #         """
# #         k = self.k
# #         fake_output_new = np.zeros(k)
# #         for j in range(k):
# #             amps = self.amplitudes_matrix[j]
# #             freq = self.frequencies_vector[j]
# #             s = np.sum(x_new**2 + amps * np.sin(freq * x_new))
# #             fake_output_new[j] = s / self.n
# #         loss_plus  = np.sum(np.exp((fake_output_new - self.real_data_noisy_output) / self.lambda_val - 1))
# #         loss_minus = np.sum(np.exp((self.real_data_noisy_output - fake_output_new) / self.lambda_val - 1))
# #         return (float(loss_plus + loss_minus), fake_output_new)

# #     def _directional_step(self, idx: int, step: float, direction: float) -> Tuple[float, np.ndarray, np.ndarray]:
# #         """
# #         Apply x[idx] <- x[idx] + step * direction (clipped), return new (loss, fake_output, x_new).
# #         """
# #         x_new = self.fake_X.copy()
# #         x_new[idx] = np.clip(x_new[idx] + step * direction, self.lower_bound, self.upper_bound)
# #         loss_new, f_out_new = self._loss_and_fake_output_for_x(x_new)
# #         return loss_new, f_out_new, x_new

# #     # -------------------------------
# #     # One outer step (does m_inner top-1 updates)
# #     # -------------------------------
# #     def coordinate_descent_step(self) -> Tuple[float, float]:
# #         """
# #         Perform one outer step: within it, do m_inner sequential top-1 coordinate updates
# #         using either Newton ("newton") or gradient ("gd") updates with Armijo backtracking.

# #         Returns:
# #             (total_loss_after, total_loss_update)
# #         """
# #         # Current loss
# #         total_loss_plus = np.sum(np.exp((self.fake_output - self.real_data_noisy_output) / self.lambda_val - 1))
# #         total_loss_minus = np.sum(np.exp((self.real_data_noisy_output - self.fake_output) / self.lambda_val - 1))
# #         total_loss = float(total_loss_plus + total_loss_minus)
# #         initial_total_loss = total_loss

# #         # Line search parameters
# #         alpha = 1e-3   # Armijo
# #         beta  = 0.7    # shrink factor
# #         t_min = 1e-6

# #         for _ in range(self.m_inner):
# #             # Full gradient (needed for both modes)
# #             grad = np.zeros(self.num_points)
# #             for i in range(self.num_points):
# #                 grad[i] = self._coord_grad(i)

# #             # Select index to update
# #             if self.update_mode == "newton":
# #                 # shortlist by |grad|
# #                 L = min(self.shortlist_size, self.num_points)
# #                 idxs_top = np.argpartition(-np.abs(grad), L-1)[:L]

# #                 if self.selection_rule == "newton_step":
# #                     # compute |g_i / H_i| and pick largest
# #                     scores = np.zeros_like(idxs_top, dtype=float)
# #                     for t, idx in enumerate(idxs_top):
# #                         g_i = grad[idx]
# #                         H_i = self._coord_hessian(idx)

# #                         # --- NO DAMPING (as requested) ---
# #                         # if H_i < self.newton_damping:
# #                         #     H_i = self.newton_damping

# #                         if H_i == 0.0:
# #                             scores[t] = 0.0
# #                         else:
# #                             scores[t] = abs(g_i / H_i)

# #                     idx = int(idxs_top[np.argmax(scores)])
# #                 else:
# #                     # fallback: just largest |grad|
# #                     idx = int(np.argmax(np.abs(grad)))

# #                 # Newton direction on that coordinate
# #                 g = grad[idx]
# #                 H = self._coord_hessian(idx)

# #                 # --- NO DAMPING (as requested) ---
# #                 # if H < self.newton_damping:
# #                 #     H = self.newton_damping

# #                 # If H is zero, skip (no meaningful Newton step)
# #                 if H == 0.0:
# #                     # Try the next-best by pure gradient if needed
# #                     # (or simply break the inner loop)
# #                     continue

# #                 direction = -g / H
# #                 dir_deriv = g * direction  # should be <= 0 near descent; if >0, Armijo rejects

# #             else:
# #                 # Pure gradient top-1
# #                 idx = int(np.argmax(np.abs(grad)))
# #                 g = grad[idx]
# #                 if abs(g) < 1e-12:
# #                     break
# #                 direction = -g
# #                 dir_deriv = g * direction  # = -g^2 <= 0

# #             # Armijo backtracking on this single coordinate
# #             t = 1.0
# #             while True:
# #                 loss_try, fake_output_try, x_try = self._directional_step(idx, t, direction)
# #                 if loss_try <= total_loss + alpha * t * dir_deriv:
# #                     # accept
# #                     self.fake_X = x_try
# #                     self.fake_output = fake_output_try
# #                     self.error = np.abs(self.fake_output - self.real_data_noisy_output)
# #                     total_loss = loss_try
# #                     break
# #                 t *= beta
# #                 if t < t_min:
# #                     # take a tiny step anyway
# #                     loss_try, fake_output_try, x_try = self._directional_step(idx, t_min, direction)
# #                     self.fake_X = x_try
# #                     self.fake_output = fake_output_try
# #                     self.error = np.abs(self.fake_output - self.real_data_noisy_output)
# #                     total_loss = loss_try
# #                     break

# #         total_loss_after = total_loss
# #         total_loss_update = initial_total_loss - total_loss_after
# #         return total_loss_after, total_loss_update

# #     # -------------------------------
# #     # Outer loop
# #     # -------------------------------
# #     def run_coordinate_descent(self, max_iterations: int = 1000, verbose: bool = True) -> dict:
# #         """
# #         Run the coordinate descent algorithm until the satisfaction ratio target is met
# #         (0.5 + eta) or max_iterations is reached.

# #         For your current (convex-LP-style) reporting, we still print tau-based convergence info,
# #         but the stopping criterion uses only the satisfaction ratio target.
# #         """
# #         if self.amplitudes_matrix is None or self.frequencies_vector is None or self.query_weights is None:
# #             raise ValueError("Must set amplitude matrix, frequency vector, and weights first")

# #         # Initial outputs & stats
# #         self.compute_query_outputs()
# #         target_ratio = 0.5 + self.eta

# #         num_iterations = 0
# #         total_loss_update = float('inf')
# #         total_loss = float('inf')

# #         initial_weighted_satisfaction = self.compute_weighted_satisfaction_ratio()

# #         if verbose:
# #             print(f"Initial weighted satisfaction ratio: {initial_weighted_satisfaction:.4f}")
# #             print(f"Target weighted satisfaction ratio: {target_ratio:.4f}")
# #             print(f"Lambda: {self.lambda_val:.6f}")
# #             print(f"Lambda/2: {self.lambda_val/2:.6f}")
# #             print(f"Tau: {self.tau}")
# #             print(f"Number of queries: {self.k}")
# #             print(f"Number of queries above lambda/2 error: {np.sum(self.error > self.lambda_val/2)}")
# #             print("NOTE: Stopping when satisfaction ratio >= 0.5 + eta (nonconvex setting).")

# #         # Main loop: stop when target ratio achieved or max iters
# #         while (num_iterations < max_iterations and
# #                self.compute_weighted_satisfaction_ratio() < target_ratio):
# #             total_loss, total_loss_update = self.coordinate_descent_step()
# #             num_iterations += 1

# #             if verbose and num_iterations % 10 == 0:
# #                 current_satisfaction = self.compute_weighted_satisfaction_ratio()
# #                 queries_above_threshold = np.sum(self.error > self.lambda_val / 2)
# #                 print(
# #                     f"Iteration {num_iterations}: "
# #                     f"Loss update = {total_loss_update:.6f}, "
# #                     f"Weighted satisfaction = {current_satisfaction:.4f}, "
# #                     f"Queries above lambda/2 = {queries_above_threshold}"
# #                 )

# #         # Final stats
# #         final_weighted_satisfaction = self.compute_weighted_satisfaction_ratio()
# #         final_error_stats = {
# #             'mean_error': float(np.mean(self.error)),
# #             'max_error': float(np.max(self.error)),
# #             'queries_above_lambda_half': int(np.sum(self.error > self.lambda_val / 2))
# #         }

# #         results = {
# #             'num_iterations': num_iterations,
# #             'iterations': num_iterations,  # for compatibility with external callers
# #             'converged': (total_loss_update <= self.tau),  # retained for reporting
# #             'target_reached': (final_weighted_satisfaction >= target_ratio),
# #             'both_conditions_met': (final_weighted_satisfaction >= target_ratio and total_loss_update <= self.tau),
# #             'initial_weighted_satisfaction': float(initial_weighted_satisfaction),
# #             'final_weighted_satisfaction': float(final_weighted_satisfaction),
# #             'target_ratio': float(target_ratio),
# #             'final_loss': float(total_loss),
# #             'final_loss_update': float(total_loss_update),
# #             'lambda_val': float(self.lambda_val),
# #             'tau': float(self.tau),
# #             'error_stats': final_error_stats,
# #             'fake_X': self.fake_X.copy(),
# #             'real_X': self.real_X.copy(),
# #             'queries': self.sampled_queries.copy(),
# #             'weights': self.query_weights.copy(),
# #             'errors': self.error.copy(),
# #             'lap_noise': self.lap_noise.copy()
# #         }

# #         if verbose:
# #             print(f"\nAlgorithm completed:")
# #             print(f"  Iterations: {num_iterations}")
# #             print(f"  Converged (loss update <= tau): {results['converged']}")
# #             print(f"  Target reached (weighted satisfaction >= 0.5 + eta): {results['target_reached']}")
# #             print(f"  Both conditions met: {results['both_conditions_met']}")
# #             print(f"  Final weighted satisfaction: {final_weighted_satisfaction:.4f}")
# #             print(f"  Final loss update: {total_loss_update:.6f}")
# #             print(f"  Queries above lambda/2: {final_error_stats['queries_above_lambda_half']}")

# #         return results


# # # -------------------------------
# # # Legacy convenience wrapper
# # # -------------------------------
# # def run_lp_norms_optimization(queries: List[float],
# #                               weights: List[float],
# #                               epsilon: float = 15.0,
# #                               delta: float = 1e-1,
# #                               beta: float = 0.05,
# #                               eta: float = 0.001,
# #                               n: int = 100,
# #                               tau: float = 0.01,
# #                               upper_bound: float = 1.0,
# #                               lower_bound: float = -1.0,
# #                               max_iterations: int = 1000,
# #                               verbose: bool = True) -> dict:
# #     """
# #     Legacy convenience function to run the optimization.
# #     Note: This sets trivial amplitudes/frequencies just to keep the interface.
# #     """
# #     optimizer = SpikyNonconvexCoordinateDescent(
# #         epsilon=epsilon,
# #         delta=delta,
# #         beta=beta,
# #         eta=eta,
# #         n=n,
# #         tau=tau,
# #         upper_bound=upper_bound,
# #         lower_bound=lower_bound
# #     )

# #     # Dummy amplitudes/frequencies for compatibility
# #     amplitudes_matrix = np.ones((len(queries), n), dtype=float)
# #     frequencies_vector = np.ones(len(queries), dtype=float) * 5.0
# #     optimizer.set_queries_and_amplitudes(amplitudes_matrix, frequencies_vector, weights)

# #     optimizer.generate_data()
# #     results = optimizer.run_coordinate_descent(max_iterations=max_iterations, verbose=verbose)
# #     return results


# # if __name__ == "__main__":
# #     print("SpikyNonconvexCoordinateDescent class loaded successfully.")
# #     print("Use your test harness to set amplitudes/frequencies and run experiments.")

# import numpy as np
# import math
# from typing import List, Tuple, Optional

# class SpikyNonconvexCoordinateDescent:
#     """
#     Coordinate descent algorithm for spiky nonconvex queries.

#     Query function (per query j):
#       q_j(X) = (1/n) * sum_i (x_i^2 + a_{j,i} * sin(w_j * x_i))

#     Loss:
#       L = sum_j [ exp((f_j(fake)-y_j)/λ - 1) + exp((y_j-f_j(fake))/λ - 1) ]

#     Update scheme (per outer step):
#       - compute gradient fresh each inner update
#       - rebuild shortlist via argpartition by |grad|
#       - score shortlist by |g/H| (Newton step magnitude)
#       - take up to m_inner coordinates (sequentially) with Newton + Armijo + step cap
#       - skip H <= 0; do NOT enforce descent by tweaking H
#     """

#     def __init__(self,
#                  epsilon: float,
#                  delta: float,
#                  beta: float,
#                  eta: float,
#                  n: int,
#                  tau: float = 0.01,
#                  upper_bound: float = math.pi,
#                  lower_bound: float = -math.pi,
#                  frequency: float = 5.0,
#                  data_precision: int = 4,
#                  seed: Optional[int] = None,
#                  # update controls
#                  update_mode: str = "newton",         # "newton" or "gd"
#                  m_inner: int = 8,                    # number of coords to update per outer step
#                  shortlist_size: int = 32,            # shortlist size L for |grad| preselection
#                  selection_rule: str = "newton_step", # "newton_step" uses |g/H| ranking
#                  newton_damping: float = 0.0):
#         # Core params
#         self.epsilon = epsilon
#         self.delta = delta
#         self.beta = beta
#         self.eta = eta
#         self.n = n
#         self.tau = tau
#         self.upper_bound = upper_bound
#         self.lower_bound = lower_bound
#         self.frequency = frequency
#         self.data_precision = data_precision

#         # Local RNG (isolated from global np.random)
#         self.rng = np.random.default_rng(seed)

#         # Derived params
#         self.num_points = n
#         self.m = self.num_points * math.ceil(math.log2(((upper_bound - lower_bound)/10**(-data_precision)) + 1))
#         self.rho = (self.upper_bound**2 + 1) / self.n

#         # Data / state
#         self.real_X = None
#         self.fake_X = None
#         self.fake_X_original = None
#         self.sampled_queries = None
#         self.query_weights = None
#         self.real_output = None
#         self.real_data_noisy_output = None
#         self.lap_noise = None
#         self.fake_output = None
#         self.error = None
#         self.lambda_val = None
#         self.amplitudes_matrix = None
#         self.frequencies_vector = None
#         self.k = 0

#         # Update controls
#         self.update_mode = update_mode
#         self.m_inner = int(m_inner)
#         self.shortlist_size = int(shortlist_size)
#         self.selection_rule = selection_rule
#         self.newton_damping = float(newton_damping)  # (unused; kept for reference)

#     # -------------------------------
#     # Setup & data generation methods
#     # -------------------------------
#     def set_queries_and_amplitudes(self,
#                                    amplitudes_matrix: np.ndarray,
#                                    frequencies_vector: np.ndarray,
#                                    weights: List[float]):
#         """
#         Set the amplitude matrix, frequency vector, and weights.
#           amplitudes_matrix: shape (k, n), entries in (0,1]
#           frequencies_vector: shape (k,), entries > 0
#           weights: list of k nonnegative weights summing to 1
#         """
#         amplitudes_matrix = np.asarray(amplitudes_matrix, dtype=float)
#         frequencies_vector = np.asarray(frequencies_vector, dtype=float)
#         k, n_amps = amplitudes_matrix.shape

#         if len(weights) != k:
#             raise ValueError("Number of amplitude vectors must match number of weights")
#         if len(frequencies_vector) != k:
#             raise ValueError("Number of frequency parameters must match number of queries")
#         if not np.isclose(sum(weights), 1.0, atol=1e-6):
#             raise ValueError("Weights must sum to 1")
#         if n_amps != self.num_points:
#             raise ValueError(f"Amplitude vectors length {n_amps} must equal num_points {self.num_points}")
#         if np.any(amplitudes_matrix <= 0) or np.any(amplitudes_matrix > 1):
#             raise ValueError("All amplitudes must be in (0, 1]")
#         if np.any(frequencies_vector <= 0):
#             raise ValueError("All frequencies must be positive")

#         self.amplitudes_matrix = amplitudes_matrix.copy()
#         self.frequencies_vector = frequencies_vector.copy()
#         self.query_weights = np.array(weights, dtype=float)
#         self.k = k

#         # For compatibility, dummy query types not used in computation
#         self.sampled_queries = np.array([2.0] * k)

#         # lambda = ln(k/beta) * 2 * rho * sqrt(2k * ln(1/delta)) / epsilon
#         self.lambda_val = (math.log(self.k / self.beta) * 2 * self.rho *
#                            math.sqrt(2 * self.k * math.log(1 / self.delta))) / self.epsilon

#         print(f"Calculated lambda: {self.lambda_val:.6f}")
#         print(f"Set {k} spiky nonconvex queries with different amplitude vectors and frequencies")

#     def generate_data(self,
#                       real_data: Optional[np.ndarray] = None,
#                       initial_fake_data: Optional[np.ndarray] = None):
#         """
#         Generate or set the real and fake data.
#         (Fast continuous sampling + local RNG)
#         """
#         if real_data is not None:
#             self.real_X = real_data.copy()
#         else:
#             # Continuous uniform on [lower_bound, upper_bound]
#             self.real_X = self.rng.uniform(self.lower_bound, self.upper_bound, size=self.num_points)

#         if initial_fake_data is not None:
#             self.fake_X = initial_fake_data.copy()
#         else:
#             # Gaussian initialization
#             self.fake_X = self.rng.normal(loc=0.0, scale=1.0, size=self.num_points)

#         self.fake_X_original = self.fake_X.copy()

#     # -------------------------------
#     # Core computations
#     # -------------------------------
#     def compute_query_outputs(self):
#         """Compute query outputs on real and fake data, including DP noise on the real outputs."""
#         if self.amplitudes_matrix is None or self.frequencies_vector is None:
#             raise ValueError("Must set amplitude matrix and frequency vector first")

#         k = self.k
#         self.real_output = np.zeros(k)
#         self.real_data_noisy_output = np.zeros(k)
#         self.lap_noise = np.zeros(k)
#         self.fake_output = np.zeros(k)
#         self.error = np.zeros(k)

#         # Noise scale for DP (compute once)
#         noise_scale = self.rho * math.sqrt(2 * k * math.log(1 / self.delta)) / self.epsilon

#         for index in range(k):
#             amps = self.amplitudes_matrix[index]
#             freq = self.frequencies_vector[index]

#             # Real output
#             sum_real = np.sum(self.real_X**2 + amps * np.sin(freq * self.real_X))
#             self.real_output[index] = sum_real / self.n

#             # Add Laplace noise (use local RNG)
#             self.lap_noise[index] = self.rng.laplace(loc=0.0, scale=noise_scale)
#             self.real_data_noisy_output[index] = self.real_output[index] + self.lap_noise[index]

#             # Fake output
#             sum_fake = np.sum(self.fake_X**2 + amps * np.sin(freq * self.fake_X))
#             self.fake_output[index] = sum_fake / self.n

#             # Error
#             self.error[index] = abs(self.fake_output[index] - self.real_data_noisy_output[index])

#     def compute_weighted_satisfaction_ratio(self) -> float:
#         """Weighted ratio of queries satisfying |error| < lambda/2."""
#         if self.query_weights is None:
#             raise ValueError("Must set queries and weights first")
#         satisfied_mask = self.error < self.lambda_val / 2
#         return float(np.sum(self.query_weights[satisfied_mask]))

#     # -------------------------------
#     # Derivatives for a single coordinate
#     # -------------------------------
#     def _coord_grad(self, i: int) -> float:
#         """
#         dL/dx_i at current self.fake_X.
#         """
#         xi = self.fake_X[i]
#         grad_sum = 0.0
#         for j in range(self.k):
#             a_i = self.amplitudes_matrix[j, i]
#             w_i = self.frequencies_vector[j]

#             # dq_j/dx_i = (1/n) * (2*xi + w_i * a_i * cos(w_i*xi))
#             dq_dxi = (1.0 / self.n) * (2 * xi + w_i * a_i * np.cos(w_i * xi))

#             diff = self.fake_output[j] - self.real_data_noisy_output[j]
#             gplus  =  (dq_dxi * np.exp( diff / self.lambda_val - 1) / self.lambda_val)
#             gminus = (-dq_dxi * np.exp(-diff / self.lambda_val - 1) / self.lambda_val)
#             grad_sum += gplus + gminus
#         return float(grad_sum)

#     def _coord_hessian(self, i: int) -> float:
#         """
#         d^2 L / dx_i^2 at current self.fake_X (exact 1D second derivative).
#         """
#         xi = self.fake_X[i]
#         H_sum = 0.0
#         for j in range(self.k):
#             a_i = self.amplitudes_matrix[j, i]
#             w_i = self.frequencies_vector[j]

#             dq_dxi   = (1.0 / self.n) * (2 * xi + w_i * a_i * np.cos(w_i * xi))
#             d2q_dxi2 = (1.0 / self.n) * (2 - (w_i**2) * a_i * np.sin(w_i * xi))

#             diff = self.fake_output[j] - self.real_data_noisy_output[j]
#             epos = np.exp( diff / self.lambda_val - 1)
#             eneg = np.exp(-diff / self.lambda_val - 1)

#             H_pos = epos * (d2q_dxi2 / self.lambda_val + (dq_dxi**2) / (self.lambda_val**2))
#             H_neg = eneg * (d2q_dxi2 / self.lambda_val + (dq_dxi**2) / (self.lambda_val**2))
#             H_sum += H_pos + H_neg
#         return float(H_sum)

#     # -------------------------------
#     # Loss & line search helpers
#     # -------------------------------
#     def _loss_from_outputs(self, fake_out: np.ndarray) -> float:
#         """Loss given current fake outputs vs current (fixed) noisy reals."""
#         loss_plus  = np.sum(np.exp((fake_out - self.real_data_noisy_output) / self.lambda_val - 1))
#         loss_minus = np.sum(np.exp((self.real_data_noisy_output - fake_out) / self.lambda_val - 1))
#         return float(loss_plus + loss_minus)

#     def _loss_and_fake_output_for_x(self, x_new: np.ndarray) -> Tuple[float, np.ndarray]:
#         """Compute total loss and fake_output for proposed x_new."""
#         k = self.k
#         fake_output_new = np.zeros(k)
#         for j in range(k):
#             amps = self.amplitudes_matrix[j]
#             freq = self.frequencies_vector[j]
#             s = np.sum(x_new**2 + amps * np.sin(freq * x_new))
#             fake_output_new[j] = s / self.n
#         loss = self._loss_from_outputs(fake_output_new)
#         return (loss, fake_output_new)

#     def _directional_step(self, idx: int, step: float, direction: float) -> Tuple[float, np.ndarray, np.ndarray]:
#         """
#         Apply x[idx] <- x[idx] + step * direction (clipped), return new (loss, fake_output, x_new).
#         """
#         x_new = self.fake_X.copy()
#         x_new[idx] = np.clip(x_new[idx] + step * direction, self.lower_bound, self.upper_bound)
#         loss_new, f_out_new = self._loss_and_fake_output_for_x(x_new)
#         return loss_new, f_out_new, x_new

#     # -------------------------------
#     # One outer step (block of m_inner Newton top-1 updates)
#     # -------------------------------
#     def coordinate_descent_step(self) -> Tuple[float, float]:
#         """
#         One outer step:
#           - recompute full gradient EACH inner update
#           - rebuild shortlist via argpartition by |grad|
#           - score shortlist by |g/H| (Newton magnitude), SKIP H <= 0
#           - take up to m_inner Newton steps with Armijo and step-size cap
#         Returns:
#           (total_loss_after, total_loss_update)
#         """
#         # Current loss
#         total_loss = self._loss_from_outputs(self.fake_output)
#         initial_total_loss = total_loss

#         # Armijo/backtracking params
#         alpha = 1e-4
#         beta  = 0.5
#         t_min = 1e-6
#         max_backtracks = 20

#         # Step-size cap to avoid huge Newton jumps when |H| is tiny
#         step_cap = 0.5 * (self.upper_bound - self.lower_bound)

#         L = min(self.shortlist_size, self.num_points)
#         m_block = min(self.m_inner, L)

#         for _ in range(m_block):
#             # ---- Recompute full gradient (fresh) ----
#             grad = np.zeros(self.num_points, dtype=float)
#             for i in range(self.num_points):
#                 grad[i] = self._coord_grad(i)

#             maxg = np.max(np.abs(grad))
#             if maxg < 1e-12:
#                 break  # nothing useful to do

#             # ---- Rebuild shortlist via argpartition by |grad| ----
#             idxs_top = np.argpartition(-np.abs(grad), L - 1)[:L]

#             # ---- Score shortlist by |g/H| (Newton step magnitude), skip H <= 0 ----
#             scores = np.zeros(L, dtype=float)
#             hess_cache = np.zeros(L, dtype=float)
#             for t, idx in enumerate(idxs_top):
#                 g_i = grad[idx]
#                 H_i = self._coord_hessian(idx)
#                 hess_cache[t] = H_i
#                 if H_i > 0.0:
#                     scores[t] = abs(g_i / H_i)
#                 else:
#                     scores[t] = 0.0  # discard non-positive curvature for Newton

#             sel = int(np.argmax(scores))
#             if scores[sel] <= 0.0:
#                 # no usable positive-curvature coords right now
#                 break

#             idx = int(idxs_top[sel])
#             g = grad[idx]
#             H = hess_cache[sel]

#             if abs(g) < 1e-12 or H <= 0.0:
#                 continue

#             # ---- Newton direction with step cap (no H sign enforcement) ----
#             direction = -g / H
#             if direction > 0:
#                 direction = min(direction, step_cap)
#             else:
#                 direction = max(direction, -step_cap)

#             # With H>0, this is <= 0 (descent)
#             dir_deriv = g * direction

#             # ---- Armijo backtracking on this coordinate ----
#             t = 1.0
#             tries = 0
#             while True:
#                 loss_try, fake_output_try, x_try = self._directional_step(idx, t, direction)
#                 if loss_try <= total_loss + alpha * t * dir_deriv:
#                     # accept
#                     self.fake_X = x_try
#                     self.fake_output = fake_output_try
#                     self.error = np.abs(self.fake_output - self.real_data_noisy_output)
#                     total_loss = loss_try
#                     break

#                 t *= beta
#                 tries += 1
#                 if tries >= max_backtracks or t < t_min:
#                     # tiny fallback step
#                     loss_try, fake_output_try, x_try = self._directional_step(idx, t_min, direction)
#                     self.fake_X = x_try
#                     self.fake_output = fake_output_try
#                     self.error = np.abs(self.fake_output - self.real_data_noisy_output)
#                     total_loss = loss_try
#                     break

#         total_loss_after = total_loss
#         total_loss_update = initial_total_loss - total_loss_after
#         return total_loss_after, total_loss_update

#     # -------------------------------
#     # Outer loop
#     # -------------------------------
#     def run_coordinate_descent(self,
#                                max_iterations: int = 1000,
#                                verbose: bool = True,
#                                *,
#                                resample_noise: bool = False) -> dict:
#         """
#         Run until weighted satisfaction >= 0.5 + eta or max_iterations is reached.

#         If resample_noise is False (default), assumes compute_query_outputs() has
#         already been called and will NOT regenerate DP noise. If True, it will
#         call compute_query_outputs() once at the start (and refresh noise).
#         """
#         if self.amplitudes_matrix is None or self.frequencies_vector is None or self.query_weights is None:
#             raise ValueError("Must set amplitude matrix, frequency vector, and weights first")

#         # Only (re)compute outputs/noise if explicitly requested.
#         if resample_noise:
#             self.compute_query_outputs()
#         else:
#             # Sanity: ensure needed arrays exist
#             req = [self.real_data_noisy_output, self.fake_output, self.error]
#             if any(x is None for x in req):
#                 raise RuntimeError("Call compute_query_outputs() once before run_coordinate_descent(resample_noise=False).")

#         target_ratio = 0.5 + self.eta
#         num_iterations = 0
#         total_loss_update = float('inf')
#         total_loss = float('inf')

#         # Initial stats (do NOT resample)
#         initial_weighted_satisfaction = self.compute_weighted_satisfaction_ratio()

#         if verbose:
#             print(f"Initial weighted satisfaction ratio: {initial_weighted_satisfaction:.4f}")
#             print(f"Target weighted satisfaction ratio: {target_ratio:.4f}")
#             print(f"Lambda: {self.lambda_val:.6f}")
#             print(f"Lambda/2: {self.lambda_val/2:.6f}")
#             print(f"Tau: {self.tau}")
#             print(f"Number of queries: {self.k}")
#             print(f"Number of queries above lambda/2 error: {np.sum(self.error > self.lambda_val/2)}")
#             print("NOTE: Stopping when satisfaction ratio >= 0.5 + eta (nonconvex setting).")

#         while (num_iterations < max_iterations and
#                self.compute_weighted_satisfaction_ratio() < target_ratio):
#             total_loss, total_loss_update = self.coordinate_descent_step()
#             num_iterations += 1

#             if verbose and num_iterations % 10 == 0:
#                 current_satisfaction = self.compute_weighted_satisfaction_ratio()
#                 queries_above_threshold = np.sum(self.error > self.lambda_val / 2)
#                 print(
#                     f"Iteration {num_iterations}: "
#                     f"Loss update = {total_loss_update:.6f}, "
#                     f"Weighted satisfaction = {current_satisfaction:.4f}, "
#                     f"Queries above lambda/2 = {queries_above_threshold}"
#                 )

#         # Final stats
#         final_weighted_satisfaction = self.compute_weighted_satisfaction_ratio()
#         final_error_stats = {
#             'mean_error': float(np.mean(self.error)),
#             'max_error': float(np.max(self.error)),
#             'queries_above_lambda_half': int(np.sum(self.error > self.lambda_val / 2))
#         }

#         results = {
#             'num_iterations': num_iterations,
#             'iterations': num_iterations,
#             'converged': (total_loss_update <= self.tau),
#             'target_reached': (final_weighted_satisfaction >= target_ratio),
#             'both_conditions_met': (final_weighted_satisfaction >= target_ratio and total_loss_update <= self.tau),
#             'initial_weighted_satisfaction': float(initial_weighted_satisfaction),
#             'final_weighted_satisfaction': float(final_weighted_satisfaction),
#             'target_ratio': float(target_ratio),
#             'final_loss': float(total_loss),
#             'final_loss_update': float(total_loss_update),
#             'lambda_val': float(self.lambda_val),
#             'tau': float(self.tau),
#             'error_stats': final_error_stats,
#             'fake_X': self.fake_X.copy(),
#             'real_X': self.real_X.copy(),
#             'queries': self.sampled_queries.copy(),
#             'weights': self.query_weights.copy(),
#             'errors': self.error.copy(),
#             'lap_noise': self.lap_noise.copy()
#         }

#         if verbose:
#             print(f"\nAlgorithm completed:")
#             print(f"  Iterations: {num_iterations}")
#             print(f"  Converged (loss update <= tau): {results['converged']}")
#             print(f"  Target reached (weighted satisfaction >= 0.5 + eta): {results['target_reached']}")
#             print(f"  Both conditions met: {results['both_conditions_met']}")
#             print(f"  Final weighted satisfaction: {final_weighted_satisfaction:.4f}")
#             print(f"  Final loss update: {total_loss_update:.6f}")
#             print(f"  Queries above lambda/2: {final_error_stats['queries_above_lambda_half']}")

#         return results


# # -------------------------------
# # Legacy convenience wrapper
# # -------------------------------
# def run_lp_norms_optimization(queries: List[float],
#                               weights: List[float],
#                               epsilon: float = 15.0,
#                               delta: float = 1e-1,
#                               beta: float = 0.05,
#                               eta: float = 0.001,
#                               n: int = 100,
#                               tau: float = 0.01,
#                               upper_bound: float = 1.0,
#                               lower_bound: float = -1.0,
#                               max_iterations: int = 1000,
#                               verbose: bool = True) -> dict:
#     """
#     Legacy convenience function to run the optimization.
#     Note: Sets trivial amplitudes/frequencies just to keep the interface.
#     """
#     optimizer = SpikyNonconvexCoordinateDescent(
#         epsilon=epsilon,
#         delta=delta,
#         beta=beta,
#         eta=eta,
#         n=n,
#         tau=tau,
#         upper_bound=upper_bound,
#         lower_bound=lower_bound
#     )

#     # Dummy amplitudes/frequencies for compatibility
#     amplitudes_matrix = np.ones((len(queries), n), dtype=float)
#     frequencies_vector = np.ones(len(queries), dtype=float) * 5.0
#     optimizer.set_queries_and_amplitudes(amplitudes_matrix, frequencies_vector, weights)

#     optimizer.generate_data()
#     # IMPORTANT: compute once so DP noise is fixed for the run
#     optimizer.compute_query_outputs()
#     results = optimizer.run_coordinate_descent(max_iterations=max_iterations, verbose=verbose, resample_noise=False)
#     return results


# if __name__ == "__main__":
#     print("SpikyNonconvexCoordinateDescent class loaded successfully.")
#     print("Use your test harness to set amplitudes/frequencies and run experiments.")

# import numpy as np
# import math
# from typing import List, Tuple, Optional

# class SpikyNonconvexCoordinateDescent:
#     """
#     Coordinate descent algorithm for spiky nonconvex queries.

#     Query (per query j):
#       f_j(X) = (1/n) * sum_i (x_i^2 + a_{j,i} * sin(w_j * x_i))

#     Loss:
#       L = sum_j [ exp((f_j(fake)-y_j)/λ - 1) + exp((y_j-f_j(fake))/λ - 1) ]

#     Outer step does a block of Newton top-1 coordinate updates:
#       - compute full gradient over all coordinates
#       - shortlist via argpartition by |grad|
#       - score by |g/H| (Newton step magnitude)
#       - pick & update sequentially m_inner coords with backtracking
#         (shortlist is rebuilt after each accepted update)
#     """

#     def __init__(self,
#                  epsilon: float,
#                  delta: float,
#                  beta: float,
#                  eta: float,
#                  n: int,
#                  tau: float = 0.01,
#                  upper_bound: float = math.pi,
#                  lower_bound: float = -math.pi,
#                  frequency: float = 5.0,
#                  data_precision: int = 4,
#                  seed: Optional[int] = None,
#                  # update controls
#                  update_mode: str = "newton",          # keep "newton"
#                  m_inner: int = 12,                    # work per outer step
#                  shortlist_size: int = 48,             # shortlist size L
#                  selection_rule: str = "newton_step",  # rank by |g/H|
#                  newton_damping: float = 0.0,          # reserved
#                  # line-search & stability
#                  armijo_alpha: float = 5e-3,           # not used in accept rule (kept for ref)
#                  armijo_beta: float  = 0.7,            # shrink factor
#                  armijo_t_min: float = 1e-6,           # minimum step to try
#                  cap_step: float = 0.5):               # cap |raw newton step|
#         # Core params
#         self.epsilon = epsilon
#         self.delta = delta
#         self.beta = beta
#         self.eta = eta
#         self.n = n
#         self.tau = tau
#         self.upper_bound = upper_bound
#         self.lower_bound = lower_bound
#         self.frequency = frequency
#         self.data_precision = data_precision

#         # Local RNG
#         self.rng = np.random.default_rng(seed)

#         # Derived params
#         self.num_points = n
#         self.m = self.num_points * math.ceil(math.log2(((upper_bound - lower_bound)/10**(-data_precision)) + 1))
#         self.rho = (self.upper_bound**2 + 1) / self.n

#         # Data / state
#         self.real_X = None
#         self.fake_X = None
#         self.fake_X_original = None
#         self.sampled_queries = None
#         self.query_weights = None
#         self.real_output = None
#         self.real_data_noisy_output = None
#         self.lap_noise = None
#         self.fake_output = None
#         self.error = None
#         self.lambda_val = None
#         self.amplitudes_matrix = None
#         self.frequencies_vector = None
#         self.k = 0

#         # Update controls
#         self.update_mode = update_mode
#         self.m_inner = int(m_inner)
#         self.shortlist_size = int(shortlist_size)
#         self.selection_rule = selection_rule
#         self.newton_damping = float(newton_damping)

#         # Line-search & stability
#         self.armijo_alpha = float(armijo_alpha)
#         self.armijo_beta  = float(armijo_beta)
#         self.armijo_t_min = float(armijo_t_min)
#         self.cap_step = float(cap_step)

#     # -------------------------------
#     # Setup & data generation methods
#     # -------------------------------
#     def set_queries_and_amplitudes(self,
#                                    amplitudes_matrix: np.ndarray,
#                                    frequencies_vector: np.ndarray,
#                                    weights: List[float]):
#         """
#         Set amplitude matrix, frequency vector, and weights.
#           amplitudes_matrix: shape (k, n), entries in (0,1]
#           frequencies_vector: shape (k,), entries > 0
#           weights: list of k nonnegative weights summing to 1
#         """
#         amplitudes_matrix = np.asarray(amplitudes_matrix, dtype=float)
#         frequencies_vector = np.asarray(frequencies_vector, dtype=float)
#         k, n_amps = amplitudes_matrix.shape

#         if len(weights) != k:
#             raise ValueError("Number of amplitude vectors must match number of weights")
#         if len(frequencies_vector) != k:
#             raise ValueError("Number of frequency parameters must match number of queries")
#         if not np.isclose(sum(weights), 1.0, atol=1e-6):
#             raise ValueError("Weights must sum to 1")
#         if n_amps != self.num_points:
#             raise ValueError(f"Amplitude vectors length {n_amps} must equal num_points {self.num_points}")
#         if np.any(amplitudes_matrix <= 0) or np.any(amplitudes_matrix > 1):
#             raise ValueError("All amplitudes must be in (0, 1]")
#         if np.any(frequencies_vector <= 0):
#             raise ValueError("All frequencies must be positive")

#         self.amplitudes_matrix = amplitudes_matrix.copy()
#         self.frequencies_vector = frequencies_vector.copy()
#         self.query_weights = np.array(weights, dtype=float)
#         self.k = k

#         # For compatibility, dummy query types not used in computation
#         self.sampled_queries = np.array([2.0] * k)

#         # lambda = ln(k/beta) * 2 * rho * sqrt(2k * ln(1/delta)) / epsilon
#         self.lambda_val = (math.log(self.k / self.beta) * 2 * self.rho *
#                            math.sqrt(2 * self.k * math.log(1 / self.delta))) / self.epsilon

#         # print(f"Calculated lambda: {self.lambda_val:.6f}")
#         print(f"Set {k} spiky nonconvex queries with different amplitude vectors and frequencies")

#     def generate_data(self,
#                       real_data: Optional[np.ndarray] = None,
#                       initial_fake_data: Optional[np.ndarray] = None):
#         """Generate or set the real and fake data (continuous sampling + local RNG)."""
#         if real_data is not None:
#             self.real_X = real_data.copy()
#         else:
#             self.real_X = self.rng.uniform(self.lower_bound, self.upper_bound, size=self.num_points)

#         if initial_fake_data is not None:
#             self.fake_X = initial_fake_data.copy()
#         else:
#             self.fake_X = self.rng.normal(loc=0.0, scale=1.0, size=self.num_points)

#         self.fake_X_original = self.fake_X.copy()

#     # -------------------------------
#     # Core computations
#     # -------------------------------
#     def compute_query_outputs(self):
#         """Compute query outputs on real and fake data, including DP noise on the real outputs (one draw)."""
#         if self.amplitudes_matrix is None or self.frequencies_vector is None:
#             raise ValueError("Must set amplitude matrix and frequency vector first")

#         k = self.k
#         self.real_output = np.zeros(k)
#         self.real_data_noisy_output = np.zeros(k)
#         self.lap_noise = np.zeros(k)
#         self.fake_output = np.zeros(k)
#         self.error = np.zeros(k)

#         # Noise scale for DP (compute once)
#         noise_scale = self.rho * math.sqrt(2 * k * math.log(1 / self.delta)) / self.epsilon
#         print(f"noise_scale={noise_scale:.6f}, lambda/2={self.lambda_val/2:.6f}")

#         for index in range(k):
#             amps = self.amplitudes_matrix[index]
#             freq = self.frequencies_vector[index]

#             # Real output
#             sum_real = np.sum(self.real_X**2 + amps * np.sin(freq * self.real_X))
#             self.real_output[index] = sum_real / self.n

#             # Add Laplace noise (use local RNG)
#             self.lap_noise[index] = self.rng.laplace(loc=0.0, scale=noise_scale)
#             self.real_data_noisy_output[index] = self.real_output[index] + self.lap_noise[index]

#             # Fake output
#             sum_fake = np.sum(self.fake_X**2 + amps * np.sin(freq * self.fake_X))
#             self.fake_output[index] = sum_fake / self.n

#             # Error
#             self.error[index] = abs(self.fake_output[index] - self.real_data_noisy_output[index])

#     def compute_weighted_satisfaction_ratio(self) -> float:
#         """Weighted ratio of queries satisfying |error| < lambda/2."""
#         if self.query_weights is None:
#             raise ValueError("Must set queries and weights first")
#         satisfied_mask = self.error < self.lambda_val / 2
#         return float(np.sum(self.query_weights[satisfied_mask]))

#     # -------------------------------
#     # Derivatives for a single coordinate
#     # -------------------------------
#     def _coord_grad(self, i: int) -> float:
#         """dL/dx_i at current self.fake_X."""
#         xi = self.fake_X[i]
#         grad_sum = 0.0
#         lam = self.lambda_val
#         n_inv = 1.0 / self.n

#         for j in range(self.k):
#             a_i = self.amplitudes_matrix[j, i]
#             w_i = self.frequencies_vector[j]

#             dq_dxi = n_inv * (2 * xi + w_i * a_i * np.cos(w_i * xi))
#             diff = self.fake_output[j] - self.real_data_noisy_output[j]
#             gplus  =  (dq_dxi * np.exp( diff / lam - 1.0) / lam)
#             gminus = (-dq_dxi * np.exp(-diff / lam - 1.0) / lam)
#             grad_sum += gplus + gminus
#         return float(grad_sum)

#     def _coord_hessian(self, i: int) -> float:
#         """d^2 L / dx_i^2 at current self.fake_X (exact 1D second derivative)."""
#         xi = self.fake_X[i]
#         H_sum = 0.0
#         lam = self.lambda_val
#         n_inv = 1.0 / self.n

#         for j in range(self.k):
#             a_i = self.amplitudes_matrix[j, i]
#             w_i = self.frequencies_vector[j]

#             dq_dxi   = n_inv * (2 * xi + w_i * a_i * np.cos(w_i * xi))
#             d2q_dxi2 = n_inv * (2 - (w_i**2) * a_i * np.sin(w_i * xi))

#             diff = self.fake_output[j] - self.real_data_noisy_output[j]
#             epos = np.exp( diff / lam - 1.0)
#             eneg = np.exp(-diff / lam - 1.0)

#             H_pos = epos * (d2q_dxi2 / lam + (dq_dxi**2) / (lam**2))
#             H_neg = eneg * (d2q_dxi2 / lam + (dq_dxi**2) / (lam**2))
#             H_sum += H_pos + H_neg
#         return float(H_sum)

#     # -------------------------------
#     # Loss helpers
#     # -------------------------------
#     def _loss_from_fake_output(self, f_out: np.ndarray) -> float:
#         diff = (f_out - self.real_data_noisy_output) / self.lambda_val
#         return float(np.sum(np.exp(diff - 1.0) + np.exp(-diff - 1.0)))

#     # -------------------------------
#     # Rank-1 directional step (incremental query update)
#     # -------------------------------
#     def _directional_step(self, idx: int, step: float, direction: float) -> Tuple[float, np.ndarray, np.ndarray]:
#         """
#         Rank-1 trial update on coordinate `idx`:
#           x_new = clip(x_old + step*direction)
#           f_out_try = fake_output + (1/n) * [ (x_new^2 - x_old^2)
#                                               + a[:,idx]*(sin(w * x_new) - sin(w * x_old)) ]
#         Returns: (loss_try, fake_output_try, x_try)
#         """
#         x_old = self.fake_X[idx]
#         x_new = np.clip(x_old + step * direction, self.lower_bound, self.upper_bound)

#         # If no change, just return current state
#         if x_new == x_old:
#             return self._loss_from_fake_output(self.fake_output), self.fake_output, self.fake_X

#         a_col = self.amplitudes_matrix[:, idx]     # (k,)
#         w     = self.frequencies_vector            # (k,)

#         delta_queries = ((x_new**2 - x_old**2) + a_col * (np.sin(w * x_new) - np.sin(w * x_old))) / self.n
#         fake_output_try = self.fake_output + delta_queries

#         loss_try = self._loss_from_fake_output(fake_output_try)

#         x_try = self.fake_X.copy()
#         x_try[idx] = x_new
#         return loss_try, fake_output_try, x_try

#     # -------------------------------
#     # One outer step (block Newton updates)
#     # -------------------------------
#     def coordinate_descent_step(self) -> Tuple[float, float]:
#         """
#         One outer step:
#         - compute full gradient
#         - repeat (up to m_inner):
#             * shortlist via argpartition by |grad|
#             * score shortlist by |g/H| (Newton)
#             * backtracking with capped raw Newton jump
#             * accept only if the loss strictly decreases
#             → update state & refresh grad locally; then rebuild shortlist next pick
#         Returns:
#         (total_loss_after, total_loss_update)
#         """
#         # Current loss
#         total_loss = self._loss_from_fake_output(self.fake_output)
#         initial_total_loss = total_loss

#         # Precompute for speed/readability
#         beta  = self.armijo_beta
#         t_min = self.armijo_t_min
#         cap   = self.cap_step

#         # ---- Full gradient ----
#         grad = np.zeros(self.num_points, dtype=float)
#         for i in range(self.num_points):
#             grad[i] = self._coord_grad(i)

#         # Block of sequential Newton updates
#         taken = 0
#         while taken < self.m_inner:
#             # Rebuild shortlist each pick from current |grad|
#             L = min(self.shortlist_size, self.num_points)
#             idxs_top = np.argpartition(-np.abs(grad), L - 1)[:L]

#             # Score shortlist by |g/H|
#             scores = np.zeros(L, dtype=float)
#             hess_cache = np.zeros(L, dtype=float)
#             for t, idx in enumerate(idxs_top):
#                 g_i = grad[idx]
#                 H_i = self._coord_hessian(idx)
#                 hess_cache[t] = H_i
#                 scores[t] = 0.0 if H_i == 0.0 else abs(g_i / H_i)

#             order = np.argsort(-scores)  # descending by |g/H|

#             # Pick best and try to move
#             moved = False
#             for r in range(L):
#                 sel = order[r]
#                 idx = int(idxs_top[sel])
#                 g = grad[idx]
#                 H = hess_cache[sel]
#                 if H == 0.0 or abs(g) < 1e-14:
#                     continue

#                 # Raw Newton direction, capped
#                 raw = -g / H
#                 direction = cap * np.sign(raw) if abs(raw) > cap else raw

#                 # Backtracking with strictly-decreasing acceptance
#                 t = 1.0
#                 while True:
#                     loss_try, fake_output_try, x_try = self._directional_step(idx, t, direction)

#                     # Strict monotone accept (epsilon guard for FP noise)
#                     if loss_try < total_loss - 1e-12:
#                         # accept
#                         self.fake_X = x_try
#                         self.fake_output = fake_output_try
#                         self.error = np.abs(self.fake_output - self.real_data_noisy_output)
#                         total_loss = loss_try

#                         # refresh just this coordinate's grad entry
#                         grad[idx] = self._coord_grad(idx)

#                         moved = True
#                         taken += 1
#                         break

#                     t *= beta
#                     if t < t_min:
#                         # tiny fallback step ONLY if it decreases loss
#                         loss_try, fake_output_try, x_try = self._directional_step(idx, t_min, direction)
#                         if loss_try < total_loss - 1e-12:
#                             self.fake_X = x_try
#                             self.fake_output = fake_output_try
#                             self.error = np.abs(self.fake_output - self.real_data_noisy_output)
#                             total_loss = loss_try
#                             grad[idx] = self._coord_grad(idx)
#                             moved = True
#                             taken += 1
#                         break  # whether accepted or not, stop backtracking for this idx

#                 if moved:
#                     break  # rebuild shortlist for next pick

#             if not moved:
#                 # no viable move among shortlist → stop early
#                 break

#         total_loss_after = total_loss
#         total_loss_update = initial_total_loss - total_loss_after
#         return total_loss_after, total_loss_update

#     # -------------------------------
#     # Outer loop (no DP resample by default)
#     # -------------------------------
#     def run_coordinate_descent(self, max_iterations: int = 1000, verbose: bool = True, *, resample_noise: bool = False) -> dict:
#         """
#         Run until weighted satisfaction >= 0.5 + eta or max_iterations is reached.

#         If resample_noise=False (default), we DO NOT call compute_query_outputs() here,
#         so the caller's existing DP-noise draw is preserved. Set resample_noise=True
#         if you explicitly want a fresh DP-noise draw at the start of the run.
#         """
#         if self.amplitudes_matrix is None or self.frequencies_vector is None or self.query_weights is None:
#             raise ValueError("Must set amplitude matrix, frequency vector, and weights first")

#         # Only (re)compute outputs if explicitly requested or not yet computed.
#         need_init = (self.fake_output is None) or (self.real_data_noisy_output is None)
#         if resample_noise or need_init:
#             self.compute_query_outputs()

#         target_ratio = 0.5 + self.eta

#         num_iterations = 0
#         total_loss_update = float('inf')
#         total_loss = self._loss_from_fake_output(self.fake_output)

#         initial_weighted_satisfaction = self.compute_weighted_satisfaction_ratio()

#         if verbose:
#             print(f"Initial weighted satisfaction ratio: {initial_weighted_satisfaction:.4f}")
#             print(f"Target weighted satisfaction ratio: {target_ratio:.4f}")
#             print(f"Lambda: {self.lambda_val:.6f}")
#             print(f"Lambda/2: {self.lambda_val/2:.6f}")
#             print(f"Tau (reporting only): {self.tau}")
#             print(f"Number of queries: {self.k}")
#             print(f"Queries above lambda/2: {np.sum(self.error > self.lambda_val/2)}")
#             print("NOTE: Stopping when satisfaction ratio >= 0.5 + eta (nonconvex setting).")

#         # Main loop
#         while (num_iterations < max_iterations and
#                self.compute_weighted_satisfaction_ratio() < target_ratio):
#             total_loss, total_loss_update = self.coordinate_descent_step()
#             num_iterations += 1

#             if verbose and num_iterations % 10 == 0:
#                 current_satisfaction = self.compute_weighted_satisfaction_ratio()
#                 queries_above_threshold = np.sum(self.error > self.lambda_val / 2)
#                 print(
#                     f"Iteration {num_iterations}: "
#                     f"Loss update = {total_loss_update:.6f}, "
#                     f"Weighted satisfaction = {current_satisfaction:.4f}, "
#                     f"Queries above lambda/2 = {queries_above_threshold}"
#                 )

#         # Final stats
#         final_weighted_satisfaction = self.compute_weighted_satisfaction_ratio()
#         final_error_stats = {
#             'mean_error': float(np.mean(self.error)),
#             'max_error': float(np.max(self.error)),
#             'queries_above_lambda_half': int(np.sum(self.error > self.lambda_val / 2))
#         }

#         results = {
#             'num_iterations': num_iterations,
#             'iterations': num_iterations,
#             'converged': (total_loss_update <= self.tau),  # retained for reporting
#             'target_reached': (final_weighted_satisfaction >= target_ratio),
#             'both_conditions_met': (final_weighted_satisfaction >= target_ratio and total_loss_update <= self.tau),
#             'initial_weighted_satisfaction': float(initial_weighted_satisfaction),
#             'final_weighted_satisfaction': float(final_weighted_satisfaction),
#             'target_ratio': float(target_ratio),
#             'final_loss': float(total_loss),
#             'final_loss_update': float(total_loss_update),
#             'lambda_val': float(self.lambda_val),
#             'tau': float(self.tau),
#             'error_stats': final_error_stats,
#             'fake_X': self.fake_X.copy(),
#             'real_X': self.real_X.copy(),
#             'queries': self.sampled_queries.copy(),
#             'weights': self.query_weights.copy(),
#             'errors': self.error.copy(),
#             'lap_noise': self.lap_noise.copy()
#         }

#         if verbose:
#             print(f"\nAlgorithm completed:")
#             print(f"  Iterations: {num_iterations}")
#             print(f"  Converged (loss update <= tau): {results['converged']}")
#             print(f"  Target reached (weighted satisfaction >= 0.5 + eta): {results['target_reached']}")
#             print(f"  Both conditions met: {results['both_conditions_met']}")
#             print(f"  Final weighted satisfaction: {final_weighted_satisfaction:.4f}")
#             print(f"  Final loss update: {total_loss_update:.6f}")
#             print(f"  Queries above lambda/2: {final_error_stats['queries_above_lambda_half']}")

#         return results

import numpy as np
import math
from typing import List, Tuple, Optional

class SpikyNonconvexCoordinateDescent:
    """
    Coordinate descent for spiky nonconvex queries.

    Query j:
      f_j(X) = (1/n) * sum_i (x_i^2 + a_{j,i} * sin(w_j * x_i))

    Loss:
      L = sum_j [ exp((f_j(fake)-y_j)/λ - 1) + exp((y_j-f_j(fake))/λ - 1) ]

    Outer step: block of capped-Newton 1D updates with shortlist selection.
    We accept only strict decreases. Step-size EXPANSION is added to get larger, yet stable, moves.
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
                 # update controls
                 update_mode: str = "newton",
                 m_inner: int = 24,               # ↑ more coordinates per outer step
                 shortlist_size: int = 64,        # ↑ larger shortlist
                 selection_rule: str = "newton_step",
                 newton_damping: float = 0.0,     # unused
                 # line-search & stability
                 armijo_alpha: float = 5e-3,      # not used for acceptance anymore
                 armijo_beta: float  = 0.7,       # backtracking shrink
                 armijo_t_min: float = 1e-6,      # minimum step
                 cap_step: Optional[float] = None,# if None → set adaptively from lambda
                 # expansion for bigger accepted steps
                 gamma_up: float = 1.6,           # step-size expansion factor
                 max_expand: int = 4):            # max expansions if loss keeps dropping
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

        # RNG
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
        self.k = 0

        # Update controls
        self.update_mode = update_mode
        self.m_inner = int(m_inner)
        self.shortlist_size = int(shortlist_size)
        self.selection_rule = selection_rule
        self.newton_damping = float(newton_damping)  # unused

        # Line-search & stability
        self.armijo_alpha = float(armijo_alpha)
        self.armijo_beta  = float(armijo_beta)
        self.armijo_t_min = float(armijo_t_min)
        self.cap_step = cap_step  # may be None → set adaptively after lambda known

        # Expansion
        self.gamma_up = float(gamma_up)
        self.max_expand = int(max_expand)

    # -------------------------------
    # Setup & data generation methods
    # -------------------------------
    def set_queries_and_amplitudes(self,
                                   amplitudes_matrix: np.ndarray,
                                   frequencies_vector: np.ndarray,
                                   weights: List[float]):
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
        self.sampled_queries = np.array([2.0] * k)

        # lambda = ln(k/beta) * 2 * rho * sqrt(2k * ln(1/delta)) / epsilon
        self.lambda_val = (math.log(self.k / self.beta) * 2 * self.rho *
                           math.sqrt(2 * self.k * math.log(1 / self.delta))) / self.epsilon

        # If cap not specified, set it relative to lambda (bigger moves)
        if self.cap_step is None:
            self.cap_step = 0.4 * self.lambda_val  # heuristic; ~raw Newton magnitude

        print(f"Calculated lambda: {self.lambda_val:.6f}")
        print(f"Set {k} spiky nonconvex queries with different amplitude vectors and frequencies")

    def generate_data(self,
                      real_data: Optional[np.ndarray] = None,
                      initial_fake_data: Optional[np.ndarray] = None):
        if real_data is not None:
            self.real_X = real_data.copy()
        else:
            self.real_X = self.rng.uniform(self.lower_bound, self.upper_bound, size=self.num_points)

        if initial_fake_data is not None:
            self.fake_X = initial_fake_data.copy()
        else:
            self.fake_X = self.rng.normal(loc=0.0, scale=1.0, size=self.num_points)

        self.fake_X_original = self.fake_X.copy()

    # -------------------------------
    # Core computations
    # -------------------------------
    def compute_query_outputs(self):
        if self.amplitudes_matrix is None or self.frequencies_vector is None:
            raise ValueError("Must set amplitude matrix and frequency vector first")

        k = self.k
        self.real_output = np.zeros(k)
        self.real_data_noisy_output = np.zeros(k)
        self.lap_noise = np.zeros(k)
        self.fake_output = np.zeros(k)
        self.error = np.zeros(k)

        noise_scale = self.rho * math.sqrt(2 * k * math.log(1 / self.delta)) / self.epsilon

        for index in range(k):
            amps = self.amplitudes_matrix[index]
            freq = self.frequencies_vector[index]

            sum_real = np.sum(self.real_X**2 + amps * np.sin(freq * self.real_X))
            self.real_output[index] = sum_real / self.n

            self.lap_noise[index] = self.rng.laplace(loc=0.0, scale=noise_scale)
            self.real_data_noisy_output[index] = self.real_output[index] + self.lap_noise[index]

            sum_fake = np.sum(self.fake_X**2 + amps * np.sin(freq * self.fake_X))
            self.fake_output[index] = sum_fake / self.n

            self.error[index] = abs(self.fake_output[index] - self.real_data_noisy_output[index])

    def compute_weighted_satisfaction_ratio(self) -> float:
        if self.query_weights is None:
            raise ValueError("Must set queries and weights first")
        satisfied_mask = self.error < self.lambda_val / 2
        return float(np.sum(self.query_weights[satisfied_mask]))

    # -------------------------------
    # Derivatives for a single coordinate
    # -------------------------------
    def _coord_grad(self, i: int) -> float:
        xi = self.fake_X[i]
        grad_sum = 0.0
        lam = self.lambda_val
        n_inv = 1.0 / self.n

        for j in range(self.k):
            a_i = self.amplitudes_matrix[j, i]
            w_i = self.frequencies_vector[j]

            dq_dxi = n_inv * (2 * xi + w_i * a_i * np.cos(w_i * xi))
            diff = self.fake_output[j] - self.real_data_noisy_output[j]

            gplus  =  (dq_dxi * np.exp( diff / lam - 1.0) / lam)
            gminus = (-dq_dxi * np.exp(-diff / lam - 1.0) / lam)
            grad_sum += gplus + gminus
        return float(grad_sum)

    def _coord_hessian(self, i: int) -> float:
        xi = self.fake_X[i]
        H_sum = 0.0
        lam = self.lambda_val
        n_inv = 1.0 / self.n

        for j in range(self.k):
            a_i = self.amplitudes_matrix[j, i]
            w_i = self.frequencies_vector[j]

            dq_dxi   = n_inv * (2 * xi + w_i * a_i * np.cos(w_i * xi))
            d2q_dxi2 = n_inv * (2 - (w_i**2) * a_i * np.sin(w_i * xi))

            diff = self.fake_output[j] - self.real_data_noisy_output[j]
            epos = np.exp( diff / lam - 1.0)
            eneg = np.exp(-diff / lam - 1.0)

            H_pos = epos * (d2q_dxi2 / lam + (dq_dxi**2) / (lam**2))
            H_neg = eneg * (d2q_dxi2 / lam + (dq_dxi**2) / (lam**2))
            H_sum += H_pos + H_neg
        return float(H_sum)

    # -------------------------------
    # Loss helpers
    # -------------------------------
    def _loss_from_fake_output(self, f_out: np.ndarray) -> float:
        diff = (f_out - self.real_data_noisy_output) / self.lambda_val
        return float(np.sum(np.exp(diff - 1.0) + np.exp(-diff - 1.0)))

    # -------------------------------
    # Rank-1 directional step (kept as-is)
    # -------------------------------
    def _directional_step(self, idx: int, step: float, direction: float) -> Tuple[float, np.ndarray, np.ndarray]:
        x_old = self.fake_X[idx]
        x_new = np.clip(x_old + step * direction, self.lower_bound, self.upper_bound)

        if x_new == x_old:
            return self._loss_from_fake_output(self.fake_output), self.fake_output, self.fake_X

        a_col = self.amplitudes_matrix[:, idx]     # (k,)
        w     = self.frequencies_vector            # (k,)

        delta_queries = ((x_new**2 - x_old**2) + a_col * (np.sin(w * x_new) - np.sin(w * x_old))) / self.n
        fake_output_try = self.fake_output + delta_queries

        loss_try = self._loss_from_fake_output(fake_output_try)

        x_try = self.fake_X.copy()
        x_try[idx] = x_new
        return loss_try, fake_output_try, x_try

    # -------------------------------
    # One outer step (with expansion)
    # -------------------------------
    def coordinate_descent_step(self) -> Tuple[float, float]:
        """
        Block of capped-Newton 1D updates.
        - shortlist by |grad|
        - pick by |g/H|
        - try t=1; if decreases loss, EXPAND t by gamma_up up to max_expand while decreasing
        - else backtrack by beta down to t_min
        - accept only strict decreases
        """
        total_loss = self._loss_from_fake_output(self.fake_output)
        initial_total_loss = total_loss

        beta  = self.armijo_beta
        t_min = self.armijo_t_min
        cap   = self.cap_step
        gamma_up = self.gamma_up
        max_expand = self.max_expand

        # Full gradient
        grad = np.zeros(self.num_points, dtype=float)
        for i in range(self.num_points):
            grad[i] = self._coord_grad(i)

        taken = 0
        while taken < self.m_inner:
            L = min(self.shortlist_size, self.num_points)
            idxs_top = np.argpartition(-np.abs(grad), L - 1)[:L]

            # Hessians and scores
            scores = np.zeros(L, dtype=float)
            hess_cache = np.zeros(L, dtype=float)
            for t, idx in enumerate(idxs_top):
                g_i = grad[idx]
                H_i = self._coord_hessian(idx)
                hess_cache[t] = H_i
                scores[t] = 0.0 if H_i == 0.0 else abs(g_i / H_i)

            order = np.argsort(-scores)

            moved = False
            for r in range(L):
                sel = order[r]
                idx = int(idxs_top[sel])
                g = grad[idx]
                H = hess_cache[sel]
                if H == 0.0 or abs(g) < 1e-14:
                    continue

                # capped Newton
                raw = -g / H
                direction = cap * np.sign(raw) if abs(raw) > cap else raw

                # 1) try t=1
                t = 1.0
                loss_try, fake_output_try, x_try = self._directional_step(idx, t, direction)

                if loss_try < total_loss - 1e-12:
                    # 2) EXPANSION: keep increasing t while loss decreases
                    expand_count = 0
                    best_loss = loss_try
                    best_fo = fake_output_try
                    best_x = x_try
                    t_exp = t

                    while expand_count < max_expand:
                        t_next = t_exp * gamma_up
                        loss_next, fo_next, x_next = self._directional_step(idx, t_next, direction)
                        if loss_next < best_loss - 1e-12:
                            best_loss = loss_next
                            best_fo = fo_next
                            best_x = x_next
                            t_exp = t_next
                            expand_count += 1
                        else:
                            break

                    # accept best expanded step
                    self.fake_X = best_x
                    self.fake_output = best_fo
                    self.error = np.abs(self.fake_output - self.real_data_noisy_output)
                    total_loss = best_loss

                    # refresh grad entry (simple: recompute this coord)
                    grad[idx] = self._coord_grad(idx)

                    taken += 1
                    moved = True
                    break

                # 3) otherwise BACKTRACK
                t = 1.0
                while True:
                    t *= beta
                    if t < t_min:
                        # tiny fallback only if it decreases loss
                        loss_min, fo_min, x_min = self._directional_step(idx, t_min, direction)
                        if loss_min < total_loss - 1e-12:
                            self.fake_X = x_min
                            self.fake_output = fo_min
                            self.error = np.abs(self.fake_output - self.real_data_noisy_output)
                            total_loss = loss_min
                            grad[idx] = self._coord_grad(idx)
                            taken += 1
                            moved = True
                        break

                    loss_bt, fo_bt, x_bt = self._directional_step(idx, t, direction)
                    if loss_bt < total_loss - 1e-12:
                        self.fake_X = x_bt
                        self.fake_output = fo_bt
                        self.error = np.abs(self.fake_output - self.real_data_noisy_output)
                        total_loss = loss_bt
                        grad[idx] = self._coord_grad(idx)
                        taken += 1
                        moved = True
                        break

            if not moved:
                break  # nothing viable in shortlist

        total_loss_after = total_loss
        total_loss_update = initial_total_loss - total_loss_after
        return total_loss_after, total_loss_update

    # -------------------------------
    # Outer loop
    # -------------------------------
    def run_coordinate_descent(self, max_iterations: int = 1000, verbose: bool = True, *, resample_noise: bool = False) -> dict:
        if self.amplitudes_matrix is None or self.frequencies_vector is None or self.query_weights is None:
            raise ValueError("Must set amplitude matrix, frequency vector, and weights first")

        self.compute_query_outputs()
        target_ratio = 0.5 + self.eta

        num_iterations = 0
        total_loss_update = float('inf')
        total_loss = self._loss_from_fake_output(self.fake_output)

        initial_weighted_satisfaction = self.compute_weighted_satisfaction_ratio()

        if verbose:
            print(f"Initial weighted satisfaction ratio: {initial_weighted_satisfaction:.4f}")
            print(f"Target weighted satisfaction ratio: {target_ratio:.4f}")
            print(f"Lambda: {self.lambda_val:.6f}")
            print(f"Lambda/2: {self.lambda_val/2:.6f}")
            print(f"Tau (reporting only): {self.tau}")
            print(f"Number of queries: {self.k}")
            print(f"Queries above lambda/2: {np.sum(self.error > self.lambda_val/2)}")
            print("NOTE: Stopping when satisfaction ratio >= 0.5 + eta (nonconvex setting).")

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

        final_weighted_satisfaction = self.compute_weighted_satisfaction_ratio()
        final_error_stats = {
            'mean_error': float(np.mean(self.error)),
            'max_error': float(np.max(self.error)),
            'queries_above_lambda_half': int(np.sum(self.error > self.lambda_val / 2))
        }

        results = {
            'num_iterations': num_iterations,
            'iterations': num_iterations,
            'converged': (total_loss_update <= self.tau),
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
