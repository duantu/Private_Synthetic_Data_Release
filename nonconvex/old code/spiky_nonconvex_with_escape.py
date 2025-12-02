import numpy as np
from typing import Tuple
from spiky_nonconvex_newton_gd import SpikyNonconvexCoordinateDescent as _Base

class SpikyNonconvexCoordinateDescent(_Base):
    """
    Drop-in subclass that adds stuck detection and escape jumps.
    Assumes the base class provides:
      - self.n, self.k, self.num_points
      - self.fake_X, self.real_X
      - self.fake_output, self.real_data_noisy_output
      - self.lambda_val, self.tau, self.eta, self.epsilon, self.delta, self.beta
      - self.lower_bound, self.upper_bound
      - self.rng (np.random.Generator or compatible)
      - self.sampled_queries, self.query_weights, self.amplitudes_matrix, self.frequencies_vector
      - methods: compute_query_outputs(), compute_weighted_satisfaction_ratio(),
                 coordinate_descent_step()
      - attribute it maintains: self.error (|fake_output - real_data_noisy_output|)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # anti-stuck controls (tunable)
        self.stuck_patience: int = 200
        self.small_update_tol: float = 1e-6
        self.escape_jump_frac: float = 0.60
        self.escape_topL: int = 64
        self.escape_random_frac: float = 0.02
        self.escape_accept_rel: float = 1.001

    # -------------------------------
    # Escape-from-stuck helper
    # -------------------------------
    def _escape_from_stuck(self) -> Tuple[bool, float]:
        """Try to escape a flat/stuck region by a large, safe jump."""
        diff = (self.fake_output - self.real_data_noisy_output) / self.lambda_val
        best_loss = float(np.sum(np.exp(diff - 1.0) + np.exp(-diff - 1.0)))
        best_state = None  # (x_new, fake_out_new)

        def _fake_output_for_x(x_vec):
            k = self.k
            out = np.zeros(k, dtype=float)
            for j in range(k):
                amps = self.amplitudes_matrix[j]
                freq = self.frequencies_vector[j]
                s = np.sum(x_vec**2 + amps * np.sin(freq * x_vec))
                out[j] = s / self.n
            return out

        # (a) Directed big jumps on top-L gradients
        L = min(self.escape_topL, self.num_points)
        grad = np.zeros(self.num_points, dtype=float)
        for i in range(self.num_points):
            grad[i] = self._coord_grad(i)

        if np.max(np.abs(grad)) >= 1e-12:
            idxs_top = np.argpartition(-np.abs(grad), L - 1)[:L]
            span = (self.upper_bound - self.lower_bound)
            step = max(1e-8, self.escape_jump_frac * span)

            for idx in idxs_top:
                g = grad[idx]
                if g == 0.0:
                    continue
                direction = -np.sign(g)
                x_try = self.fake_X.copy()
                x_try[idx] = np.clip(x_try[idx] + step * direction, self.lower_bound, self.upper_bound)
                f_out_try = _fake_output_for_x(x_try)
                d = (f_out_try - self.real_data_noisy_output) / self.lambda_val
                loss_try = float(np.sum(np.exp(d - 1.0) + np.exp(-d - 1.0)))
                if loss_try < best_loss:
                    best_loss = loss_try
                    best_state = (x_try, f_out_try)

        # (b) Randomized kick on a small subset (fallback)
        if best_state is None:
            x_try = self.fake_X.copy()
            k = max(1, int(self.escape_random_frac * self.num_points))
            idxs = self.rng.choice(self.num_points, size=k, replace=False)
            x_try[idxs] = self.rng.uniform(self.lower_bound, self.upper_bound, size=k)
            f_out_try = _fake_output_for_x(x_try)
            d = (f_out_try - self.real_data_noisy_output) / self.lambda_val
            loss_try = float(np.sum(np.exp(d - 1.0) + np.exp(-d - 1.0)))
            if loss_try <= best_loss * self.escape_accept_rel:
                best_loss = loss_try
                best_state = (x_try, f_out_try)

        if best_state is not None:
            self.fake_X, self.fake_output = best_state
            self.error = np.abs(self.fake_output - self.real_data_noisy_output)
            return True, float(best_loss)
        return False, float(best_loss)

    # -------------------------------
    # Run with stuck detection
    # -------------------------------
    def run_coordinate_descent(
        self,
        max_iterations: int = 1000,
        verbose: bool = True,
        *,
        resample_noise: bool = False  # kept for API compatibility; unused here
    ) -> dict:
        # Establish the initial state used by the runner
        self.compute_query_outputs()

        target_ratio = 0.5 + self.eta

        # Snapshot initial metrics
        total_loss = float(
            np.sum(
                np.exp((self.fake_output - self.real_data_noisy_output) / self.lambda_val - 1.0)
                + np.exp((self.real_data_noisy_output - self.fake_output) / self.lambda_val - 1.0)
            )
        )
        loss_before = float(total_loss)
        initial_weighted_satisfaction = float(self.compute_weighted_satisfaction_ratio())
        total_loss_update = float('inf')
        num_iterations = 0

        if verbose:
            print(f"Initial weighted satisfaction ratio: {initial_weighted_satisfaction:.4f}")
            print(f"Target weighted satisfaction ratio: {target_ratio:.4f}")
            print(f"Lambda: {self.lambda_val:.6f}")
            print(f"Tau: {self.tau}")
            print(f"Number of queries: {self.k}")
            print(f"Number of queries above λ/2: {int(np.sum(self.error > self.lambda_val/2))}")
            print("NOTE: Stopping when satisfaction ratio >= 0.5 + eta.")

        # Early exit: already satisfied
        if initial_weighted_satisfaction >= target_ratio:
            final_weighted_satisfaction = initial_weighted_satisfaction
            final_error_stats = {
                'mean_error': float(np.mean(self.error)),
                'max_error': float(np.max(self.error)),
                'min_error': float(np.min(self.error)),
                'std_error': float(np.std(self.error)),
                'median_error': float(np.median(self.error)),
                'queries_above_lambda_half': int(np.sum(self.error > self.lambda_val / 2)),
            }
            return {
                'converged': True,
                'target_reached': True,
                'both_conditions_met': True,
                'initial_weighted_satisfaction': float(initial_weighted_satisfaction),
                'final_weighted_satisfaction': float(final_weighted_satisfaction),
                'target_ratio': float(target_ratio),
                'final_loss': float(total_loss),
                'final_loss_update': 0.0,
                'loss_before': float(loss_before),
                'loss_after': float(total_loss),
                'lambda_val': float(self.lambda_val),
                'tau': float(self.tau),
                'error_stats': final_error_stats,
                'fake_X': self.fake_X.copy(),
                'real_X': self.real_X.copy(),
                'queries': self.sampled_queries.copy(),
                'weights': self.query_weights.copy(),
                'eps_delta_beta_eta': (self.epsilon, self.delta, self.beta, self.eta),
                'lap_noise': getattr(self, "lap_noise", None),
                'iterations': 0,
                'num_iterations': 0,
            }

        # Otherwise, optimize
        from collections import deque
        recent_updates = deque(maxlen=self.stuck_patience)

        while (num_iterations < max_iterations and
               float(self.compute_weighted_satisfaction_ratio()) < target_ratio):
            total_loss, total_loss_update = self.coordinate_descent_step()
            num_iterations += 1

            # Anti-stuck tracking
            recent_updates.append(abs(float(total_loss_update)))
            if (len(recent_updates) == self.stuck_patience and
                all(u < self.small_update_tol for u in recent_updates)):
                if verbose:
                    print(f"[escape] flat for {self.stuck_patience} iters (ΔL < {self.small_update_tol:g}); attempting big jump...")
                accepted, new_loss = self._escape_from_stuck()
                recent_updates.clear()
                if verbose:
                    print(f"[escape] {'accepted' if accepted else 'rejected'}; loss -> {new_loss:.6f}")

            if verbose and num_iterations % 10 == 0:
                current_satisfaction = float(self.compute_weighted_satisfaction_ratio())
                q_above = int(np.sum(self.error > self.lambda_val / 2))
                print(f"Iteration {num_iterations}: ΔL = {total_loss_update:.6e}, "
                      f"satisfaction = {current_satisfaction:.4f}, queries above λ/2 = {q_above}")

        final_weighted_satisfaction = float(self.compute_weighted_satisfaction_ratio())
        final_error_stats = {
            'mean_error': float(np.mean(self.error)),
            'max_error': float(np.max(self.error)),
            'min_error': float(np.min(self.error)),
            'std_error': float(np.std(self.error)),
            'median_error': float(np.median(self.error)),
            'queries_above_lambda_half': int(np.sum(self.error > self.lambda_val / 2)),
        }

        results = {
            'converged': (final_weighted_satisfaction >= target_ratio),
            'target_reached': (final_weighted_satisfaction >= target_ratio),
            'both_conditions_met': (final_weighted_satisfaction >= target_ratio and total_loss_update <= self.tau),
            'initial_weighted_satisfaction': float(initial_weighted_satisfaction),
            'final_weighted_satisfaction': float(final_weighted_satisfaction),
            'target_ratio': float(target_ratio),
            'final_loss': float(total_loss),
            'final_loss_update': float(total_loss_update),
            'loss_before': float(loss_before),
            'loss_after': float(total_loss),
            'lambda_val': float(self.lambda_val),
            'tau': float(self.tau),
            'error_stats': final_error_stats,
            'fake_X': self.fake_X.copy(),
            'real_X': self.real_X.copy(),
            'queries': self.sampled_queries.copy(),
            'weights': self.query_weights.copy(),
            'eps_delta_beta_eta': (self.epsilon, self.delta, self.beta, self.eta),
            'lap_noise': getattr(self, "lap_noise", None),
            'iterations': num_iterations,
            'num_iterations': num_iterations,
        }
        return results

    # --- fallback finite-diff gradient if base doesn't provide one ---
    def _coord_grad(self, i: int) -> float:
        h = 1e-6
        xi = self.fake_X[i]
        self.fake_X[i] = min(self.upper_bound, xi + h)
        f_plus = self._loss_given_fakeX(self.fake_X)
        self.fake_X[i] = max(self.lower_bound, xi - h)
        f_minus = self._loss_given_fakeX(self.fake_X)
        self.fake_X[i] = xi
        return (f_plus - f_minus) / (2 * h)

    def _loss_given_fakeX(self, x_vec) -> float:
        k = self.k
        out = np.zeros(k, dtype=float)
        for j in range(k):
            amps = self.amplitudes_matrix[j]
            freq = self.frequencies_vector[j]
            s = np.sum(x_vec**2 + amps * np.sin(freq * x_vec))
            out[j] = s / self.n
        d = (out - self.real_data_noisy_output) / self.lambda_val
        return float(np.sum(np.exp(d - 1.0) + np.exp(-d - 1.0)))
