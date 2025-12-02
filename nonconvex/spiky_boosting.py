# # """
# # Boosting algorithm for Spiky Nonconvex Queries.

# # Implements query boosting where, on each round t, we train a base
# # synopsis A_t and keep all synopses. The output for a query q is the
# # median over i=1..T of A_i(q). Early stopping checks that condition
# # after each round.

# # Key points:
# # - Uses the ENTIRE query set Q each round (no subsampling).
# # - λ and ρ are read from the base optimizer (spiky_nonconvex_newton_gd).
# # - μ is computed from the boosting theorem using the SAME ρ, and with
# #   the edge γ identified as your config η.
# # - We capture and report how many iterations the base optimizer ran.
# # """

# # import math
# # from typing import Optional, Dict, List

# # import numpy as np

# # from spiky_nonconvex_newton_gd import SpikyNonconvexCoordinateDescent


# # def _safe_log_one_over(x: float, eps: float = 1e-12) -> float:
# #     """Stable log(1/x)."""
# #     return math.log(1.0 / max(x, eps))


# # def compute_mu_from_theorem(
# #     rho: float,
# #     k: int,
# #     epsilon_sample: float,   # ε_sample
# #     delta_sample: float,     # δ_sample
# #     gamma_margin: float,     # η (edge)
# #     T: int,                  # boosting horizon
# #     *,
# #     C: float = 1.0,
# #     exponent_gamma: int = 3,  # kept for backwards compatibility; not used now
# # ) -> float:
# #     """
# #     Compute μ using the theorem:

# #         μ = C * [ sqrt(2 k ln((1+2η)/(1-2η))) * T^{3/2} * ρ *
# #                    ( sqrt(log(1/δ_sample)) + sqrt(log(1/δ_sample) + 2 ε_sample) )
# #                 ] / ε_sample

# #     Parameters
# #     ----------
# #     rho          : ρ in the theorem (sensitivity of base mechanism)
# #     k            : |Q|
# #     epsilon_sample : ε_sample
# #     delta_sample : δ_sample
# #     gamma_margin : η (edge)
# #     T            : boosting horizon (number of rounds)
# #     C            : hidden constant to tune μ up/down
# #     """

# #     if epsilon_sample <= 0:
# #         raise ValueError("epsilon_sample must be positive.")
# #     if not (0.0 < delta_sample < 1.0):
# #         raise ValueError("delta_sample must lie in (0,1).")
# #     if not (0.0 < gamma_margin < 0.5):
# #         raise ValueError("gamma_margin η must lie in (0, 0.5) for ln((1+2η)/(1-2η)) to be well-defined.")
# #     if T <= 0:
# #         raise ValueError("T must be positive.")

# #     # log((1+2η)/(1-2η))
# #     log_eta = math.log((1.0 + 2.0 * gamma_margin) / (1.0 - 2.0 * gamma_margin))

# #     # sqrt(log(1/δ_sample)) and sqrt(log(1/δ_sample) + 2 ε_sample)
# #     log_one_over_delta = _safe_log_one_over(delta_sample)
# #     inner = math.sqrt(log_one_over_delta) + math.sqrt(log_one_over_delta + 2.0 * epsilon_sample)

# #     prefactor = math.sqrt(2.0 * k * log_eta) * (T ** 1.5) * rho

# #     mu = C * prefactor * inner / float(epsilon_sample)
# #     return float(mu)


# # class SpikyBoostingAlgorithm:
# #     def __init__(
# #         self,
# #         k: int,
# #         lambda_param: Optional[float],
# #         eta: float,
# #         rho: Optional[float],
# #         mu: Optional[float],
# #         T: int,                  # practical boosting horizon (T_cap)
# #         epsilon_sample: float,
# #         delta: float,
# #         beta: float,
# #         n: int,
# #         upper_bound: float = math.pi,
# #         lower_bound: float = -math.pi,
# #         tau: float = 1e-2,
# #         *,
# #         mu_constant: float = 1.0,
# #         mu_gamma_exponent: int = 3,
# #         min_rounds: int = 1,
# #         stop_threshold_abs: Optional[float] = None,
# #         max_base_restarts: int = 1000,
# #         T_theorem: Optional[int] = None,   # NEW
# #     ):
# #         self.k = int(k)
# #         self.lambda_param = None if lambda_param is None else float(lambda_param)
# #         self.eta = float(eta)
# #         self.rho = None if rho is None else float(rho)
# #         self.mu = None if mu is None else float(mu)
# #         self.T = int(T)
# #         self.epsilon_sample = float(epsilon_sample)
# #         self.delta = float(delta)
# #         self.beta = float(beta)
# #         self.n = int(n)
# #         self.upper_bound = float(upper_bound)
# #         self.lower_bound = float(lower_bound)
# #         self.tau = float(tau)
# #         self.max_base_restarts = int(max_base_restarts)

# #         self.mu_constant = float(mu_constant)
# #         self.mu_gamma_exponent = int(mu_gamma_exponent)
# #         self.min_rounds = int(min_rounds)
# #         self.stop_threshold_abs = None if stop_threshold_abs is None else float(stop_threshold_abs)

# #         # NEW: choose T_theorem; default to min(self.T, 10)
# #         if T_theorem is None:
# #             self.T_theorem = min(self.T, 5)
# #         else:
# #             self.T_theorem = int(T_theorem)

# #         self.alpha = 0.5 * math.log((1.0 + 2.0 * self.eta) / (1.0 - 2.0 * self.eta))

# #         self.Q: Optional[Dict[str, np.ndarray]] = None
# #         self.num_queries: Optional[int] = None
# #         self.all_synopses: List[Dict] = []


# #     # ---------- query plumbing ----------

# #     def set_queries(self, amplitudes_matrix: np.ndarray, frequencies_vector: np.ndarray):
# #         A = np.asarray(amplitudes_matrix, dtype=float)
# #         W = np.asarray(frequencies_vector, dtype=float)
# #         if A.shape[0] != W.shape[0]:
# #             raise ValueError("Number of amplitude vectors must match number of frequencies.")
# #         if A.shape[1] != self.n:
# #             raise ValueError(f"Amplitude vector length {A.shape[1]} must equal n {self.n}.")
# #         self.Q = {"amplitudes": A, "frequencies": W}
# #         self.num_queries = int(W.shape[0])

# #     def compute_query_answer(self, synopsis_data: np.ndarray, query_idx: int) -> float:
# #         amps = self.Q["amplitudes"][query_idx]
# #         w = self.Q["frequencies"][query_idx]
# #         return float(np.sum(synopsis_data ** 2 + amps * np.sin(w * synopsis_data)) / self.n)

# #     def is_lambda_accurate(self, error: float, lambda_param: float) -> bool:
# #         return error <= lambda_param

# #     def is_lambda_mu_accurate(self, error: float, lambda_param: float, mu: float) -> bool:
# #         return error <= (lambda_param + mu)

# #     # ---------- boosting loop ----------

# #     def run_boosting(self, real_data: np.ndarray, verbose: bool = True) -> dict:
# #         if self.Q is None:
# #             raise ValueError("Must set queries first via set_queries().")

# #         D = np.ones(self.num_queries, dtype=float) / float(self.num_queries)
# #         u = np.ones(self.num_queries, dtype=float)

# #         if verbose:
# #             print("=== Boosting Algorithm ===")

# #         self.all_synopses.clear()
# #         cumulative_accuracy = np.zeros(self.num_queries, dtype=float)
# #         early_stopped = False
# #         iterations_run = 0

# #         # ground truth answers (fixed, based on the real_data passed in)
# #         true_answers = np.zeros(self.num_queries, dtype=float)
# #         for q in range(self.num_queries):
# #             amps = self.Q["amplitudes"][q]
# #             w = self.Q["frequencies"][q]
# #             true_answers[q] = float(np.sum(real_data ** 2 + amps * np.sin(w * real_data)) / self.n)

# #         for t in range(1, self.T + 1):
# #             if verbose:
# #                 print(f"--- Iteration {t}/{self.T} ---")

# #             # === Use ALL queries; pass D as weights to the base optimizer ===
# #             sampled_indices = np.arange(self.num_queries, dtype=int)
# #             sampled_amplitudes = self.Q["amplitudes"][sampled_indices]
# #             sampled_frequencies = self.Q["frequencies"][sampled_indices]
# #             sampled_weights = D.astype(float).tolist()

# #             # === Run base optimizer, with restarts until satisfaction ≥ 0.5 + eta ===
# #             target_ratio = 0.5 + self.eta
# #             best_optimizer = None
# #             best_ret = None
# #             best_satisfaction = -1.0
# #             base_iters = None

# #             for attempt in range(1, self.max_base_restarts + 1):
# #                 if verbose:
# #                     print(f"Running base synopsis generator (attempt {attempt}/{self.max_base_restarts})...")

# #                 optimizer = SpikyNonconvexCoordinateDescent(
# #                     epsilon=self.epsilon_sample,
# #                     delta=self.delta,
# #                     beta=self.beta,
# #                     eta=self.eta,  # your 'eta' is used inside base optimizer as well
# #                     n=self.n,
# #                     tau=self.tau,
# #                     upper_bound=self.upper_bound,
# #                     lower_bound=self.lower_bound,
# #                 )
# #                 optimizer.set_queries_and_amplitudes(sampled_amplitudes, sampled_frequencies, sampled_weights)

# #                 # IMPORTANT: keep the same real_data across all attempts and rounds
# #                 optimizer.generate_data(real_data=real_data)

# #                 # Run base optimizer and capture how many iterations it actually ran
# #                 ret = optimizer.run_coordinate_descent(max_iterations=1000, verbose=False)

# #                 # Satisfaction ratio after this attempt
# #                 sat = float(optimizer.compute_weighted_satisfaction_ratio())
# #                 if sat > best_satisfaction:
# #                     best_satisfaction = sat
# #                     best_optimizer = optimizer
# #                     best_ret = ret

# #                 # Extract iteration count
# #                 base_iters_local = None
# #                 if isinstance(ret, dict):
# #                     for key in ("num_iterations", "iterations", "iters", "t", "steps", "n_iter"):
# #                         if key in ret:
# #                             try:
# #                                 base_iters_local = int(ret[key])
# #                                 break
# #                             except Exception:
# #                                 pass
# #                 if base_iters_local is None:
# #                     for attr in ("num_iterations", "iterations_run", "last_num_iterations",
# #                                  "iters_run", "n_iter_", "n_iter"):
# #                         if hasattr(optimizer, attr):
# #                             try:
# #                                 base_iters_local = int(getattr(optimizer, attr))
# #                                 break
# #                             except Exception:
# #                                 pass

# #                 if verbose:
# #                     print(
# #                         f"  attempt {attempt}: satisfaction={sat:.4f} "
# #                         f"(target={target_ratio:.4f}), "
# #                         f"iters={base_iters_local if base_iters_local is not None else 'unknown'}"
# #                     )

# #                 # If this attempt is "truly satisfactory", accept and stop restarting
# #                 if sat >= target_ratio:
# #                     base_iters = base_iters_local
# #                     break

# #             # If never hit target_ratio, fall back to the best attempt
# #             if base_iters is None:
# #                 if isinstance(best_ret, dict):
# #                     for key in ("num_iterations", "iterations", "iters", "t", "steps", "n_iter"):
# #                         if key in best_ret:
# #                             try:
# #                                 base_iters = int(best_ret[key])
# #                                 break
# #                             except Exception:
# #                                 pass
# #                 if verbose:
# #                     print(
# #                         f"WARNING: base synopsis never reached satisfaction ≥ {target_ratio:.4f} "
# #                         f"after {self.max_base_restarts} attempts. "
# #                         f"Using best attempt with satisfaction={best_satisfaction:.4f}."
# #                     )

# #             optimizer = best_optimizer
# #             ret = best_ret

# #             if verbose:
# #                 print(
# #                     f"Base synopsis generator iterations (final accepted attempt): "
# #                     f"{base_iters if base_iters is not None else 'unknown'}"
# #                 )

# #             # === Pull λ and ρ from the base optimizer ===
# #             base_lambda = getattr(optimizer, "lambda_val", None)
# #             if base_lambda is not None:
# #                 self.lambda_param = float(base_lambda)

# #             base_rho = None
# #             for attr in ("rho", "sensitivity", "sensitivity_rho", "query_sensitivity"):
# #                 if hasattr(optimizer, attr):
# #                     try:
# #                         base_rho = float(getattr(optimizer, attr))
# #                         break
# #                     except Exception:
# #                         pass
# #             if base_rho is not None:
# #                 self.rho = base_rho

# #                         # If μ not set yet, compute it from theorem using ρ from base and γ=η
# #             if self.mu is None:
# #                 if self.rho is None:
# #                     self.rho = 1.0
# #                 self.mu = compute_mu_from_theorem(
# #                     rho=self.rho,
# #                     k=self.num_queries,
# #                     epsilon_sample=self.epsilon_sample,
# #                     delta_sample=self.delta,
# #                     gamma_margin=self.eta,
# #                     T=self.T_theorem,    # NEW: theorem horizon, not T_cap
# #                     C=self.mu_constant,
# #                     exponent_gamma=self.mu_gamma_exponent,
# #                 )
# #                 if verbose:
# #                     print(
# #                         f"λ (from base)={self.lambda_param:.6f}, "
# #                         f"ρ={self.rho:.6f}, μ={self.mu:.6f} "
# #                         f"(using T_theorem={self.T_theorem})"
# #                     )



# #             lam = float(self.lambda_param)
# #             mu = float(self.mu)
# #             # thr_current = lam + mu
# #             thr_current = 0.5*lam
# #             if (self.stop_threshold_abs is not None) and (self.stop_threshold_abs < thr_current):
# #                 thr_current = self.stop_threshold_abs

# #             # Compute query answers for this synopsis
# #             synopsis_answers = np.zeros(self.num_queries, dtype=float)
# #             for q in range(self.num_queries):
# #                 synopsis_answers[q] = self.compute_query_answer(optimizer.fake_X, q)

# #             # Per-query errors and λ, λ+μ-accuracy indicators
# #             errors = np.abs(synopsis_answers - true_answers)
# #             lambda_accurate = (errors <= lam)
# #             lambda_mu_accurate = (errors <= thr_current)

# #             cumulative_accuracy += lambda_mu_accurate.astype(float)
# #             iterations_run = t

# #             # Record this synopsis
# #             self.all_synopses.append(
# #                 {
# #                     "iteration": t,
# #                     "answers": synopsis_answers.copy(),
# #                     "errors": errors.copy(),
# #                     "lambda_accurate": lambda_accurate.copy(),
# #                     "lambda_mu_accurate": lambda_mu_accurate.copy(),
# #                     "base_iterations": base_iters,
# #                 }
# #             )

# #             # Early stopping check using median-of-answers
# #             if t >= self.min_rounds:
# #                 all_answers = np.zeros((t, self.num_queries), dtype=float)
# #                 for i, synopsis in enumerate(self.all_synopses):
# #                     all_answers[i, :] = synopsis["answers"]

# #                 per_query_medians = np.median(all_answers, axis=0)
# #                 median_errors = np.abs(per_query_medians - true_answers)
# #                 median_lambda_mu_accurate = (median_errors <= thr_current)
# #                 frac_good = float(np.mean(median_lambda_mu_accurate))

# #                 if verbose:
# #                     print(
# #                         f"[check] median-of-answers: max_err={np.max(median_errors):.6f} "
# #                         f"med={np.median(median_errors):.6f} mean={np.mean(median_errors):.6f} "
# #                         f"frac(|err|<=thr)={frac_good:.3f} "
# #                         f"(thr={thr_current:.6f}, λ={lam:.6f}, μ={mu:.6f})"
# #                     )

# #                 # Early stop condition: all queries λ+μ-accurate (or up to threshold)
# #                 if np.all(median_lambda_mu_accurate):
# #                     early_stopped = True
# #                     if verbose:
# #                         print("Early stopping: all queries λ+μ-accurate under median-of-answers.")
# #                     break

# #             # Update weights D via AdaBoost-style rule
# #             for i in range(self.num_queries):
# #                 if lambda_mu_accurate[i]:
# #                     u[i] *= math.exp(-self.alpha)
# #                 else:
# #                     u[i] *= math.exp(self.alpha)
# #             u_sum = np.sum(u)
# #             if u_sum <= 0:
# #                 if verbose:
# #                     print("Warning: weight vector u became non-positive; resetting to uniform.")
# #                 u = np.ones(self.num_queries, dtype=float)
# #                 u_sum = float(self.num_queries)
# #             D = u / u_sum

# #             if verbose:
# #                 frac_lambda_mu = float(np.mean(lambda_mu_accurate))
# #                 print(
# #                     f"Round {t}: λ+μ-accurate fraction={frac_lambda_mu:.3f} "
# #                     f"(thr={thr_current:.6f}, λ={lam:.6f}, μ={mu:.6f})"
# #                 )

# #         # Final median-of-answers summary
# #         all_answers = np.zeros((iterations_run, self.num_queries), dtype=float)
# #         for i, synopsis in enumerate(self.all_synopses):
# #             all_answers[i, :] = synopsis["answers"]

# #         per_query_medians = np.median(all_answers, axis=0)
# #         final_errors = np.abs(per_query_medians - true_answers)
# #         final_thr = lam + mu
# #         if (self.stop_threshold_abs is not None) and (self.stop_threshold_abs < final_thr):
# #             final_thr = self.stop_threshold_abs

# #         if verbose:
# #             print(
# #                 f"[final] median-of-answers: max={np.max(final_errors):.6f}  "
# #                 f"med={np.median(final_errors):.6f}  mean={np.mean(final_errors):.6f}  "
# #                 f"frac(|err|<=thr)={np.mean(final_errors <= final_thr):.3f}  "
# #                 f"(thr={final_thr:.6f}, λ={self.lambda_param:.6f}, μ={self.mu:.6f})"
# #             )
# #             print()

# #         return {
# #             "final_answers": {int(q): float(per_query_medians[q]) for q in range(self.num_queries)},
# #             "all_synopses": self.all_synopses,
# #             "final_distribution": D,
# #             "iterations_run": iterations_run,
# #             "early_stopped": early_stopped,
# #         }

# #!/usr/bin/env python3
# """
# Boosting algorithm for Spiky Nonconvex Queries.

# - On each round t, train a base synopsis A_t.
# - The final answer for a query q is the median over i=1..T of A_i(q).
# - Early stopping is based on the per-query median-of-answers condition.
# - λ and ρ are read from the base optimizer (spiky_nonconvex_newton_gd).
# - μ is computed from the boosting theorem using the SAME ρ and your η as edge γ.
# """

# import math
# from typing import Optional, Dict, List

# import numpy as np

# from spiky_nonconvex_newton_gd import SpikyNonconvexCoordinateDescent


# def _safe_log_one_over(x: float, eps: float = 1e-12) -> float:
#     """Stable log(1/x)."""
#     return math.log(1.0 / max(x, eps))


# def compute_mu_from_theorem(
#     rho: float,
#     k: int,
#     epsilon_sample: float,   # ε_sample
#     delta_sample: float,     # δ_sample
#     gamma_margin: float,     # η (edge)
#     T: int,                  # boosting horizon
#     *,
#     C: float = 1.0,
# ) -> float:
#     """
#     Compute μ using the theorem (your fixed version from the screenshot):

#         μ = C * [ sqrt(2 k ln((1+2η)/(1-2η))) * T^{3/2} * ρ *
#                   ( sqrt(log(1/δ_sample)) + sqrt(log(1/δ_sample) + 2 ε_sample) )
#                 ] / ε_sample
#     """

#     if epsilon_sample <= 0:
#         raise ValueError("epsilon_sample must be positive.")
#     if not (0.0 < delta_sample < 1.0):
#         raise ValueError("delta_sample must lie in (0,1).")
#     if not (0.0 < gamma_margin < 0.5):
#         raise ValueError("gamma_margin η must lie in (0, 0.5) for ln((1+2η)/(1-2η)) to be well-defined.")
#     if T <= 0:
#         raise ValueError("T must be positive.")

#     log_eta = math.log((1.0 + 2.0 * gamma_margin) / (1.0 - 2.0 * gamma_margin))
#     log_one_over_delta = _safe_log_one_over(delta_sample)

#     inner = math.sqrt(log_one_over_delta) + math.sqrt(log_one_over_delta + 2.0 * epsilon_sample)
#     prefactor = math.sqrt(2.0 * k * log_eta) * (T ** 1.5) * rho

#     mu = C * prefactor * inner / float(epsilon_sample)
#     return float(mu)


# class SpikyBoostingAlgorithm:
#     def __init__(
#         self,
#         k: int,
#         lambda_param: Optional[float],   # may be None → overwritten by base optimizer
#         eta: float,                      # treated as the weak-learning edge γ
#         rho: Optional[float],            # may be None → taken from base optimizer
#         mu: Optional[float],             # may be None → computed from theorem
#         T: int,
#         epsilon_sample: float,
#         delta: float,
#         beta: float,
#         n: int,
#         upper_bound: float = math.pi,
#         lower_bound: float = -math.pi,
#         tau: float = 1e-2,
#         *,
#         mu_constant: float = 1.0,        # hidden constant C in μ
#         min_rounds: int = 1,
#         stop_threshold_abs: Optional[float] = None,
#         max_base_restarts: int = 1000,   # max re-tries per boosting round
#         threshold_scale_lambda: float = 1.0,  # thr = threshold_scale_lambda * λ
#     ):
#         self.k = int(k)
#         self.lambda_param = None if lambda_param is None else float(lambda_param)
#         self.eta = float(eta)            # edge γ
#         self.rho = None if rho is None else float(rho)
#         self.mu = None if mu is None else float(mu)
#         self.T = int(T)
#         self.epsilon_sample = float(epsilon_sample)
#         self.delta = float(delta)
#         self.beta = float(beta)
#         self.n = int(n)
#         self.upper_bound = float(upper_bound)
#         self.lower_bound = float(lower_bound)
#         self.tau = float(tau)
#         self.max_base_restarts = int(max_base_restarts)

#         self.mu_constant = float(mu_constant)
#         self.min_rounds = int(min_rounds)
#         self.stop_threshold_abs = None if stop_threshold_abs is None else float(stop_threshold_abs)
#         self.threshold_scale_lambda = float(threshold_scale_lambda)

#         # AdaBoost-style α using 'eta' as advantage parameter
#         self.alpha = 0.5 * math.log((1.0 + 2.0 * self.eta) / (1.0 - 2.0 * self.eta))

#         self.Q: Optional[Dict[str, np.ndarray]] = None
#         self.num_queries: Optional[int] = None
#         self.all_synopses: List[Dict] = []

#     # ---------- helper for threshold ----------

#     def _compute_threshold(self, lam: float, mu: float) -> float:
#         """
#         Unified place to define the empirical stopping threshold.

#         Currently: thr = threshold_scale_lambda * λ, then optionally clipped
#         by an absolute stop_threshold_abs if provided.
#         """
#         thr = self.threshold_scale_lambda * float(lam)
#         if self.stop_threshold_abs is not None:
#             thr = min(thr, self.stop_threshold_abs)
#         return float(thr)

#     # ---------- query plumbing ----------

#     def set_queries(self, amplitudes_matrix: np.ndarray, frequencies_vector: np.ndarray):
#         A = np.asarray(amplitudes_matrix, dtype=float)
#         W = np.asarray(frequencies_vector, dtype=float)
#         if A.shape[0] != W.shape[0]:
#             raise ValueError("Number of amplitude vectors must match number of frequencies.")
#         if A.shape[1] != self.n:
#             raise ValueError(f"Amplitude vector length {A.shape[1]} must equal n {self.n}.")
#         self.Q = {"amplitudes": A, "frequencies": W}
#         self.num_queries = int(W.shape[0])

#     def compute_query_answer(self, synopsis_data: np.ndarray, query_idx: int) -> float:
#         amps = self.Q["amplitudes"][query_idx]
#         w = self.Q["frequencies"][query_idx]
#         return float(np.sum(synopsis_data ** 2 + amps * np.sin(w * synopsis_data)) / self.n)

#     # ---------- boosting loop ----------

#     def run_boosting(self, real_data: np.ndarray, verbose: bool = True) -> dict:
#         if self.Q is None:
#             raise ValueError("Must set queries first via set_queries().")

#         D = np.ones(self.num_queries, dtype=float) / float(self.num_queries)
#         u = np.ones(self.num_queries, dtype=float)

#         if verbose:
#             print("=== Boosting Algorithm ===")

#         self.all_synopses.clear()
#         early_stopped = False
#         iterations_run = 0

#         # ground truth answers (fixed)
#         true_answers = np.zeros(self.num_queries, dtype=float)
#         for q in range(self.num_queries):
#             amps = self.Q["amplitudes"][q]
#             w = self.Q["frequencies"][q]
#             true_answers[q] = float(
#                 np.sum(real_data ** 2 + amps * np.sin(w * real_data)) / self.n
#             )

#         for t in range(1, self.T + 1):
#             if verbose:
#                 print(f"--- Iteration {t}/{self.T} ---")

#             # === Use ALL queries; pass D as weights to the base optimizer ===
#             sampled_indices = np.arange(self.num_queries, dtype=int)
#             sampled_amplitudes = self.Q["amplitudes"][sampled_indices]
#             sampled_frequencies = self.Q["frequencies"][sampled_indices]
#             sampled_weights = D.astype(float).tolist()

#             # === Run base optimizer, with restarts until satisfaction ≥ 0.5 + eta ===
#             target_ratio = 0.5 + self.eta
#             best_optimizer = None
#             best_ret = None
#             best_satisfaction = -1.0
#             base_iters = None

#             for attempt in range(1, self.max_base_restarts + 1):
#                 if verbose:
#                     print(f"Running base synopsis generator (attempt {attempt}/{self.max_base_restarts})...")

#                 optimizer = SpikyNonconvexCoordinateDescent(
#                     epsilon=self.epsilon_sample,
#                     delta=self.delta,
#                     beta=self.beta,
#                     eta=self.eta,  # same η inside base optimizer
#                     n=self.n,
#                     tau=self.tau,
#                     upper_bound=self.upper_bound,
#                     lower_bound=self.lower_bound,
#                 )
#                 optimizer.set_queries_and_amplitudes(
#                     sampled_amplitudes, sampled_frequencies, sampled_weights
#                 )
#                 # keep real_data fixed
#                 optimizer.generate_data(real_data=real_data)

#                 ret = optimizer.run_coordinate_descent(max_iterations=1000, verbose=False)

#                 sat = float(optimizer.compute_weighted_satisfaction_ratio())
#                 if sat > best_satisfaction:
#                     best_satisfaction = sat
#                     best_optimizer = optimizer
#                     best_ret = ret

#                 base_iters_local = None
#                 if isinstance(ret, dict):
#                     for key in ("num_iterations", "iterations", "iters", "t", "steps", "n_iter"):
#                         if key in ret:
#                             try:
#                                 base_iters_local = int(ret[key])
#                                 break
#                             except Exception:
#                                 pass
#                 if base_iters_local is None:
#                     for attr in (
#                         "num_iterations", "iterations_run", "last_num_iterations",
#                         "iters_run", "n_iter_", "n_iter"
#                     ):
#                         if hasattr(optimizer, attr):
#                             try:
#                                 base_iters_local = int(getattr(optimizer, attr))
#                                 break
#                             except Exception:
#                                 pass

#                 if verbose:
#                     print(
#                         f"  attempt {attempt}: satisfaction={sat:.4f} "
#                         f"(target={target_ratio:.4f}), "
#                         f"iters={base_iters_local if base_iters_local is not None else 'unknown'}"
#                     )

#                 if sat >= target_ratio:
#                     base_iters = base_iters_local
#                     break

#             if base_iters is None:
#                 if isinstance(best_ret, dict):
#                     for key in ("num_iterations", "iterations", "iters", "t", "steps", "n_iter"):
#                         if key in best_ret:
#                             try:
#                                 base_iters = int(best_ret[key])
#                                 break
#                             except Exception:
#                                 pass
#                 if verbose:
#                     print(
#                         f"WARNING: base synopsis never reached satisfaction ≥ {target_ratio:.4f} "
#                         f"after {self.max_base_restarts} attempts. "
#                         f"Using best attempt with satisfaction={best_satisfaction:.4f}."
#                     )

#             optimizer = best_optimizer
#             ret = best_ret

#             if verbose:
#                 print(
#                     f"Base synopsis generator iterations (final accepted attempt): "
#                     f"{base_iters if base_iters is not None else 'unknown'}"
#                 )

#             # === Pull λ and ρ from the base optimizer ===
#             base_lambda = getattr(optimizer, "lambda_val", None)
#             if base_lambda is not None:
#                 self.lambda_param = float(base_lambda)

#             base_rho = None
#             for attr in ("rho", "sensitivity", "sensitivity_rho", "query_sensitivity"):
#                 if hasattr(optimizer, attr):
#                     try:
#                         base_rho = float(getattr(optimizer, attr))
#                         break
#                     except Exception:
#                         pass
#             if base_rho is not None:
#                 self.rho = base_rho

#             # If μ not set yet, compute it from theorem using ρ from base and γ=η
#             if self.mu is None:
#                 if self.rho is None:
#                     self.rho = 1.0
#                 self.mu = compute_mu_from_theorem(
#                     rho=self.rho,
#                     k=self.num_queries,
#                     epsilon_sample=self.epsilon_sample,
#                     delta_sample=self.delta,
#                     gamma_margin=self.eta,
#                     T=self.T,                  # horizon used in theorem
#                     C=self.mu_constant,
#                 )
#                 if verbose:
#                     print(
#                         f"λ (from base)={self.lambda_param:.6f}, "
#                         f"ρ={self.rho:.6f}, μ(theorem)={self.mu:.6f}"
#                     )

#             lam = float(self.lambda_param)
#             mu = float(self.mu)
#             thr_current = self._compute_threshold(lam, mu)

#             # Compute query answers for this synopsis
#             synopsis_answers = np.zeros(self.num_queries, dtype=float)
#             for q in range(self.num_queries):
#                 synopsis_answers[q] = self.compute_query_answer(optimizer.fake_X, q)

#             # Per-query errors vs truth for THIS synopsis
#             errors_t = np.abs(synopsis_answers - true_answers)
#             good_t = (errors_t <= thr_current)

#             iterations_run = t

#             self.all_synopses.append(
#                 {
#                     "iteration": t,
#                     "answers": synopsis_answers.copy(),
#                     "errors": errors_t.copy(),
#                     "good_under_threshold": good_t.copy(),
#                     "base_iterations": base_iters,
#                 }
#             )

#             # ---- Early stop via per-query median over A_1..A_t ----
#             all_answers = np.zeros((t, self.num_queries), dtype=float)
#             for i, synopsis in enumerate(self.all_synopses):
#                 all_answers[i, :] = synopsis["answers"]

#             per_query_medians = np.median(all_answers, axis=0)
#             median_errors = np.abs(per_query_medians - true_answers)
#             median_good = (median_errors <= thr_current)
#             frac_good = float(np.mean(median_good))

#             if verbose:
#                 print(
#                     f"[check] median-of-answers: max_err={np.max(median_errors):.6f} "
#                     f"med={np.median(median_errors):.6f} mean={np.mean(median_errors):.6f} "
#                     f"frac(|err|<=thr)={frac_good:.3f} "
#                     f"(thr={thr_current:.6f}, λ={lam:.6f}, μ={mu:.6f})"
#                 )

#             if (t >= self.min_rounds) and np.all(median_good):
#                 early_stopped = True
#                 if verbose:
#                     print("Early stopping: all queries within threshold under median-of-answers.")
#                 break

#             # ---- Update weights D via AdaBoost-style rule using THIS round's errors ----
#             for i in range(self.num_queries):
#                 if good_t[i]:
#                     u[i] *= math.exp(-self.alpha)
#                 else:
#                     u[i] *= math.exp(self.alpha)
#             u_sum = np.sum(u)
#             if u_sum <= 0:
#                 if verbose:
#                     print("Warning: weights became non-positive; resetting to uniform.")
#                 u = np.ones(self.num_queries, dtype=float)
#                 u_sum = float(self.num_queries)
#             D = u / u_sum

#             if verbose:
#                 frac_good_t = float(np.mean(good_t))
#                 print(
#                     f"Round {t}: per-round good-fraction={frac_good_t:.3f} "
#                     f"(thr={thr_current:.6f}, λ={lam:.6f}, μ={mu:.6f})"
#                 )

#         # Final median-of-answers summary (over all iterations_run rounds)
#         all_answers = np.zeros((iterations_run, self.num_queries), dtype=float)
#         for i, synopsis in enumerate(self.all_synopses):
#             all_answers[i, :] = synopsis["answers"]

#         per_query_medians = np.median(all_answers, axis=0)
#         final_errors = np.abs(per_query_medians - true_answers)

#         lam = float(self.lambda_param)
#         mu = float(self.mu)
#         final_thr = self._compute_threshold(lam, mu)

#         if verbose:
#             print(
#                 f"[final] median-of-answers: max={np.max(final_errors):.6f}  "
#                 f"med={np.median(final_errors):.6f}  mean={np.mean(final_errors):.6f}  "
#                 f"frac(|err|<=thr)={np.mean(final_errors <= final_thr):.3f}  "
#                 f"(thr={final_thr:.6f}, λ={lam:.6f}, μ={mu:.6f})"
#             )
#             print()

#         # Unified return: everything based on the final MEDIAN ensemble
#         return {
#             "final_answers": {int(q): float(per_query_medians[q]) for q in range(self.num_queries)},
#             "final_errors": [float(e) for e in final_errors],
#             "threshold": float(final_thr),
#             "all_synopses": self.all_synopses,
#             "final_distribution": D,
#             "iterations_run": iterations_run,
#             "early_stopped": early_stopped,
#             "lambda": lam,
#             "mu": mu,
#         }

#!/usr/bin/env python3
"""
Boosting algorithm for Spiky Nonconvex Queries.

Implements query boosting where, on each round t, we train a base
synopsis A_t and keep all synopses. The output for a query q is the
median over i=1..T of A_i(q). Early stopping checks that condition
after each round.

Key points:
- Uses the ENTIRE query set Q each round (no subsampling).
- λ and ρ are read from the base optimizer (spiky_nonconvex_newton_gd).
- μ is computed from the boosting theorem using the SAME ρ, and with
  the edge γ identified as your config η.
- We capture and report how many iterations the base optimizer ran.
"""

import math
from typing import Optional, Dict, List

import numpy as np

from spiky_nonconvex_newton_gd import SpikyNonconvexCoordinateDescent


def _safe_log_one_over(x: float, eps: float = 1e-12) -> float:
    """Stable log(1/x)."""
    return math.log(1.0 / max(x, eps))


def compute_mu_from_theorem(
    rho: float,
    k: int,
    epsilon_sample: float,   # ε_sample
    delta_sample: float,     # δ_sample
    gamma_margin: float,     # η (edge)
    T: int,                  # boosting horizon *used in theorem*
    *,
    C: float = 1.0,
    exponent_gamma: int = 3,  # kept for backwards compatibility; not used now
) -> float:
    """
    Compute μ using the theorem:

        μ = C * [ sqrt(2 k ln((1+2η)/(1-2η))) * T^{3/2} * ρ *
                   ( sqrt(log(1/δ_sample)) + sqrt(log(1/δ_sample) + 2 ε_sample) )
                ] / ε_sample

    Parameters
    ----------
    rho          : ρ in the theorem (sensitivity of base mechanism)
    k            : |Q|
    epsilon_sample : ε_sample
    delta_sample : δ_sample
    gamma_margin : η (edge)
    T            : boosting horizon (number of rounds used in theorem)
    C            : hidden constant to tune μ up/down
    """

    if epsilon_sample <= 0:
        raise ValueError("epsilon_sample must be positive.")
    if not (0.0 < delta_sample < 1.0):
        raise ValueError("delta_sample must lie in (0,1).")
    if not (0.0 < gamma_margin < 0.5):
        raise ValueError("gamma_margin η must lie in (0, 0.5) for ln((1+2η)/(1-2η)) to be well-defined.")
    if T <= 0:
        raise ValueError("T must be positive.")

    # ln((1+2η)/(1-2η)) (outside the sqrt, as in your corrected version)
    log_eta = math.log((1.0 + 2.0 * gamma_margin) / (1.0 - 2.0 * gamma_margin))

    # sqrt(log(1/δ_sample)) and sqrt(log(1/δ_sample) + 2 ε_sample)
    log_one_over_delta = _safe_log_one_over(delta_sample)
    inner = math.sqrt(log_one_over_delta) + math.sqrt(log_one_over_delta + 2.0 * epsilon_sample)

    # sqrt(2 k ln((1+2η)/(1-2η))) * T^{3/2} * ρ
    prefactor = math.sqrt(2.0 * k * log_eta) * (T ** 1.5) * rho

    mu = C * prefactor * inner / float(epsilon_sample)
    return float(mu)


class SpikyBoostingAlgorithm:
    def __init__(
        self,
        k: int,
        lambda_param: Optional[float],   # may be None → will be overwritten by base optimizer
        eta: float,                      # treated as the weak-learning edge γ
        rho: Optional[float],            # may be None → taken from base optimizer if available
        mu: Optional[float],             # may be None → computed from theorem using base ρ and η as γ
        T: int,
        epsilon_sample: float,
        delta: float,
        beta: float,
        n: int,
        upper_bound: float = math.pi,
        lower_bound: float = -math.pi,
        tau: float = 1e-2,
        *,
        mu_constant: float = 1.0,        # hidden constant C in μ
        mu_gamma_exponent: int = 3,      # kept for compatibility; not currently used
        min_rounds: int = 1,
        stop_threshold_abs: Optional[float] = None,
        max_base_restarts: int = 1000,   # max re-tries per boosting round
        theorem_T: int = 5,              # T used in the theorem for μ, separate from T_cap
    ):
        self.k = int(k)
        self.lambda_param = None if lambda_param is None else float(lambda_param)
        self.eta = float(eta)            # edge γ
        self.rho = None if rho is None else float(rho)
        self.mu = None if mu is None else float(mu)
        self.T = int(T)
        self.epsilon_sample = float(epsilon_sample)
        self.delta = float(delta)
        self.beta = float(beta)
        self.n = int(n)
        self.upper_bound = float(upper_bound)
        self.lower_bound = float(lower_bound)
        self.tau = float(tau)
        self.max_base_restarts = int(max_base_restarts)

        self.mu_constant = float(mu_constant)
        self.mu_gamma_exponent = int(mu_gamma_exponent)
        self.min_rounds = int(min_rounds)
        self.stop_threshold_abs = None if stop_threshold_abs is None else float(stop_threshold_abs)

        # NEW: T used in μ theorem (clamped internally to ≤ self.T)
        self.theorem_T = int(theorem_T)

        # AdaBoost-style α using 'eta' as advantage parameter
        self.alpha = 0.5 * math.log((1.0 + 2.0 * self.eta) / (1.0 - 2.0 * self.eta))

        self.Q: Optional[Dict[str, np.ndarray]] = None
        self.num_queries: Optional[int] = None
        self.all_synopses: List[Dict] = []

    # ---------- query plumbing ----------

    def set_queries(self, amplitudes_matrix: np.ndarray, frequencies_vector: np.ndarray):
        A = np.asarray(amplitudes_matrix, dtype=float)
        W = np.asarray(frequencies_vector, dtype=float)
        if A.shape[0] != W.shape[0]:
            raise ValueError("Number of amplitude vectors must match number of frequencies.")
        if A.shape[1] != self.n:
            raise ValueError(f"Amplitude vector length {A.shape[1]} must equal n {self.n}.")
        self.Q = {"amplitudes": A, "frequencies": W}
        self.num_queries = int(W.shape[0])

    def compute_query_answer(self, synopsis_data: np.ndarray, query_idx: int) -> float:
        amps = self.Q["amplitudes"][query_idx]
        w = self.Q["frequencies"][query_idx]
        return float(np.sum(synopsis_data ** 2 + amps * np.sin(w * synopsis_data)) / self.n)

    def is_lambda_accurate(self, error: float, lambda_param: float) -> bool:
        return error <= lambda_param

    def is_lambda_mu_accurate(self, error: float, lambda_param: float, mu: float) -> bool:
        return error <= (lambda_param + mu)

    # ---------- boosting loop ----------

    def run_boosting(self, real_data: np.ndarray, verbose: bool = True) -> dict:
        if self.Q is None:
            raise ValueError("Must set queries first via set_queries().")

        D = np.ones(self.num_queries, dtype=float) / float(self.num_queries)
        u = np.ones(self.num_queries, dtype=float)

        if verbose:
            print("=== Boosting Algorithm ===")

        self.all_synopses.clear()
        cumulative_accuracy = np.zeros(self.num_queries, dtype=float)
        early_stopped = False
        iterations_run = 0

        # ground truth answers (fixed, based on the real_data passed in)
        true_answers = np.zeros(self.num_queries, dtype=float)
        for q in range(self.num_queries):
            amps = self.Q["amplitudes"][q]
            w = self.Q["frequencies"][q]
            true_answers[q] = float(np.sum(real_data ** 2 + amps * np.sin(w * real_data)) / self.n)

        for t in range(1, self.T + 1):
            if verbose:
                print(f"--- Iteration {t}/{self.T} ---")

            # === Use ALL queries; pass D as weights to the base optimizer ===
            sampled_indices = np.arange(self.num_queries, dtype=int)
            sampled_amplitudes = self.Q["amplitudes"][sampled_indices]
            sampled_frequencies = self.Q["frequencies"][sampled_indices]
            sampled_weights = D.astype(float).tolist()

            # === Run base optimizer, with restarts until satisfaction ≥ 0.5 + eta ===
            target_ratio = 0.5 + self.eta
            best_optimizer = None
            best_ret = None
            best_satisfaction = -1.0
            base_iters = None

            for attempt in range(1, self.max_base_restarts + 1):
                if verbose:
                    print(f"Running base synopsis generator (attempt {attempt}/{self.max_base_restarts})...")

                optimizer = SpikyNonconvexCoordinateDescent(
                    epsilon=self.epsilon_sample,
                    delta=self.delta,
                    beta=self.beta,
                    eta=self.eta,  # your 'eta' is used inside base optimizer as well
                    n=self.n,
                    tau=self.tau,
                    upper_bound=self.upper_bound,
                    lower_bound=self.lower_bound,
                )
                optimizer.set_queries_and_amplitudes(sampled_amplitudes, sampled_frequencies, sampled_weights)

                # IMPORTANT: keep the same real_data across all attempts and rounds
                optimizer.generate_data(real_data=real_data)

                # Run base optimizer and capture how many iterations it actually ran
                ret = optimizer.run_coordinate_descent(max_iterations=1000, verbose=False)

                # Satisfaction ratio after this attempt
                sat = float(optimizer.compute_weighted_satisfaction_ratio())
                if sat > best_satisfaction:
                    best_satisfaction = sat
                    best_optimizer = optimizer
                    best_ret = ret

                # Extract iteration count
                base_iters_local = None
                if isinstance(ret, dict):
                    for key in ("num_iterations", "iterations", "iters", "t", "steps", "n_iter"):
                        if key in ret:
                            try:
                                base_iters_local = int(ret[key])
                                break
                            except Exception:
                                pass
                if base_iters_local is None:
                    for attr in ("num_iterations", "iterations_run", "last_num_iterations",
                                 "iters_run", "n_iter_", "n_iter"):
                        if hasattr(optimizer, attr):
                            try:
                                base_iters_local = int(getattr(optimizer, attr))
                                break
                            except Exception:
                                pass

                if verbose:
                    print(
                        f"  attempt {attempt}: satisfaction={sat:.4f} "
                        f"(target={target_ratio:.4f}), "
                        f"iters={base_iters_local if base_iters_local is not None else 'unknown'}"
                    )

                # If this attempt is "truly satisfactory", accept and stop restarting
                if sat >= target_ratio:
                    base_iters = base_iters_local
                    break

            # If never hit target_ratio, fall back to the best attempt
            if base_iters is None:
                if isinstance(best_ret, dict):
                    for key in ("num_iterations", "iterations", "iters", "t", "steps", "n_iter"):
                        if key in best_ret:
                            try:
                                base_iters = int(best_ret[key])
                                break
                            except Exception:
                                pass
                if verbose:
                    print(
                        f"WARNING: base synopsis never reached satisfaction ≥ {target_ratio:.4f} "
                        f"after {self.max_base_restarts} attempts. "
                        f"Using best attempt with satisfaction={best_satisfaction:.4f}."
                    )

            optimizer = best_optimizer
            ret = best_ret

            if verbose:
                print(
                    f"Base synopsis generator iterations (final accepted attempt): "
                    f"{base_iters if base_iters is not None else 'unknown'}"
                )

            # === Pull λ and ρ from the base optimizer ===
            base_lambda = getattr(optimizer, "lambda_val", None)
            if base_lambda is not None:
                self.lambda_param = float(base_lambda)

            base_rho = None
            for attr in ("rho", "sensitivity", "sensitivity_rho", "query_sensitivity"):
                if hasattr(optimizer, attr):
                    try:
                        base_rho = float(getattr(optimizer, attr))
                        break
                    except Exception:
                        pass
            if base_rho is not None:
                self.rho = base_rho

            # If μ not set yet, compute it from theorem using ρ from base and γ=η
            if self.mu is None:
                if self.rho is None:
                    self.rho = 1.0

                # Clamp theorem T to min(self.T, self.theorem_T)
                T_theorem = min(self.T, self.theorem_T)

                self.mu = compute_mu_from_theorem(
                    rho=self.rho,
                    k=self.num_queries,
                    epsilon_sample=self.epsilon_sample,  # ε_sample
                    delta_sample=self.delta,             # δ_sample
                    gamma_margin=self.eta,               # η
                    T=T_theorem,                         # theorem horizon, not T_cap
                    C=self.mu_constant,
                    exponent_gamma=self.mu_gamma_exponent,
                )
                if verbose:
                    print(
                        f"λ (from base)={self.lambda_param:.6f}, ρ={self.rho:.6f}, "
                        f"μ(theorem)={self.mu:.6f} (using T_theorem={T_theorem})"
                    )

            lam = float(self.lambda_param)
            mu = float(self.mu)
            # thr_current = lam + mu
            thr_current = 0.5*lam
            if (self.stop_threshold_abs is not None) and (self.stop_threshold_abs < thr_current):
                thr_current = self.stop_threshold_abs

            # Compute query answers for this synopsis
            synopsis_answers = np.zeros(self.num_queries, dtype=float)
            for q in range(self.num_queries):
                synopsis_answers[q] = self.compute_query_answer(optimizer.fake_X, q)

            # Per-query errors and λ, λ+μ-accuracy indicators
            errors = np.abs(synopsis_answers - true_answers)
            lambda_accurate = (errors <= lam)
            lambda_mu_accurate = (errors <= thr_current)

            cumulative_accuracy += lambda_mu_accurate.astype(float)
            iterations_run = t

            # Record this synopsis
            self.all_synopses.append(
                {
                    "iteration": t,
                    "answers": synopsis_answers.copy(),
                    "errors": errors.copy(),
                    "lambda_accurate": lambda_accurate.copy(),
                    "lambda_mu_accurate": lambda_mu_accurate.copy(),
                    "base_iterations": base_iters,
                }
            )

            # Early stopping check using median-of-answers across all synopses so far
            if t >= self.min_rounds:
                all_answers = np.zeros((t, self.num_queries), dtype=float)
                for i, synopsis in enumerate(self.all_synopses):
                    all_answers[i, :] = synopsis["answers"]

                per_query_medians = np.median(all_answers, axis=0)
                median_errors = np.abs(per_query_medians - true_answers)
                median_lambda_mu_accurate = (median_errors <= thr_current)
                frac_good = float(np.mean(median_lambda_mu_accurate))

                if verbose:
                    print(
                        f"[check] median-of-answers: max_err={np.max(median_errors):.6f} "
                        f"med={np.median(median_errors):.6f} mean={np.mean(median_errors):.6f} "
                        f"frac(|err|<=thr)={frac_good:.3f} "
                        f"(thr={thr_current:.6f}, λ={lam:.6f}, μ={mu:.6f})"
                    )

                # Early stop condition: all queries λ+μ-accurate (or up to threshold)
                if np.all(median_lambda_mu_accurate):
                    early_stopped = True
                    if verbose:
                        print("Early stopping: all queries λ+μ-accurate under median-of-answers.")
                    break

            # Update weights D via AdaBoost-style rule
            for i in range(self.num_queries):
                if lambda_mu_accurate[i]:
                    u[i] *= math.exp(-self.alpha)
                else:
                    u[i] *= math.exp(self.alpha)
            u_sum = np.sum(u)
            if u_sum <= 0:
                if verbose:
                    print("Warning: weight vector u became non-positive; resetting to uniform.")
                u = np.ones(self.num_queries, dtype=float)
                u_sum = float(self.num_queries)
            D = u / u_sum

            if verbose:
                frac_lambda_mu = float(np.mean(lambda_mu_accurate))
                print(
                    f"Round {t}: per-round good-fraction={frac_lambda_mu:.3f} "
                    f"(thr={thr_current:.6f}, λ={lam:.6f}, μ={mu:.6f})"
                )

        # Final median-of-answers summary across all synopses
        all_answers = np.zeros((iterations_run, self.num_queries), dtype=float)
        for i, synopsis in enumerate(self.all_synopses):
            all_answers[i, :] = synopsis["answers"]

        per_query_medians = np.median(all_answers, axis=0)
        final_errors = np.abs(per_query_medians - true_answers)
        lam = float(self.lambda_param)
        mu = float(self.mu)
        final_thr = lam + mu
        if (self.stop_threshold_abs is not None) and (self.stop_threshold_abs < final_thr):
            final_thr = self.stop_threshold_abs

        if verbose:
            print(
                f"[final] median-of-answers: max={np.max(final_errors):.6f}  "
                f"med={np.median(final_errors):.6f}  mean={np.mean(final_errors):.6f}  "
                f"frac(|err|<=thr)={np.mean(final_errors <= final_thr):.3f}  "
                f"(thr={final_thr:.6f}, λ={self.lambda_param:.6f}, μ={self.mu:.6f})"
            )
            print()

        return {
            "final_answers": {int(q): float(per_query_medians[q]) for q in range(self.num_queries)},
            "all_synopses": self.all_synopses,
            "final_distribution": D,
            "iterations_run": iterations_run,
            "early_stopped": early_stopped,
        }
