#!/usr/bin/env python3
"""
Boosting algorithm for Spiky Nonconvex Queries.

This implements the query boosting algorithm from the pseudocode screenshot,
using SpikyNonconvexCoordinateDescent as the base synopsis generator.
"""

import numpy as np
import math
from typing import List, Optional
from spiky_nonconvex import SpikyNonconvexCoordinateDescent


class SpikyBoostingAlgorithm:
    """
    Boosting algorithm for improving query accuracy through iterative refinement.
    
    Algorithm: Boosting for Queries
    - Iteratively generates synopses using base generator (coordinate descent)
    - Reweights queries based on their accuracy scores
    - Returns median of all synopsis answers for each query
    """
    
    def __init__(self,
                 k: int,
                 lambda_param: float,
                 eta: float,
                 rho: float,
                 mu: float,
                 T: int,
                 epsilon: float,
                 delta: float,
                 beta: float,
                 n: int,
                 upper_bound: float = math.pi,
                 lower_bound: float = -math.pi,
                 tau: float = 0.01):
        """
        Initialize the boosting algorithm.
        
        Args:
            k: Number of queries to sample in each iteration
            lambda_param: Accuracy threshold parameter
            eta: Edge parameter for boosting (typically small, e.g., 0.01)
            rho: Sensitivity parameter
            mu: Margin parameter for accuracy scoring
            T: Number of boosting iterations
            epsilon: DP parameter
            delta: DP parameter
            beta: Failure probability
            n: Number of data points
            upper_bound: Data upper bound
            lower_bound: Data lower bound
            tau: Stopping criterion for base generator
        """
        self.k = k
        self.lambda_param = lambda_param
        self.eta = eta
        self.rho = rho
        self.mu = mu
        self.T = T
        self.epsilon = epsilon
        self.delta = delta
        self.beta = beta
        self.n = n
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.tau = tau
        
        # Calculate alpha parameter for exponential weighting
        self.alpha = 0.5 * math.log((1 + 2 * eta) / (1 - 2 * eta))
        
        # Store all query results across iterations
        self.all_synopses = []  # List of dicts containing query answers from each iteration
        
        # Will be set when queries are provided
        self.Q = None  # Set of all queries (amplitudes and frequencies)
        self.query_names = None  # Names or indices for queries
        self.num_queries = None
        
    def set_queries(self, amplitudes_matrix: np.ndarray, frequencies_vector: np.ndarray):
        """
        Set the queries (amplitudes and frequencies) for boosting.
        
        Args:
            amplitudes_matrix: Matrix of amplitude vectors for each query, shape (num_queries, n)
            frequencies_vector: Vector of frequency parameters for each query, shape (num_queries,)
        """
        amplitudes_matrix = np.asarray(amplitudes_matrix, dtype=float)
        frequencies_vector = np.asarray(frequencies_vector, dtype=float)
        
        if amplitudes_matrix.shape[0] != len(frequencies_vector):
            raise ValueError("Number of amplitude vectors must match number of frequencies")
        
        if amplitudes_matrix.shape[1] != self.n:
            raise ValueError(f"Amplitude vector length {amplitudes_matrix.shape[1]} must equal n {self.n}")
        
        self.Q = {
            'amplitudes': amplitudes_matrix,
            'frequencies': frequencies_vector
        }
        self.num_queries = len(frequencies_vector)
        
    def compute_query_answer(self, synopsis_data: np.ndarray, query_idx: int) -> float:
        """
        Compute the answer to a specific query using synthetic data.
        
        Args:
            synopsis_data: Synthetic data array (fake_X)
            query_idx: Index of the query in the query set
            
        Returns:
            Query answer: (1/n) * sum_i (x_i^2 + a_i * sin(w_i*x_i))
        """
        amps = self.Q['amplitudes'][query_idx]
        freq = self.Q['frequencies'][query_idx]
        
        sum_result = np.sum(synopsis_data**2 + amps * np.sin(freq * synopsis_data))
        return sum_result / self.n
    
    def is_lambda_accurate(self, error: float, lambda_param: float) -> bool:
        """Check if error is within lambda accuracy threshold."""
        return error <= lambda_param
    
    def is_lambda_mu_inaccurate(self, error: float, lambda_param: float, mu: float) -> bool:
        """Check if error exceeds lambda + mu threshold."""
        return error > lambda_param + mu
    
    def compute_accuracy_score(self, error: float) -> float:
        """
        Compute accuracy score a_t,q for a query given its error.
        
        Args:
            error: Error d_q,t for this query at iteration t
            
        Returns:
            Accuracy score a_t,q
        """
        if self.is_lambda_accurate(error, self.lambda_param):
            return 1.0
        elif self.is_lambda_mu_inaccurate(error, self.lambda_param, self.mu):
            return -1.0
        else:
            # Linear interpolation between lambda and lambda + mu
            return 1.0 - 2.0 * (error - self.lambda_param) / self.mu
    
    def run_boosting(self, real_data: np.ndarray, verbose: bool = True) -> dict:
        """
        Run the boosting algorithm.
        
        Args:
            real_data: Real database x
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing all synopses and final boosted answers
        """
        if self.Q is None:
            raise ValueError("Must set queries first using set_queries()")
        
        # Initialize distribution D_1 (uniform over all queries)
        D = np.ones(self.num_queries) / self.num_queries
        u = np.ones(self.num_queries)  # Unnormalized weights
        
        if verbose:
            print(f"=== Boosting Algorithm ===")
            print(f"Parameters: k={self.k}, λ={self.lambda_param}, η={self.eta}, μ={self.mu}, T={self.T}")
            print(f"Number of queries in set Q: {self.num_queries}")
            print(f"Alpha (weight parameter): {self.alpha:.6f}")
            print()
        
        # Main boosting loop
        for t in range(1, self.T + 1):
            if verbose:
                print(f"--- Iteration {t}/{self.T} ---")
            
            # Sample k queries from distribution D_t
            sampled_indices = np.random.choice(
                self.num_queries, 
                size=min(self.k, self.num_queries), 
                replace=False, 
                p=D
            )
            
            if verbose:
                print(f"Sampled {len(sampled_indices)} queries from distribution D_{t}")
            
            # Create base synopsis generator with sampled queries
            optimizer = SpikyNonconvexCoordinateDescent(
                epsilon=self.epsilon,
                delta=self.delta,
                beta=self.beta,
                eta=self.eta,
                n=self.n,
                tau=self.tau,
                upper_bound=self.upper_bound,
                lower_bound=self.lower_bound
            )
            
            # Set sampled queries with uniform weights (since distribution is already in D)
            sampled_amplitudes = self.Q['amplitudes'][sampled_indices]
            sampled_frequencies = self.Q['frequencies'][sampled_indices]
            sampled_weights = [1.0 / len(sampled_indices)] * len(sampled_indices)
            
            optimizer.set_queries_and_amplitudes(sampled_amplitudes, sampled_frequencies, sampled_weights)
            optimizer.real_X = real_data.copy()
            optimizer.generate_data()
            
            # Run base synopsis generator
            if verbose:
                print(f"Running base synopsis generator...")
            
            results = optimizer.run_coordinate_descent(max_iterations=1000, verbose=False)
            
            # Get synopsis data
            synopsis_data = optimizer.fake_X  # This is A_t
            
            # Compute accuracy scores for ALL queries in Q
            synopsis_answers = {}
            accuracy_scores = np.zeros(self.num_queries)
            
            # Compute answers for ALL queries using the synopsis data
            for query_idx in range(self.num_queries):
                synopsis_answers[query_idx] = self.compute_query_answer(synopsis_data, query_idx)
            
            # Now compute errors and accuracy scores
            real_query_answers = {}
            for query_idx in range(self.num_queries):
                # Compute real query answer
                amps = self.Q['amplitudes'][query_idx]
                freq = self.Q['frequencies'][query_idx]
                real_sum = np.sum(real_data**2 + amps * np.sin(freq * real_data))
                real_query_answers[query_idx] = real_sum / self.n
                
                # Compute error
                error = abs(synopsis_answers[query_idx] - real_query_answers[query_idx])
                
                # Compute accuracy score
                accuracy_scores[query_idx] = self.compute_accuracy_score(error)
            
            # Store synopsis for this iteration
            self.all_synopses.append({
                'iteration': t,
                'sampled_indices': sampled_indices,
                'synopsis_data': synopsis_data.copy(),
                'synopsis_answers': synopsis_answers.copy(),
                'real_answers': real_query_answers.copy(),
                'accuracy_scores': accuracy_scores.copy()
            })
            
            # Update unnormalized weights: u_{t+1,q} = exp(-α * Σ_{j=1}^t a_{j,q})
            # Accumulate accuracy scores across iterations
            if t == 1:
                cumulative_accuracy = accuracy_scores
            else:
                cumulative_accuracy = cumulative_accuracy + accuracy_scores
            
            u = np.exp(-self.alpha * cumulative_accuracy)
            
            # Renormalize to get D_{t+1}
            Z = np.sum(u)
            D = u / Z
            
            if verbose:
                avg_accuracy = np.mean(accuracy_scores)
                print(f"Average accuracy score: {avg_accuracy:.4f}")
                print(f"Number of λ-accurate queries: {np.sum(accuracy_scores == 1.0)}")
                print(f"Number of (λ+μ)-inaccurate queries: {np.sum(accuracy_scores == -1.0)}")
                print(f"Distribution entropy: {-np.sum(D * np.log(D + 1e-10)):.4f}")
                print()
        
        # Compute final boosted answers using median
        if verbose:
            print("=== Computing Final Boosted Answers ===")
        
        final_boosted_answers = {}
        for query_idx in range(self.num_queries):
            # Get all answers for this query across all iterations
            query_answers = [self.all_synopses[t]['synopsis_answers'][query_idx] 
                            for t in range(self.T)]
            # Use median as the final boosted answer
            final_boosted_answers[query_idx] = np.median(query_answers)
        
        if verbose:
            print("Boosting complete!")
            print()
        
        return {
            'final_answers': final_boosted_answers,
            'all_synopses': self.all_synopses,
            'cumulative_accuracy': cumulative_accuracy,
            'final_distribution': D
        }
