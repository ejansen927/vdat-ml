#!/usr/bin/env python3
"""
Acquisition functions for active learning.

UCB:  α(x) = μ(x) + β·σ(x)         # Exploitation + exploration
UCB': α(x) = |∇μ(x)| + β·|∇σ(x)|  # Gradient-based (targets phase transitions)
"""

import torch


def ucb(surrogate, X, beta=2.0):
    """
    Upper Confidence Bound: α(x) = μ(x) + β·σ(x)
    
    High score = high predicted value OR high uncertainty.
    """
    mean, std = surrogate.predict(X)
    # Average across output dimensions
    return mean.mean(dim=-1) + beta * std.mean(dim=-1)


def ucb_prime(surrogate, X, beta=1.0):
    """
    Gradient-based UCB: α(x) = |∇μ(x)| + β·|∇σ(x)|
    
    High score = prediction changing rapidly (phase transitions).
    """
    grad_mean, grad_std = surrogate.gradients(X)
    # L2 norm of gradient vectors
    return grad_mean.norm(dim=-1) + beta * grad_std.norm(dim=-1)


def uncertainty(surrogate, X):
    """Pure uncertainty: α(x) = σ(x)"""
    _, std = surrogate.predict(X)
    return std.mean(dim=-1)


# Registry
ACQUISITION_FUNCTIONS = {
    "ucb": ucb,
    "ucb_prime": ucb_prime,
    "uncertainty": uncertainty,
}
