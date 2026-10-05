#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
r"""Standard (record-level) DP-SGD privacy accounting.

Used by the naive DP-SGD baseline (``main_NaiveDPSGD.py``); the node-level
accountant of the paper lives in :mod:`privacy.mix`.

The RDP part is based on Opacus, which is in turn based on Google's TF Privacy
(https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/rdp_accountant.py),
and computes the Renyi DP of the Sampled Gaussian Mechanism (SGM), see
https://arxiv.org/abs/1908.10530. Example::

    >>> orders = range(2, 33)
    >>> rdp = compute_rdp(q, sigma, steps, orders)
    >>> epsilon, opt_order = get_privacy_spent(orders, rdp, delta)
"""

import math
from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from scipy import special

STD_CACHE_FILE = Path(__file__).parent / 'stds.pt'


########################
# LOG-SPACE ARITHMETIC #
########################


def _log_add(logx: float, logy: float) -> float:
    r"""Returns ``log(exp(logx) + exp(logy))``."""
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b


def _log_sub(logx: float, logy: float) -> float:
    r"""Returns ``log(exp(logx) - exp(logy))``; requires ``logx >= logy``."""
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy
    except OverflowError:
        return logx


def _log_erfc(x: float) -> float:
    r"""Returns ``log(erfc(x))``, accurate also for large ``x``."""
    return math.log(2) + special.log_ndtr(-x * 2 ** 0.5)


##################
# RDP OF THE SGM #
##################


def _compute_log_a_for_int_alpha(q: float, sigma: float, alpha: int) -> float:
    r"""Computes :math:`log(A_\alpha)` for integer ``alpha`` (Sec. 3.3 of arXiv:1908.10530)."""
    # Initialize with 0 in the log space.
    log_a = -np.inf

    for i in range(alpha + 1):
        log_coef_i = (
            math.log(special.binom(alpha, i))
            + i * math.log(q)
            + (alpha - i) * math.log(1 - q)
        )

        s = log_coef_i + (i * i - i) / (2 * (sigma ** 2))
        log_a = _log_add(log_a, s)

    return float(log_a)


def _compute_log_a_for_frac_alpha(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for fractional ``alpha`` (Sec. 3.3 of arXiv:1908.10530)."""
    # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
    # initialized to 0 in the log space:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5

    while True:  # do ... until loop
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1)


def _compute_log_a(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for any positive finite ``alpha``."""
    if float(alpha).is_integer():
        return _compute_log_a_for_int_alpha(q, sigma, int(alpha))
    else:
        return _compute_log_a_for_frac_alpha(q, sigma, alpha)


def _compute_rdp(q: float, sigma: float, alpha: float) -> float:
    r"""RDP of one step of the SGM at order ``alpha`` (can be ``np.inf``)."""
    if q == 0:
        return 0

    # no privacy
    if sigma == 0:
        return np.inf

    if q == 1.0:
        return alpha / (2 * sigma ** 2)

    if np.isinf(alpha):
        return np.inf

    return _compute_log_a(q, sigma, alpha) / (alpha - 1)


def compute_rdp(
    q: float, noise_multiplier: float, steps: int, orders: Union[List[float], float]
) -> Union[List[float], float]:
    r"""RDP of the SGM composed over ``steps`` iterations.

    Args:
        q: Sampling rate of SGM.
        noise_multiplier: Noise std divided by the L2-sensitivity.
        steps: The number of iterations of the mechanism.
        orders: An array (or a scalar) of RDP orders.
    """
    if isinstance(orders, float):
        rdp = _compute_rdp(q, noise_multiplier, orders)
    else:
        rdp = np.array([_compute_rdp(q, noise_multiplier, order) for order in orders])

    return rdp * steps


def get_privacy_spent(
    orders: Union[List[float], float], rdp: Union[List[float], float], delta: float
) -> Tuple[float, float]:
    r"""Converts RDP values at several orders into ``(epsilon, best_order)`` for ``delta``.

    Uses Theorem 21 of Balle et al., "Hypothesis testing interpretations and
    Renyi differential privacy", AISTATS 2020 (https://arxiv.org/abs/1905.09982).

    Raises:
        ValueError: If the lengths of ``orders`` and ``rdp`` are not equal.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    eps = (
        rdp_vec
        - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
        + np.log((orders_vec - 1) / orders_vec)
    )

    # special case when there is no privacy
    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = np.nanargmin(eps)  # Ignore NaNs
    return eps[idx_opt], orders_vec[idx_opt]


###########################
# NOISE FROM THE BUDGET   #
###########################

DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def get_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    steps: int,
    alphas: List[float] = DEFAULT_ALPHAS,
    sigma_min: float = 1e-4,
    sigma_max: float = 100.0,
) -> Tuple[float, float]:
    r"""Smallest noise multiplier (up to ~0.01 in epsilon) such that ``steps``
    SGM steps at ``sample_rate`` are ``(target_epsilon, target_delta)``-DP by RDP.

    Returns:
        ``(sigma, epsilon)`` where ``epsilon < target_epsilon`` is the RDP epsilon at ``sigma``.
    """
    eps = float("inf")
    while eps > target_epsilon:
        sigma_max = 2 * sigma_max
        rdp = compute_rdp(sample_rate, sigma_max, steps, alphas)
        eps = get_privacy_spent(alphas, rdp, target_delta)[0]
        if sigma_max > 2000:
            raise ValueError("The privacy budget is too low.")
    eps_at_sigma_max = eps

    # Bisection; `sigma_max` always satisfies the budget, so it is what we return.
    while target_epsilon - eps_at_sigma_max > 1e-2 and sigma_max - sigma_min > 1e-6:
        sigma = (sigma_min + sigma_max) / 2
        rdp = compute_rdp(sample_rate, sigma, steps, alphas)
        eps = get_privacy_spent(alphas, rdp, target_delta)[0]

        if eps < target_epsilon:
            sigma_max, eps_at_sigma_max = sigma, eps
        else:
            sigma_min = sigma

    return sigma_max, eps_at_sigma_max


def get_exact_noise_multiplier(q: float, num_steps: int, epsilon: float, delta: float) -> float:
    """Noise multiplier from the numerical PRV accountant (``prv_accountant`` package)."""
    from prv_accountant import Accountant

    small_sigma, big_sigma = 0.001, 1000
    error = 0.001
    while big_sigma - small_sigma > error:
        sigma = (small_sigma + big_sigma) / 2
        accountant = Accountant(
            noise_multiplier=sigma,
            sampling_probability=q,
            delta=delta,
            eps_error=0.1,
            max_compositions=num_steps,
        )
        _, __, eps_estimate = accountant.compute_epsilon(num_compositions=num_steps)
        if eps_estimate < epsilon:
            big_sigma = sigma
        else:
            small_sigma = sigma
    return big_sigma


def cached_std(cache_file: Path, key: str, compute: Callable[[], float]) -> float:
    """Returns ``cache[key]`` from the ``torch.save``-d dict in ``cache_file``,
    computing (and storing) it with ``compute()`` on a miss."""
    cache_file = Path(cache_file)
    cache = torch.load(cache_file) if cache_file.exists() else {}
    if key in cache:
        print(f'==> loading std from {cache_file}')
        std = cache[key]
    else:
        std = compute()
        cache[key] = std
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_file)
    print(f'==> choosing std = {std}')
    return std


def get_std(q: float, num_steps: int, epsilon: float, delta: float, verbose: bool = True) -> float:
    """Noise multiplier for record-level DP-SGD with Poisson rate ``q`` over ``num_steps`` steps.

    Takes the smaller of the PRV-based and the RDP-based noise multipliers.
    ``epsilon > 1000`` is treated as "non-private" and uses a fixed small PRV
    noise of 0.09. Results are cached in ``privacy/stds.pt``.
    """
    def calculate():
        if epsilon > 1000:
            std_numerical = 0.09
        else:
            try:
                if verbose: print('==> calculating std using [ NUMERICAL ] method')
                std_numerical = get_exact_noise_multiplier(q, num_steps, epsilon, delta)
            except Exception as e:
                print(f'==>[error]: when calculating std using numerical method, {e}')
                std_numerical = 100

        try:
            if verbose: print('==> calculating std using [ ANALYTICAL ] method')
            std_rdp = get_noise_multiplier(
                target_epsilon=epsilon,
                target_delta=delta,
                sample_rate=q,
                steps=num_steps,
            )[0]
        except Exception as e:
            print(f'==>[error]: when calculating std using analytical method, {e}')
            std_rdp = 100

        if verbose:
            print(f'==> numerical 1:{round(std_numerical, 4)}, RDP 2:{round(std_rdp, 4)}')
        return min(std_numerical, std_rdp)

    print(f'\n\n{"=" * 40}\nprivacy accounting...')
    print(f'q = {q:.5f}, steps = {num_steps}, epsilon = {epsilon:.3f}, delta = {delta:.3e}')
    # full-precision key: rounding q/delta could reuse a std computed for other parameters
    key = f'q={q!r}|steps={num_steps}|epsilon={epsilon!r}|delta={delta!r}'
    std = cached_std(STD_CACHE_FILE, key, calculate)
    print('=' * 40)
    return std


if __name__ == "__main__":
    print(get_std(q=0.09308, num_steps=9 * 11, epsilon=9.5, delta=1e-5))
