#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
*Based on Google's TF Privacy:* https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/rdp_accountant.py.
*Here, we update this code to Python 3, and optimize dependencies.*
Functionality for computing Renyi Differential Privacy (RDP) of an additive
Sampled Gaussian Mechanism (SGM).
Example:
    Suppose that we have run an SGM applied to a function with L2-sensitivity of 1.
    Its parameters are given as a list of tuples
    ``[(q_1, sigma_1, steps_1), ..., (q_k, sigma_k, steps_k)],``
    and we wish to compute epsilon for a given target delta.
    The example code would be:
    >>> max_order = 32
    >>> orders = range(2, max_order + 1)
    >>> rdp = np.zeros_like(orders, dtype=float)
    >>> for q, sigma, steps in parameters:
    >>>     rdp += privacy_analysis.compute_rdp(q, sigma, steps, orders)
    >>> epsilon, opt_order = privacy_analysis.get_privacy_spent(orders, rdp, delta)
"""


import math
from typing import List, Tuple, Union

import numpy as np
from scipy import special

from . import mix
########################
# LOG-SPACE ARITHMETIC #
########################


def _log_add(logx: float, logy: float) -> float:
    r"""Adds two numbers in the log space.
    Args:
        logx: First term in log space.
        logy: Second term in log space.
    Returns:
        Sum of numbers in log space.
    """
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx: float, logy: float) -> float:
    r"""Subtracts two numbers in the log space.
    Args:
        logx: First term in log space. Expected to be greater than the second term.
        logy: First term in log space. Expected to be less than the first term.
    Returns:
        Difference of numbers in log space.
    Raises:
        ValueError
            If the result is negative.
    """
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx


def _compute_log_a_for_int_alpha(q: float, sigma: float, alpha: int) -> float:
    r"""Computes :math:`log(A_\alpha)` for integer ``alpha``.
    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.
        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.
    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.
    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """

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
    r"""Computes :math:`log(A_\alpha)` for fractional ``alpha``.
    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.
        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.
    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.
    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """
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
    r"""Computes :math:`log(A_\alpha)` for any positive finite ``alpha``.
    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.
        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf
        for details.
    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.
    Returns:
        :math:`log(A_\alpha)` as defined in the paper mentioned above.
    """
    if float(alpha).is_integer():
        return _compute_log_a_for_int_alpha(q, sigma, int(alpha))
    else:
        return _compute_log_a_for_frac_alpha(q, sigma, alpha)


def _log_erfc(x: float) -> float:
    r"""Computes :math:`log(erfc(x))` with high accuracy for large ``x``.
    Helper function used in computation of :math:`log(A_\alpha)`
    for a fractional alpha.
    Args:
        x: The input to the function
    Returns:
        :math:`log(erfc(x))`
    """
    return math.log(2) + special.log_ndtr(-x * 2 ** 0.5)


def _compute_rdp(q: float, sigma: float, alpha: float) -> float:
    r"""Computes RDP of the Sampled Gaussian Mechanism at order ``alpha``.
    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.
    Returns:
        RDP at order ``alpha``; can be np.inf.
    """
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
    r"""Computes Renyi Differential Privacy (RDP) guarantees of the
    Sampled Gaussian Mechanism (SGM) iterated ``steps`` times.
    Args:
        q: Sampling rate of SGM.
        noise_multiplier: The ratio of the standard deviation of the
            additive Gaussian noise to the L2-sensitivity of the function
            to which it is added. Note that this is same as the standard
            deviation of the additive Gaussian noise when the L2-sensitivity
            of the function is 1.
        steps: The number of iterations of the mechanism.
        orders: An array (or a scalar) of RDP orders.
    Returns:
        The RDP guarantees at all orders; can be ``np.inf``.
    """
    if isinstance(orders, float):
        rdp = _compute_rdp(q, noise_multiplier, orders)
    else:
        rdp = np.array([_compute_rdp(q, noise_multiplier, order) for order in orders])

    return rdp * steps


def get_privacy_spent(
    orders: Union[List[float], float], rdp: Union[List[float], float], delta: float
) -> Tuple[float, float]:
    r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
    multiple RDP orders and target ``delta``.
    The computation of epslion, i.e. conversion from RDP to (eps, delta)-DP,
    is based on the theorem presented in the following work:
    Borja Balle et al. "Hypothesis testing interpretations and Renyi differential privacy."
    International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
    Particullary, Theorem 21 in the arXiv version https://arxiv.org/abs/1905.09982.
    Args:
        orders: An array (or a scalar) of orders (alphas).
        rdp: A list (or a scalar) of RDP guarantees.
        delta: The target delta.
    Returns:
        Pair of epsilon and optimal order alpha.
    Raises:
        ValueError
            If the lengths of ``orders`` and ``rdp`` are not equal.
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

DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
def get_noise_multiplier(
    target_epsilon, #float
    target_delta, #float
    sample_rate, #float
    epochs, #int
    alphas = DEFAULT_ALPHAS, # List[float]
    sigma_min = 1e-4, 
    sigma_max = 100.0,
):
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate
    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        epochs: the number of epochs to run
        alphas: the list of orders at which to compute RDP
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """
    eps = float("inf")
    while eps > target_epsilon:
        sigma_max = 2 * sigma_max
        rdp = compute_rdp(
            sample_rate, sigma_max, epochs / sample_rate, alphas
        )
        eps = get_privacy_spent(alphas, rdp, target_delta)[0]
        if sigma_max > 2000:
            raise ValueError("The privacy budget is too low.")

    while (eps - target_epsilon)**2 > 1e-4: #sigma_max - sigma_min > 0.01:
        sigma = (sigma_min + sigma_max) / 2
        rdp = compute_rdp(
            sample_rate, sigma, epochs / sample_rate, alphas
        )
        eps = get_privacy_spent(alphas, rdp, target_delta)[0]

        if eps < target_epsilon:
            sigma_max = sigma
        else:
            sigma_min = sigma

    return sigma, eps


def LB_of_sigma_square(T=10000, epsilon=2, delta=1e-5):
    tmp1 = 2*T/(epsilon**2)*np.log(1/delta) + T/epsilon
    tmp2 = -2*T/(epsilon**2)*(np.log( 2*np.log(1/delta) ) +1 - np.log(epsilon))
    tmp3 = (np.log(np.log(1/delta))**2)/np.log(1/delta) # term of big O notation, should be small relatively
    if tmp3 > (tmp1+tmp2)/100:
        print('\n==> symptotic assumption does not hold, the constant hidding in big O is too large')
        print(tmp1+tmp2,tmp3)
        raise ValueError('symptotic assumption does not hold, the constant hidding in big O is too large')
    return tmp1+tmp2+tmp3


def get_exact_noise_multiplier(q = None, EPOCH = None, epsilon = None, delta = None, ):
    from prv_accountant import Accountant

    small_sigma, big_sigma = 0.001, 1000
    error = 0.001
    T = int(EPOCH/q)
    while big_sigma - small_sigma > error:
        sigma = (small_sigma + big_sigma) / 2
        accountant = Accountant(
                            noise_multiplier = sigma,
                            sampling_probability = q,
                            delta = delta,
                            eps_error = 0.1,
                            max_compositions = T,
                        )
        _, __, eps_estimate = accountant.compute_epsilon(num_compositions = T)
        # print(sigma, eps_estimate)
        if eps_estimate < epsilon:
            big_sigma = sigma
        else:
            small_sigma = sigma
    return big_sigma

                            
def get_std(q = None, EPOCH = None, epsilon = None, delta = None, verbose = True):
    def calculate():
        if epsilon > 1000:
            std_analyszer_1 = 0.09
        else:
            try:
                if verbose: print('==> calculating std using [ NUMERICAL ] method')
                std_analyszer_1 = get_exact_noise_multiplier(q = q, EPOCH = EPOCH, epsilon = epsilon, delta = delta, )
            except Exception as e:
                print(f'==>[error]: when calculating std using approximate method, {e}')
                std_analyszer_1 = 100

        try:
            if verbose: print('==> calculating std using [ ANALYTICAL ] method')
            std_analyszer_2 = get_noise_multiplier(
                                                    target_epsilon = epsilon, 
                                                    target_delta = delta,
                                                    sample_rate = q,
                                                    epochs = EPOCH,
                                                    )[0]
        except Exception as e:
            print(f'==>[error]: when calculating std using analytical method, {e}')
            std_analyszer_2 = 100
        
        std = min(std_analyszer_1, std_analyszer_2)
        if verbose:
            print(f'==> numerical 1:{round(std_analyszer_1, 4)}, RDP 2:{round(std_analyszer_2,4)}')
            print(f'==> choosing std = {std}')
        print(f'{"="*40}')
        return std

    from pathlib import Path
    import torch
    print(f'\n\n{"="*40}')
    print(f'privacy accounting...')
    print(f'q = {q:.3f}, EPOCH = {EPOCH}, epsilon = {epsilon:.3f}, delta = {delta:7f}')
    key_name = f'q_{q:.3f}_EPOCH_{EPOCH}_epsilon_{epsilon:.3f}_delta_{delta:7f}'
    file_name = f'stds.pt'
    file_path = Path(__file__).parent / file_name
    if file_path.exists():
        std_dict = torch.load(file_path)
        if key_name in std_dict:
            if verbose: print('==> loading std from file')
            std = std_dict[key_name]
            print(f'==> choosing std = {std}')
            return std
        else:
            std = calculate()
            std_dict[key_name] = std
            torch.save(std_dict, file_path)
            return std
    else:
        std = calculate()
        std_dict = {key_name: std}
        torch.save(std_dict, file_path)
        return std
    
def get_std_node_dp(
    q = None,
    EPOCH = None,
    D_out = None,
    M_train = None,
    epsilon = None,
    delta = None,
    saving_path = None,
    ):

    computer = mix.divergence_computer(
        q_b = q, 
        epoch = EPOCH,
        D_out = D_out, 
        M_train = M_train, 
        saving_path = saving_path,
        )

    sigma = computer.from_privacy_budget_to_DP(epsilon, delta)
    # print(f'sigma: {sigma}')   
    return sigma


def __decay_func(i ,epoch, high, low):
    import math
    b = high * 2
    a = (epoch - 1) / math.log(b/low-1)
    return b/(1+math.exp(i/a)) 

def get_decaying_noise_multiplyer(q = None, EPOCH = None, epsilon = None, delta = None, verbose = True):
    pass
    decaying_function = __decay_func
    decaying_function = lambda i, epoch, high, low: high - i * (high - low) / (epoch-1)
    
    the_std = get_std(q = q, EPOCH = EPOCH, epsilon = epsilon, delta = delta, verbose = verbose)
    low_std = the_std * 0.8
    
    ratio = 1.43
    rdp = 0
    for i in range(EPOCH):
         rdp +=  compute_rdp(q, 
                             decaying_function(i, EPOCH, the_std * ratio, low_std),
                             1 / q, 
                             DEFAULT_ALPHAS)
    
    eps = get_privacy_spent(DEFAULT_ALPHAS, rdp, delta)[0]
    print(111, eps)
    
    print([decaying_function(i, EPOCH, the_std * ratio, low_std) for i in range(EPOCH)])
    
    



if __name__ == "__main__":
    a = get_std(q = 0.09308, EPOCH = 9, epsilon = 9.5, delta = 1e-5, verbose = True)
    print(a)
