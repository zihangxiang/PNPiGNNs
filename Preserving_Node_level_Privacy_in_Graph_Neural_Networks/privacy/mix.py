r"""Node-level privacy accountant (Theorem 2 of the paper).

Training privatises the sum of per-subgraph gradients, each clipped to norm
``C / 2``, with Gaussian noise of std ``sigma * C`` (see ``train_scheduler.py``).
Subgraph roots are Poisson-sampled with rate ``q``, and a node ``u`` with
``d_out`` potential roots joins a sampled root's subgraph with probability
``M_train / d_out`` (see ``privacy/sampling.py``).

From the point of view of one node, one training step is modelled as the
Gaussian mixture ``P = sum_i w_i N(s_i / sigma, 1)`` against ``Q = N(0, 1)``
(in units of ``C``), with components

* ``s = 1/2``, ``w = q``: the node is sampled as a root;
* ``s = n``,   ``w = (1 - q) * Binom(n; d_out, q * M_train / d_out)``,
  ``n = 0..d_out``: the node appears as a neighbour in ``n`` subgraphs.

The per-step Renyi divergence ``max(D_a(P||Q), D_a(Q||P))`` is computed by
numerical integration, composed over all steps, converted to
``(epsilon, delta)``-DP, and evaluated at the worst-case ``d_out`` in
``1..D_out``.

Example (run from the project directory)::

    python -m privacy.mix
"""
import math
import time
from pathlib import Path

import numpy as np
import torch

from privacy.accounting_analysis import cached_std, get_privacy_spent

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64

# RDP orders. Built as a float32 tensor, so each order is used (consistently)
# at its float32 value.
ALPHAS = torch.tensor([1 + x / 10.0 for x in range(1, 100)] + list(range(12, 30))).numpy()
STD_NORMAL = torch.distributions.normal.Normal(0, 1)
# Mixture components whose (normalised) log-weight is below this are dropped.
LOG_WEIGHT_CUTOFF = -30
# Grid in CDF space of N(0, 1): integrating f(Phi^-1(t)) dt over (0, 1) is E_{x~N(0,1)}[f(x)].
GRID_EPS = 1e-8
GRID_POINTS = int(1e6)
# (alpha, sigma) at which the worst-case d_out is selected.
WORST_CASE_ALPHA, WORST_CASE_SIGMA = 1.3, 1


def _normalize_log(log_values, log_dx):
    """Shifts ``log_values`` so that ``sum(exp(log_values + log_dx)) == 1``."""
    return log_values - torch.logsumexp(log_values + log_dx, dim=0)


class NodeDPAccountant:
    """Node-level RDP accountant for subgraph-sampled DP-SGD.

    Args:
        q: Poisson sampling rate of root nodes (``expected_batchsize / num_roots``).
        num_steps: Number of noisy gradient steps taken in training.
        D_out: Largest ``d_out`` (number of potential roots of a node) considered.
        M_train: Neighbour-sampling budget ``M`` used in training (``--num_neighbors``).
    """

    def __init__(self, q, num_steps, D_out, M_train=1):
        assert 0 <= q <= 1, f'q should be in [0, 1], but got {q}'
        assert D_out > 0, f'D_out should be greater than 0, but got {D_out}'
        self.q = q
        self.num_steps = num_steps
        self.D_out = D_out
        self.M_train = M_train

        grid = torch.linspace(GRID_EPS, 1 - GRID_EPS, GRID_POINTS, device=DEVICE, dtype=DTYPE)
        self.log_dt = torch.log(grid[1] - grid[0])
        self.x = STD_NORMAL.icdf(grid).reshape(1, -1)

        self._set_mixture(self._worst_case_d_out())

    # ---------------------------------------------------------------- mixture
    def _set_mixture(self, d_out):
        """Sets ``self.sens`` / ``self.log_w`` to the (truncated) mixture for ``d_out``."""
        sens = torch.tensor([0, 0.5] + list(range(1, d_out + 1)), device=DEVICE, dtype=DTYPE)
        log_w = _normalize_log(self._log_weights(d_out), 0)
        keep = log_w >= LOG_WEIGHT_CUTOFF
        self.sens, self.log_w = sens[keep], log_w[keep]

    def _log_weights(self, d_out):
        """Log mixture weights, aligned with sensitivities ``[0, 1/2, 1, ..., d_out]``."""
        p = self.q * self.M_train / d_out
        n = torch.arange(0, d_out + 1, dtype=DTYPE)
        log_fact = torch.lgamma(n + 1)
        log_binom = log_fact[-1] - log_fact - log_fact.flip(dims=(0,))
        log_w = math.log(1 - self.q) + log_binom + n * math.log(p) + (d_out - n) * math.log(1 - p)
        log_root = torch.tensor([math.log(self.q)], dtype=DTYPE)
        log_w = torch.cat([log_w[:1], log_root, log_w[1:]]).to(DEVICE)
        assert torch.all(log_w <= 0), "All entries in log_w should be less or equal to 0"
        return log_w

    def _worst_case_d_out(self):
        print(f'Computing RDP for D_out from 1 to {self.D_out} to find the worst case...')
        start = time.time()
        max_rdp, worst_d_out = 0, None
        for d_out in range(1, self.D_out + 1):
            self._set_mixture(d_out)
            rdp = self._rdp_one_step(WORST_CASE_ALPHA, self._log_likelihood_ratio(WORST_CASE_SIGMA))
            if rdp > max_rdp:
                max_rdp, worst_d_out = rdp, d_out
        print(f'Max RDP {max_rdp * self.num_steps:.4f} at D_out = {worst_d_out}, '
              f'computation time: {time.time() - start:.2f} seconds')
        return worst_d_out

    # ------------------------------------------------------------- divergence
    def _log_likelihood_ratio(self, sigma):
        """``log P(x) / Q(x)`` on the grid, normalised to integrate to one."""
        mu = (self.sens / sigma).reshape(-1, 1)
        log_ratio = torch.logsumexp(0.5 * (2 * self.x - mu) * mu + self.log_w.reshape(-1, 1), dim=0)
        return _normalize_log(log_ratio, self.log_dt)

    def _rdp_one_step(self, alpha, log_ratio):
        """``max(D_alpha(P||Q), D_alpha(Q||P))`` for one step."""
        rdp_q_p = torch.logsumexp(log_ratio * (1 - alpha) + self.log_dt, dim=0) / (alpha - 1)
        rdp_p_q = torch.logsumexp(log_ratio * alpha + self.log_dt, dim=0) / (alpha - 1)
        return max(rdp_q_p.item(), rdp_p_q.item())

    # ------------------------------------------------------------- public API
    def eps_from_noise(self, sigma, delta=1e-5):
        """``(epsilon, best_alpha)`` after ``num_steps`` steps at noise multiplier ``sigma``."""
        log_ratio = self._log_likelihood_ratio(sigma)
        rdp = np.array([self._rdp_one_step(alpha, log_ratio) for alpha in ALPHAS]) * self.num_steps
        return get_privacy_spent(ALPHAS, rdp, delta)

    def noise_from_eps(self, eps, delta=1e-5):
        """Binary-searches the noise multiplier (to within 0.01) that achieves ``(eps, delta)``.

        Returns the upper end of the final bracket, which is verified to satisfy the budget.
        """
        print('privacy accounting...')
        sigma_small, sigma_large = 0.001, 100.0
        if self.eps_from_noise(sigma_large, delta)[0] > eps:
            raise ValueError(f'The privacy budget is too low: sigma = {sigma_large} gives epsilon > {eps}.')
        while sigma_large - sigma_small > 1e-2:
            sigma = (sigma_small + sigma_large) / 2
            if self.eps_from_noise(sigma, delta)[0] > eps:
                sigma_small = sigma
            else:
                sigma_large = sigma
        return sigma_large


def get_std_node_dp(q, num_steps, D_out, M_train, epsilon, delta, cache_dir=None):
    """Noise multiplier for node-level ``(epsilon, delta)``-DP; cached in ``cache_dir`` if given."""
    def compute():
        accountant = NodeDPAccountant(q=q, num_steps=num_steps, D_out=D_out, M_train=M_train)
        return accountant.noise_from_eps(epsilon, delta)

    if cache_dir is None:
        return compute()
    key = f'q={q!r}|steps={num_steps}|D_out={D_out}|M={M_train}|epsilon={epsilon!r}|delta={delta!r}'
    return cached_std(Path(cache_dir) / 'node_dp_stds.pt', key, compute)


if __name__ == "__main__":
    accountant = NodeDPAccountant(q=0.2, num_steps=45, D_out=20000, M_train=1)

    delta = 1e-5
    eps, alpha = accountant.eps_from_noise(sigma=1.65, delta=delta)
    print(f'eps: {eps}, alpha: {alpha}')

    sigma = accountant.noise_from_eps(2, delta)
    print(f'sigma: {sigma}')
