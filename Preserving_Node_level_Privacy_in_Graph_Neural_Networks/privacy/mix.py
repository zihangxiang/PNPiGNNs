# from scipy.special import binom
from scipy.stats import binom
import torch
import numpy as np
import math
import time
import os
from tqdm import tqdm
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Device: {DEVICE}')


def log_factorial(n):
    """
    Compute the log of n!.
    """

    '''when n is a integer'''
    return np.log(np.array(range(1, n + 1))).sum()

    # return torch.log(torch.tensor(range(1, n + 1))).sum()


def get_privacy_spent(orders, rdp, delta):
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
    # print(f'best alpha: {orders_vec[idx_opt]}')

    return eps[idx_opt], orders_vec[idx_opt], idx_opt


class divergence_computer:
    def __init__(self, q_b, epoch, D_out, M_train = 1, saving_path = None):
        # self.noise_sigma = noise_sigma
        self.q_b = q_b
        self.epoch = epoch
        self.D_out = D_out
        self.M_train = M_train
        self.saving_path = saving_path
        # self.probs = self.generate_probs()

        ''''''
        self.sens = self.generate_sens()
        self.log_tmp_probs = self.compute_log_probs()

        n_alpha_points = 100
        min_alpha = 1.0001
        max_alpha = 10.0 

        self.alphas = torch.linspace(min_alpha, max_alpha, n_alpha_points, device = DEVICE)
    

    def from_rdp_to_dp(self, noise_sigma, delta = 1e-5, show_flag = True):
        s_time = time.time()

        ''' one direction'''
        tmp_alphas = self.alphas.view(1, -1)
        tmp_sens = self.sens.view(-1, 1)
        tmp_log_probs = self.log_tmp_probs.view(-1, 1)

        assert tmp_sens.shape == tmp_log_probs.shape, f'tmp_sens.shape: {tmp_sens.shape}, tmp_log_probs.shape: {tmp_log_probs.shape}'
        
        exponential_max = torch.zeros_like(tmp_alphas)

        '''gaussian'''
        # exponentials = tmp_alphas * (tmp_alphas - 1) * tmp_sens**2 / 2 / noise_sigma**2 + tmp_log_probs

        # '''laplacian'''
        the_b = noise_sigma / 2**0.5

        # '''right laplacian'''
        exponentials = (tmp_alphas - 1) * tmp_sens / the_b + tmp_log_probs + torch.log( 2 * tmp_alphas / (2 * tmp_alphas - 1) )
        exponential_max, _ = torch.max(exponentials, dim = 0)
        # print(f'exponential_max: {exponential_max}, max_index: {max_index}')
        tmp_1 = torch.exp( exponentials - exponential_max )
        tmp_2 = torch.exp( -tmp_alphas * tmp_sens / the_b + torch.log( (2 * tmp_alphas - 2) / (2 * tmp_alphas - 1) ) + tmp_log_probs - exponential_max ) 
        # tmp_3 = torch.exp( exponentials - exponential_max )
        exponentials = torch.log(  (tmp_1 + tmp_2) / 2 )

        # '''stablizing computation'''

        exp = torch.exp(exponentials)
        sum_exp = exp.sum(dim = 0)
        ln_exp_sum_max_reduced = torch.log(sum_exp)
        ln_exp_sum  = ln_exp_sum_max_reduced + exponential_max.view(1,-1)
        ln_exp_sum[ln_exp_sum < 0] = 1e-6
        # print(f'ln_exp_sum: {ln_exp_sum}')

        rdp = ( 1 / (tmp_alphas - 1) * ln_exp_sum ) \
            * int(self.epoch / self.q_b)

        # print(f'rdp: {rdp[:20]}, alpha: {self.alphas[:20]}')

        tmp_alphas = self.alphas.view(-1).to('cpu').numpy()
        rdp = rdp.view(-1).to('cpu').numpy()

        eps, best_alpha, idx_opt = get_privacy_spent(tmp_alphas, rdp, delta)

        if show_flag:
            print(f'eps: {eps}, delta: {delta}, at alpha: {best_alpha}')
            print(f'==> ToTaL TiMe FoR OnE RuN: {time.time() - s_time:.4f}\n')

        return eps, best_alpha
    

    def from_privacy_budget_to_DP(self, eps, delta = 1e-5, verbose = True):
        '''return sigma'''
        '''using binary search'''
        # s_time = time.time()
        def calculate():
            sigma_small = 0.001
            sigma_large = 100
            sigma = (sigma_small + sigma_large) / 2
            while sigma_large - sigma_small > 1e-2:

                tmp_eps, _ = self.from_rdp_to_dp(noise_sigma= sigma, delta = delta, show_flag = False)
                if tmp_eps > eps:
                    sigma_small = sigma
                else:
                    sigma_large = sigma
                sigma = (sigma_small + sigma_large) / 2
            # print(f'==> ToTaL TiMe FoR OnE RuN: {time.time() - s_time:.4f}\n')
            return sigma

        print(f'privacy accounting...')
        print(f'q = {self.q_b:.3f}, EPOCH = {self.epoch}, M_train: {self.M_train}, epsilon = {eps:.2f}, delta = {delta:7f}')
        key_name = f'q_{self.q_b:.4f}_EPOCH_{self.epoch}_M_train:_{self.M_train}_epsilon_{eps:.2f}_delta = {delta:8f}'
        file_name = f'stds_node_dp.pt'
        # file_path = self.saving_path / file_name
        file_path = Path(__file__).parent / file_name
        if file_path.exists():
            std_dict = torch.load(file_path)
            if key_name in std_dict:
                if verbose: print(f'==> loading std from file:{file_path}')
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

    
    
    def generate_sens(self):
        sens =  torch.tensor(range(self.D_out + 1))
        sens = sens.tolist()
        sens = [sens[0], 0.5] + sens[1:]
        sens = torch.tensor(sens, device = DEVICE)
        return sens
    
    def compute_log_probs(self):
        if self.D_out == 0:
             return torch.tensor([np.log(1 - self.q_b), np.log(self.q_b)], device=DEVICE)
        

        choosing_prob = self.q_b * self.M_train / self.D_out #/ np.log(self.D_out + 1)
        # log_of_choose_prob = np.log(self.q_b * self.M_train) - np.log(self.D_out)

       
        max_n = int(1e7)
        '''first check if there is pt file containing the log of n! up to 1000000'''
        file_name = f'log_factorial_{max_n}.pt'

        dir_path =  self.saving_path
        file_path = f'{dir_path}/{file_name}'
        os.mkdir(dir_path) if not os.path.exists(dir_path) else None
        if os.path.exists(file_path):
            print(f'==> load processed log of factorials from {file_path}')
            log_factorial_all = torch.load(file_path)

        else:
            print(f'==> compute log of factorials and save to {file_path}')
            
            log_factorial_all = torch.zeros(max_n+1, dtype=torch.float64)
            for i in tqdm(range(1, max_n+1)):
                log_factorial_all[i] = log_factorial_all[i - 1] + np.log(i)
            torch.save(log_factorial_all, file_path)

        log_of_factorials = log_factorial_all.view(-1)[:(self.D_out + 1)]
        # print(f'check precomputed: {log_of_factorials[-1]}, fresh: {log_factorial(self.D_out)}, at D_out {self.D_out}')
        #12891699.0, 12815518.0, at D_out 1000000

        ns =  torch.arange(0, self.D_out + 1, dtype=torch.int32)
        log_of_factorials = log_of_factorials.reshape(-1)


        log_binom = log_of_factorials[-1] - log_of_factorials - log_of_factorials.flip( dims = (0,) ) 
        log_prob_1 = ns.view(-1) * np.log(choosing_prob)
        log_prob_2 = (self.D_out - ns).view(-1) * np.log(1 - choosing_prob)
        log_prob = np.log(1 - self.q_b) + log_binom + log_prob_1 + log_prob_2

        log_prob = log_prob.reshape(-1).tolist()
        log_prob = log_prob[:1] + [np.log(self.q_b)] + log_prob[1:]
        # print(f'==> log_prob: {log_prob[:10]}')
        log_prob = torch.tensor(log_prob, device=DEVICE)
        # print(f'log_prob: {log_prob}')
        return log_prob
    


if __name__ == "__main__":

    computer = divergence_computer(
        q_b = 0.9, 
        epoch = 10,
        D_out = int(1e0), 
        M_train = 1, 
        saving_path = Path(__file__).parent,
        )

    delta = 1e-5
    computer.from_rdp_to_dp(noise_sigma = 1.65, delta = 1e-5)

    eps = 16
    sigma = computer.from_privacy_budget_to_DP(eps, delta)
    print(f'sigma: {sigma}')    
