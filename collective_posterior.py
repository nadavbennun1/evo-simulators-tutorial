# Collective Posterior Distribution - PyTorch implementation
import torch
import math

# Optimizers for Maximum A-Posteriori (MAP)
import numpy as np
from scipy.optimize import minimize

# Progress bars
from tqdm import tqdm

# Parallelization (for rejection sampling)
from concurrent.futures import ThreadPoolExecutor, as_completed


class CollectivePosterior:
    """
    A class to represent and sample from a collective posterior distribution.

    Parameters:
        prior: Prior distribution object compatible with `torch.distributions`.
        amortized_posterior: Pretrained posterior model (from sbi or other libraries).
        Xs: Set of observed data points.
        log_C: Normalizing constant for the posterior. Default is 1. Should be estimated using get_log_C()
        epsilon: Sensitivity parameter, used as a lower bound for posterior probabilities. Default is -10000. Can be estimated as described in paper. 
        n_eval: Number of samples used for Monte Carlo estimation. Default is 1e5.
        sample_var: Variance for MCMC-like sampling. Default is 0.05.
        posterior_list: In case of using non-amortized posteriors, a list of log_prob functions can be used instead.
    """
       

    def __init__(self, prior, Xs, amortized_posterior=None, log_C=None, epsilon=-10000,
                 n_eval=int(1e5), sample_var=0.05, posterior_list=[]):
        self.prior = prior
        self.amortized_posterior = amortized_posterior
        self.Xs = Xs
        self.log_C = log_C
        self.epsilon = epsilon
        self.map = None
        self.samples = torch.empty((0, prior.sample().shape[0]))
        self.theta_dim = prior.sample().shape[0]
        self.n_eval = n_eval
        self.sample_var = sample_var
        self.posterior_list = posterior_list
        
        self.temp = math.sqrt(len(self.Xs))

        # Ensure inference is possible
        assert (len(posterior_list) > 0 or amortized_posterior is not None)


    def log_prob(self, theta):
        """
        Compute log p_collective(theta) = sum_i log p_i(theta|x_i) - (r-1) log p(theta) - log C
        Assumes self.log_C was set via get_log_C() or provided at init.
        """
        if self.log_C is None:
            raise RuntimeError("log_C is not set. Call get_log_C() first or pass it at init.")

        # to tensor, allow (d,) or (N,d)
        theta = torch.as_tensor(theta, dtype=torch.float32)
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)  # (1, d)

        N = theta.shape[0] # parameters
        r = len(self.Xs) #replicates

        # collect per-replicate log p_i
        log_probs = torch.empty((N, r), dtype=torch.float32)
        for i in range(r):
            if len(self.posterior_list) == 0:  # amortized posterior
                lp_i = self.amortized_posterior.set_default_x(self.Xs[i, :]).log_prob(theta, norm_posterior=False)
            else:  # provided per-replicate log-probs
                lp_i = self.posterior_list[i](theta)  # expects (N,) or (N,1)
            log_probs[:, i] = torch.clamp(lp_i.reshape(N,), min=self.epsilon)  # epsilon is a LOG floor

        sum_logp = log_probs.sum(dim=1)                           # (N,)
        logp = self.prior.log_prob(theta).reshape(N,)             # (N,)

        return sum_logp - (r - 1) * logp - self.log_C # log p_collective(theta) = sum_i log p_i(theta|x_i) - (r-1) log p(theta) - log C


    def sample(self, n_samples, keep=True, step_size=0.05, take_sn=50, jump=1e5, m=1.5, num_workers=24, method='mcmc'):
        if method == 'mcmc':
            return self.mcmc_from_top_sn(n_total=n_samples, step_size=step_size, take_sn=take_sn)
        elif method == 'rejection':
            return self.rejection_sample(n_samples, jump=jump, m=m, keep=keep, n_workers=num_workers)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def mcmc_from_top_sn(self, n_total=1000, step_size=0.05, take_sn=50):
        """
        Run independent Metropolis-Hastings chains from each top SN samples (by log-probs) in self.samples.
        n_total: total number of samples to generate (will be split across chains).
        Includes a global progress bar.
        Returns: tensor of shape (n_total, theta_dim)
        """
        assert len(self.samples) >= take_sn # enough samples to start

        top_sn_samples = self.samples[:take_sn]
        log_prob_fn = lambda theta: self.log_prob(theta)
        all_samples = []
        theta_dim = top_sn_samples.shape[1]
        n_per_chain = math.ceil(n_total / top_sn_samples.shape[0])
        total_steps = n_per_chain * top_sn_samples.shape[0]
        with tqdm(total=total_steps, desc=f"MCMC from top {take_sn} candidates", initial=take_sn) as global_pbar:
            for idx, start in enumerate(top_sn_samples):
                chain = [start.clone()]
                cur_logp = log_prob_fn(start.unsqueeze(0))[0]
                # chain_pbar = tqdm(total=n_per_chain, desc=f"Chain {idx+1}/{take_sn}", leave=False)
                for _ in range(n_per_chain - 1):
                    proposal = chain[-1] + torch.randn(theta_dim) * step_size
                    prop_logp = log_prob_fn(proposal.unsqueeze(0))[0]
                    accept = torch.rand(1).item() < torch.exp(prop_logp - cur_logp)
                    if accept:
                        chain.append(proposal)
                        cur_logp = prop_logp
                    else:
                        chain.append(chain[-1].clone())
                    # chain_pbar.update(1)
                    global_pbar.update(1)
                # chain_pbar.close()
                all_samples.append(torch.stack(chain))
        samples = torch.cat(all_samples, dim=0)[:n_total]
        return samples


# parallel version of rejection sampling
    def rejection_sample(self, n_samples, jump=int(1e5), m = 1.5, keep=True, n_workers=8):
        """
        Sample from the collective posterior using parallel rejection sampling.

        Parameters:
            n_samples (int): Number of samples to generate.
            jump (int): Number of prior samples to draw in each batch. Default is 1e5.
            m (float): Offset for the acceptance criterion. Default is 5.
            keep (bool): Whether to store the samples in the object. Default is True.
            n_workers (int): Number of parallel workers. Default is 8.

        Returns:
            torch.Tensor: Samples from the posterior.
        """
        samples = torch.empty((1, self.theta_dim))
        cur = 0
        with tqdm(total=n_samples, desc="Rejection Sampling") as pbar:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                while cur < n_samples:
                    futures.append(executor.submit(self._rejection_sample_batch, jump, m))
                    # Collect results as they complete
                    for future in as_completed(futures):
                        samps, count = future.result()
                        samples = torch.cat([samples, samps])
                        cur += count
                        pbar.update(count)
                        futures.remove(future)
                        if cur >= n_samples:
                            break
        if keep:
            self.samples = samples[1:n_samples+1]
        return samples[1:n_samples+1]
    
    # rejection sampling helper
    def _rejection_sample_batch(self, jump, m):
        samps = self.prior.sample((jump,))
        prior_probs = self.prior.log_prob(samps)
        probs = torch.log(torch.rand(samps.size()[0]))
        lp = self.log_prob(samps)
        next_idx = (lp - prior_probs) > (probs + m)
        samples_to_add = samps[next_idx]
        return samples_to_add, next_idx.sum().item()
    
    


    
    def get_map(self, n_init=100, keep=True):
        """
        Compute the Maximum A Posteriori (MAP) estimate.

        Parameters:
            keep (bool): Whether to store the MAP estimate in the object. Default is True.
        
        Returns:
            torch.Tensor: MAP estimate of the posterior mode.
        """
        samples = self.samples
        
        if len(self.posterior_list) > 0:
            func = lambda theta: np.array(-1*self.log_prob(theta)).reshape(1,-1) 
        else:
            func = lambda theta: np.array(-1*self.log_prob(theta)) # max(log_prob) = min(-log_prob) + np array for optimizer
        if len(self.samples) > 0:
            x0 = self.samples[self.log_prob(samples).argmax()]
        else:
            samples = self.sample(n_init)
            x0 = self.samples[self.log_prob(samples).argmax()] # get a nice random guess
        x0 = list(x0)
        collective_map = torch.from_numpy(minimize(func, x0, method='Nelder-Mead').x) # Optimization
        if keep:
            self.map = collective_map
        return collective_map
        
    
    def get_log_C(self, n_reps=10, S=0.005):
        """
        Estimate log C = log ∫ u(θ) dθ via importance sampling with θ ~ p(θ),
        where u(θ) = Π_i p_i(θ|x_i) / p(θ)^{r-1}.
        Monte Carlo identity:
            C = E_{θ~p} [ exp( Σ_i log p_i(θ|x_i) - r * log p(θ) ) ].
        We average in probability space across n_reps repetitions, then take log.
        We also keep S * self.n_eval samples as initializers for MCMC sampling. Defaults yield 5000.
        """
        r = len(self.Xs)
        N = int(self.n_eval)

        rep_logs = []
        for _ in range(n_reps):
            theta = self.prior.sample((N,))                       # (N,d)
            prior_logps = self.prior.log_prob(theta).reshape(N,)  # (N,)

            # build (N, r) matrix of log p_i
            log_probs = torch.empty((N, r), dtype=torch.float32)
            for i in range(r):
                if len(self.posterior_list) == 0:
                    lp_i = self.amortized_posterior.set_default_x(self.Xs[i, :]).log_prob(theta, norm_posterior=False)
                else:
                    lp_i = self.posterior_list[i](theta)
                log_probs[:, i] = torch.clamp(lp_i.reshape(N,), min=self.epsilon)  # LOG floor

            sum_logp = log_probs.sum(dim=1)                       # (N,)
            # importance weights in log-space: log u(θ) - log p(θ) = Σ log p_i - r log p
            weights_log = sum_logp - r * prior_logps              # (N,)
            # log-mean-exp over N samples
            logC_rep = torch.logsumexp(weights_log, dim=0) - math.log(N)
            rep_logs.append(logC_rep)
            # add top S samples in self.samples
            topk = weights_log.topk(int(S*N)).indices

            self.samples = torch.cat([self.samples, theta[topk]], dim=0)

        # average across repetitions in probability space, then log
        self.log_C = torch.logsumexp(torch.stack(rep_logs), dim=0) - math.log(len(rep_logs))
        # sort samples by log prob
        if len(self.samples) > 0:
            sorted_indices = self.log_prob(self.samples).argsort(descending=True)
            self.samples = self.samples[sorted_indices]

        return self.log_C


    def estimate_epsilon(self, x, prior=None, posterior=None, T=100, quant=0.95, n_reps=10):
        """
        Estimate epsilon value for the collective posterior distribution
        By taking the 95th percentile of the log probabilities of random prior samples.
        Requires:
        x - list of observations (to be simulated separately).
        posterior - amortized posterior.
        """
        if prior is None:
            prior = self.prior
        if posterior is None:
            posterior = self.amortized_posterior
        S = len(x)
        total_lp = 0
        for i in range(n_reps):
            lp_random = torch.empty(S, T)
            for i in range(S):
                lp_random[i] = posterior.set_default_x(x[i]).log_prob(prior.sample((T,)), norm_posterior=False)
            total_lp += lp_random.quantile(quant)
        return total_lp / n_reps, lp_random



    
    def sample_via_sir_jitter(self, n_draws=100_000, n_final=10_000, bandwidth_scale=1, temperature=None, excess_quantile=0.5):
        """
        Gradient-free sampler: SIR with Oversampling/Pruning and Gaussian Jitter.
        
        Args:
            n_draws (int): Size of the initial proposal pool (from prior).
            n_final (int): Number of final posterior samples desired.
            bandwidth_scale (float): Scale of Gaussian jitter (0.15 recommended).
            temperature (float): Tempering factor (T > 1 flattens, T < 1 sharpens).
            excess_quantile (float): Fraction of extra samples to draw and then discard.
                                     (e.g., 0.05 = sample 105%, drop worst 5%).
        """
        n_reps = len(self.Xs)
        
        # --- PHASE 1: STANDARD SIR ---
        # 1. Proposal
        theta_pool = self.prior.sample((n_draws,))
        log_proposal = self.prior.log_prob(theta_pool)
        temperature = temperature if temperature is not None else self.temp
        # 2. Evaluate Target
        log_liks = []
        for i in tqdm(range(n_reps), desc=f"Evaluating {n_draws} samples"):
            if self.amortized_posterior is not None:
                if hasattr(self.amortized_posterior, 'set_default_x'):
                    self.amortized_posterior.set_default_x(self.Xs[i])
                ll = self.amortized_posterior.log_prob(theta_pool, norm_posterior=False)
            else:
                ll = self.posterior_list[i].log_prob(theta_pool)
            log_liks.append(ll)
        
        log_L_matrix = torch.stack(log_liks, dim=0)
        
        # Robust Aggregation
        eps_tensor = torch.as_tensor(self.epsilon, device=log_L_matrix.device, dtype=log_L_matrix.dtype)
        log_L_robust = torch.logaddexp(log_L_matrix, eps_tensor)
        
        log_target_unnorm = log_L_robust.sum(dim=0) - (n_reps - 1) * self.prior.log_prob(theta_pool)
        
        # 3. Weights
        log_weights = log_target_unnorm - log_proposal
        log_weights = torch.nan_to_num(log_weights, nan=-float('inf'))
        
        # --- TEMPERATURE SCALING ---
        if temperature != 1.0:
            log_weights = log_weights / temperature
        
        weights = torch.softmax(log_weights, dim=0)
        
        if weights.sum() == 0 or torch.isnan(weights).any():
             print("Warning: All samples rejected. Returning prior.")
             return self.prior.sample((n_final,))

        # --- PHASE 1.5: OVERSAMPLING & PRUNING ---
        # Draw extra samples (e.g., 105%)
        n_safe = int(n_final * (1 + excess_quantile))
        indices = torch.multinomial(weights, n_safe, replacement=True)
        
        # Retrieve the original unnormalized weights of the CHOSEN samples
        selected_log_w = log_weights[indices]
        
        # Keep only the top n_final samples (discard the worst %)
        # torch.topk works largest->smallest
        best_indices_local = torch.topk(selected_log_w, n_final).indices
        
        # Map back to pool indices
        final_indices = indices[best_indices_local]
        raw_samples = theta_pool[final_indices]

        # --- PHASE 2: JITTER (Gaussian Perturbation) ---
        sigma = raw_samples.std(dim=0)
        sigma[sigma == 0] = 1e-6 
        
        noise = torch.randn_like(raw_samples) * (sigma * bandwidth_scale)
        smoothed_samples = raw_samples + noise
        
        # --- PHASE 3: BOUNDARY REFLECTION ---
        low, high = None, None
        
        if hasattr(self.prior, 'support'):
            try:
                low = self.prior.support.base_constraint.lower_bound
                high = self.prior.support.base_constraint.upper_bound
            except: pass
        elif hasattr(self.prior, 'low'):
            low, high = self.prior.low, self.prior.high
            
        if low is not None and high is not None:
            if torch.is_tensor(low): low = low.to(smoothed_samples.device)
            if torch.is_tensor(high): high = high.to(smoothed_samples.device)
            
            # 1. Reflect Lower
            smoothed_samples = torch.where(
                smoothed_samples < low, 
                2 * low - smoothed_samples, 
                smoothed_samples
            )
            
            # 2. Reflect Upper
            smoothed_samples = torch.where(
                smoothed_samples > high, 
                2 * high - smoothed_samples, 
                smoothed_samples
            )
            
            # 3. Final Clamp
            smoothed_samples = torch.clamp(smoothed_samples, low, high)
            
        self.samples = smoothed_samples
        ess = 1.0 / torch.sum(weights ** 2)
        print(f"Sampled {n_final} points with jitter and reflection. ESS = {ess}")
        return smoothed_samples

    ### ARCHIVE ###

    # def sample_one(self, jump=int(1e5), keep=False):
    #     """
    #     Draw a single sample from the posterior.

    #     Parameters:
    #         jump (int): Number of prior samples to draw in each batch. Default is 1e4.
    #         keep (bool): Whether to store the sample in the object. Default is True.
        
    #     Returns:
    #         torch.Tensor: A single sample from the posterior.
    #     """
    #     sampled=False
    #     while not(sampled):
    #         samps = self.prior.sample((jump,))
    #         probs = torch.rand(samps.size()[0])
    #         lp = self.log_prob(samps)
    #         next_idx = lp > probs
    #         if next_idx.sum()>0:
    #             sampled=True
    #     return samps[next_idx][0]
    
    # def sample_around(self, theta, jump=int(1e4)):
    #     """
    #     Sample around a given point using a Gaussian proposal.

    #     Parameters:
    #         theta (torch.Tensor): Center of the Gaussian proposal.
    #         jump (int): Number of candidates to sample. Default is 1e4.
        
    #     Returns:
    #         tuple: (Accepted samples, new center theta).
    #     """
    #     dist =  torch.distributions.multivariate_normal.MultivariateNormal(theta, torch.diag(torch.tensor([self.sample_var]*self.theta_dim)))
    #     cands = dist.sample((jump,))
    #     probs = self.log_prob(cands)
    #     baseline = torch.rand((len(cands),))
    #     res = cands[probs>baseline]
    #     new_theta = cands[probs.argmax()]
    #     return res, new_theta
    
#     def sample_unimodal(self, n_samples, jump=int(1e5), keep=True):
#         """
#         Sample from the posterior using iterative proposals.

#         Parameters:
#             n_samples (int): Number of samples to generate.
#             jump (int): Number of candidates to sample at each iteration. Default is 1e4.
#             keep (bool): Whether to store the samples in the object. Default is True.
        
#         Returns:
#             torch.Tensor: Samples from the posterior.
#         """
#         theta = self.sample_one(jump, keep=False)
#         samples = torch.empty((n_samples, len(theta)))
#         cur = 0

#         with tqdm(total=n_samples, desc="Sampling") as pbar:
#             while cur < n_samples:
#                 samps, theta = self.sample_around(theta, jump)
#                 how_many = len(samps)
#                 if cur + how_many > n_samples:
#                     how_many = n_samples - cur
#                 if how_many > 0:
#                     samples[cur:cur + how_many, :] = samps[:how_many, :]
#                     cur += how_many
#                     pbar.update(how_many)  # Update the progress bar

#         if keep:
#             self.samples = samples

#         return samples



    # def sample_multimodal(
    #     self,
    #     n_samples: int,                     # number of starting centres
    #     jump: int = 100_000,
    #     keep: bool = True,
    #     k: int = 10,
    #     T: float = 10.0                     # exploration threshold parameter
    # ):
    #     """
    #     Draw `n_samples` points by running `k` independent short explorations.
    #     Threshold of exploration is log_prob of current centre - T.

    #     If self.samples is not empty, use those as centres.
    #     Each sub-run starts from a different seed drawn with `sample_one`.

    #     NOTE: correctness hinges on `sample_one` / `sample_around`
    #     being proper rejection samplers (see §2 below).
    #     """
    #     assert k >= 1, "k must be at least 1"
    #     per_chain = math.ceil(n_samples / k)      # sub-samples per centre
    #     all_chunks = []
    #     if len(self.samples) > 0:
    #         init_thetas = self.samples[torch.randint(0, len(self.samples), (k,))]
    #     else:
    #         init_thetas = self.rejection_sample(k, jump, keep)
    #     with tqdm(total=n_samples, desc="Sampling") as bar:
    #         for j in range(k):
    #             # 1) independent seed
    #             theta = init_thetas[j]
    #             chunk = torch.empty((per_chain, self.theta_dim))
    #             cur = 0
    #             # Set threshold for exploration
    #             threshold = self.log_prob(theta) - T
    #             while cur < per_chain:
    #                 samps, new_theta = self.sample_around(theta, jump)
    #                 # Only keep samples above the threshold
    #                 mask = self.log_prob(samps) > threshold
    #                 filtered_samps = samps[mask]
    #                 take = min(per_chain - cur, filtered_samps.size(0))
    #                 if take:
    #                     chunk[cur : cur + take] = filtered_samps[:take]
    #                     cur += take
    #                     bar.update(take)
    #                 theta = new_theta
    #             all_chunks.append(chunk)
    #     samples = torch.cat(all_chunks, dim=0)[:n_samples]   # exact count
    #     if keep:
    #         self.samples = samples
    #     return samples

    # def sample_via_importance(self, n_draws=100_000, n_final=10_000, proposal='prior', 
    #                           temperature=None, excess_quantile=0.3):
    #     """
    #     Fast sampling using SIR with 'Oversample & Prune' strategy.
        
    #     Args:
    #         excess_quantile (float): Fraction of extra samples to draw and then discard.
    #                                  e.g., 0.05 means sample 105% of n_final, then drop the worst 5%.
    #     """
    #     n_reps = len(self.Xs)
    #     temperature = temperature if temperature is not None else self.temp
        
    #     # --- 1. SAMPLE FROM PROPOSAL ---
    #     if proposal == 'prior':
    #         theta_pool = self.prior.sample((n_draws,))
    #         log_proposal = self.prior.log_prob(theta_pool)
            
    #     elif proposal == 'mixture':
    #         n_per_rep = n_draws // n_reps
    #         proposals_list = []
            
    #         for i in tqdm(range(n_reps), desc="Proposal Sampling"):
    #             if self.amortized_posterior is not None:
    #                 post_obj = self.amortized_posterior
    #                 if hasattr(post_obj, 'set_default_x'):
    #                     post_obj.set_default_x(self.Xs[i])
    #             else:
    #                 post_obj = self.posterior_list[i]
                
    #             theta_i = post_obj.sample((n_per_rep,), show_progress_bars=False)
    #             proposals_list.append(theta_i)
                
    #         theta_pool = torch.cat(proposals_list, dim=0)
            
    #         # Calculate Mixture Density q(theta)
    #         log_likelihoods_for_q = []
    #         for i in range(n_reps):
    #             if self.amortized_posterior is not None:
    #                 post_obj = self.amortized_posterior
    #                 if hasattr(post_obj, 'set_default_x'):
    #                     post_obj.set_default_x(self.Xs[i])
    #             else:
    #                 post_obj = self.posterior_list[i]
    #             log_likelihoods_for_q.append(post_obj.log_prob(theta_pool))
            
    #         log_L_q_matrix = torch.stack(log_likelihoods_for_q, dim=0)
    #         log_proposal = torch.logsumexp(log_L_q_matrix, dim=0) - np.log(n_reps)

    #     # --- 2. EVALUATE TARGET ---
    #     log_prior = self.prior.log_prob(theta_pool)
        
    #     log_likelihoods = []
    #     for i in tqdm(range(n_reps), desc="Sampling"):
    #         if self.amortized_posterior is not None:
    #             post_obj = self.amortized_posterior
    #             if hasattr(post_obj, 'set_default_x'):
    #                 post_obj.set_default_x(self.Xs[i])
    #         else:
    #             post_obj = self.posterior_list[i]

    #         ll = post_obj.log_prob(theta_pool)
    #         log_likelihoods.append(ll)
            
    #     log_L_matrix = torch.stack(log_likelihoods, dim=0)
        
    #     # --- 3. CALCULATE WEIGHTS ---
    #     eps_tensor = torch.as_tensor(self.epsilon, device=log_L_matrix.device, dtype=log_L_matrix.dtype)
        
    #     # A. Robust Likelihood
    #     log_L_robust = torch.logaddexp(log_L_matrix, eps_tensor)
        
    #     # B. Prior Correction with Inf handling
    #     sum_log_lik = log_L_robust.sum(dim=0)
    #     prior_term = (n_reps - 1) * log_prior
        
    #     log_target_unnorm = sum_log_lik - prior_term
    #     log_target_unnorm[torch.isinf(log_prior)] = -float('inf')
        
    #     # C. Importance Weights
    #     log_weights = log_target_unnorm - log_proposal
    #     log_weights = torch.nan_to_num(log_weights, nan=-float('inf'))
        
    #     # --- TEMPERING ---
    #     if temperature != 1.0:
    #         log_weights = log_weights / temperature

    #     # Normalize weights for resampling
    #     weights = torch.softmax(log_weights, dim=0)
        
    #     # --- 4. RESAMPLING WITH OVERSAMPLING & PRUNING ---
    #     if weights.sum() == 0 or torch.isnan(weights).any():
    #         print("Warning: All samples rejected. Returning prior samples.")
    #         final_samples = self.prior.sample((n_final,))
    #         ess = 0.0
    #     else:
    #         ess = 1.0 / torch.sum(weights ** 2)
            
    #         # Step A: Oversample (e.g., 105% of n_final)
    #         n_safe = int(n_final * (1 + excess_quantile))
    #         indices = torch.multinomial(weights, n_safe, replacement=True)
            
    #         # Step B: Retrieve the original unnormalized weights of the SELECTED samples
    #         # We want to drop the ones that had the worst weights, even if they got lucky and were picked
    #         selected_log_w = log_weights[indices]
            
    #         # Step C: Keep only the top n_final samples
    #         # torch.topk works largest->smallest
    #         best_indices_local = torch.topk(selected_log_w, n_final).indices
            
    #         # Map back to pool indices
    #         final_indices = indices[best_indices_local]
    #         final_samples = theta_pool[final_indices]
        
    #     self.samples = final_samples
    #     return final_samples, weights, ess.item()
    
    