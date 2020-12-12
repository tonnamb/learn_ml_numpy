"""
CRF implementation and demo.
"""
import random
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from sympy.utilities.iterables import variations


class LinearChainCRF:
    """
    p(y|x) = 1/Z product_t phi_t_yt_yt-1_xt
    phi_t_yt_yt-1_xt = exp sum_k theta_k * f_k_yt_yt-1_xt

    Reference:
        Sutton and McCallum 2011
        https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf
    """
    BEGIN_LABEL = '<BEGIN>'
    BEGIN_LABEL_IDX = -1

    def __init__(self,
                 labels: Tuple[str],
                 n_features: int) -> None:
        """
        self.theta.shape is (n_features + n_transitions)
        """
        self.n_features = n_features
        self.labels = labels
        self.n_labels = len(labels)
        self.label_map = {idx: label for idx, label in enumerate(labels)}
        self.label_indexes = self.label_map.keys()
        self.full_label_map = self.label_map.copy()
        self.full_label_map[self.BEGIN_LABEL_IDX] = self.BEGIN_LABEL

        # transitions are defined as tuple of (y_tlag, y_t)
        begin_transition = [(self.BEGIN_LABEL, label) for label in labels]
        transitions = (begin_transition +
                       list(variations(labels, 2, repetition=True)))
        self.transition_map = {transition: idx
                               for idx, transition in enumerate(transitions)}
        self.x_transition_map = {idx: one_hot for idx, one_hot
                                 in enumerate(np.eye(len(transitions)))}
        self.theta = np.random.standard_normal(n_features + len(transitions))

    def fit(self,
            X: List[List[np.ndarray]],
            y: List[List[str]],
            *,
            n_epochs: int = 5,
            batch_size: int = 10,
            sigma: float = 1.0,
            m_0: float = 0.1) -> float:
        """
        Parameter estimation via maximum likelihood using stochastic gradient
        descent.

        Args:
            X.shape is (n_samples, n_timesteps, n_features)
            y.shape is (n_samples, n_timesteps)

        Returns the log likelihood of the last iteration.

        Paragraph above Eq. (4.19)
        Z = b_0_y0 or sum_i a_T_i

        Eq. (5.21)
        theta_m = theta_m-1 + lr_m dL_i/dtheta_m-1

        Eq. (5.22)
        lr_m = 1 / (sigma^2 * ( m_0 + m ))
        """
        n_samples = len(X)
        n_iterations = int(n_samples * n_epochs / batch_size)
        iteration_num = 1
        while iteration_num <= n_iterations:
            batch = random.sample(list(zip(X, y)), batch_size)
            X_batch = [x for x, _ in batch]
            y_batch = [y for _, y in batch]
            alphas = [self.alpha_forward(x) for x in X_batch]
            betas = [self.beta_backward(x) for x in X_batch]
            Zs = [b[0, self.BEGIN_LABEL_IDX] for b in betas]
            log_likelihood = self.log_likelihood(X_batch, y_batch, Zs, sigma)
            dL_dtheta = self.dL_dtheta(X_batch,
                                       y_batch,
                                       alphas,
                                       betas,
                                       Zs,
                                       sigma,
                                       batch_size,
                                       n_samples)
            lr_m = 1 / (sigma ** 2 * (m_0 * iteration_num))
            self.theta = self.theta + lr_m * dL_dtheta
            if iteration_num % batch_size == 0:
                epoch_num = int(iteration_num / batch_size)
                print(f"Epoch {epoch_num}/{n_epochs}")
                print(f"Log likelihood: {log_likelihood:.3f}")
            iteration_num += 1
        return log_likelihood

    def predict(self,
                X: List[List[np.ndarray]]
                ) -> List[List[str]]:
        """
        Viterbi algorithm to predict the sequence of labels.

        Args:
            X.shape is (n_samples, n_timesteps, n_features)

        Returns:
            y.shape is (n_samples, n_timesteps)

        Eq. (4.15)
        d_t_j = max_y phi_t_j_yt-1 product_t-1 phi_t_yt_yt-1

        Eq. (4.16)
        d_t_j = max_i phi_t_j_i * d_t-1_i

        Once d_t_j are computed, assignments can be done by backward recursion
            y_T = argmax_i d_T_i
            y_t = argmax_i phi_t_yt+1_i_xt * d_t_i

        Example:
            if n_labels = 2, n_timesteps = 3
            d_1_0 = phi_1_0_y0
            d_1_1 = phi_1_1_y0
            d_2_0 = max(phi_2_0_0 * phi_1_0_y0,
                        phi_2_0_1 * phi_1_1_y0)
            d_2_1 = max(phi_2_1_0 * phi_1_0_y0,
                        phi_2_1_1 * phi_1_1_y0)
        """
        y = []
        for sample_idx, x_sample in enumerate(X):
            d_t_j = {}
            for t, x_t in enumerate(x_sample):
                for y_t in self.label_indexes:
                    if t == 0:
                        d_t_j[(0, y_t)] = self.factor(y_t,
                                                      self.BEGIN_LABEL_IDX,
                                                      x_t)
                    else:
                        d_t_j[(t, y_t)] = max(phi * d_t_j[(t - 1, y_tlag)]
                                              for y_tlag in self.label_indexes
                                              for phi in self.factor(y_t,
                                                                     y_tlag,
                                                                     x_t))
            y.append([])
            for t in reversed(range(len(x_sample))):
                t_max = len(x_sample) - 1
                if t == t_max:
                    y[sample_idx].append(
                        np.argmax(d_t_j[(t, y_t)]
                                  for y_t in self.label_indexes)
                    )
                else:
                    y[sample_idx].append(
                        np.argmax(
                            # `y[sample_idx][-1]` is the predicted y_t+1
                            self.factor(y[sample_idx][-1],
                                        y_t,
                                        x_sample[t + 1]) * d_t_j[(t, y_t)]
                            for y_t in self.label_indexes
                        )
                    )
            # Reverse `y` since it was appended in reverse order.
            # Also, map label_idx back to label string.
            y[sample_idx] = reversed(self.full_label_map[y[sample_idx]])
        return y

    @lru_cache(maxsize=10000)
    def x_concat(self,
                 y_t: int,
                 y_tlag: int,
                 x_t: np.ndarray) -> np.ndarray:
        """
        Construct the x_concat (also referred to as f_k_yt_yt-1_xt).

        Args:
            x_t.shape is (n_features)

        Returns:
            x_concat.shape is (n_features + n_transitions)
        """
        transition_idx = self.transition_map[(y_tlag, y_t)]
        x_transition = self.x_transition_map[transition_idx]
        x_concat = np.concatenate([x_t, x_transition])
        return x_concat

    @lru_cache(maxsize=10000)
    def factor(self,
               y_t: int,
               y_tlag: int,
               x_t: np.ndarray) -> float:
        """
        Calculate the factor (phi) given the yt, yt-1 and xt.

        Args:
            x_t.shape is (n_features)

        Eq. (4.18)
        phi_t_yt_yt-1_xt = exp sum_k theta_k * f_k_yt_yt-1_xt
        """
        x_concat = self.x_concat(y_t, y_tlag, x_t)
        phi = np.exp(np.dot(self.theta, x_concat))
        return phi

    def alpha_forward(self,
                      x_one_sample: List[np.ndarray]
                      ) -> np.ndarray:
        """
        Forward recursion to calculate the alpha variables.

        Args:
            x_one_sample is (n_timesteps, n_features)

        Returns:
            alpha.shape is (n_timesteps, n_labels)

        Eq. (4.5)
        a_t_j = sum_i phi_t_j_i_xt * a_t-1_i
        a_1_j = phi_1_j_y0_x1
        """
        n_timesteps = len(x_one_sample)
        alpha = np.zeros(n_timesteps, self.n_labels)
        for t, x_t in enumerate(x_one_sample):
            for y_t in self.label_indexes:
                if t == 0:
                    alpha[0, y_t] = self.factor(y_t, self.BEGIN_LABEL_IDX, x_t)
                else:
                    alpha[t, y_t] = np.sum([phi * alpha[t - 1, y_tlag]
                                            for y_tlag in self.label_indexes
                                            for phi in self.factor(y_t,
                                                                   y_tlag,
                                                                   x_t)])
        return alpha

    def beta_backward(self,
                      x_one_sample: List[np.ndarray]
                      ) -> np.ndarray:
        """
        Backward recursion to calculate the beta variables.

        Args:
            x_one_sample is (n_timesteps, n_features)

        Returns:
            beta.shape is (n_timesteps, n_labels)

        Eq. (4.9)
        b_t_i = sum_j phi_t+1_j_i_xt+1 * b_t+1_j
        b_T_i = 1
        """
        n_timesteps = len(x_one_sample)
        beta = np.zeros(n_timesteps, self.n_labels)
        for t in reversed(range(n_timesteps)):
            for y_tlag in self.label_indexes:
                if t == (n_timesteps - 1):
                    beta[t, y_tlag] = 1
                elif t > 0:
                    beta[t, y_tlag] = np.sum(
                        [phi * beta[t + 1, y_t]
                         for y_t in self.label_indexes
                         for phi in self.factor(y_t,
                                                y_tlag,
                                                x_one_sample[t + 1])]
                    )
            if t == 0:
                beta[0, self.BEGIN_LABEL_IDX] = np.sum(
                    [phi * beta[t + 1, y_t]
                     for y_t in self.label_indexes
                     for phi in self.factor(y_t,
                                            self.BEGIN_LABEL_IDX,
                                            x_one_sample[t + 1])]
                )
        return beta

    def log_likelihood(self,
                       X: List[List[np.ndarray]],
                       y: List[List[str]],
                       Zs: List[float],
                       sigma: float
                       ) -> float:
        """
        Calculate log likelihood.

        Args:
            X.shape is (n_samples, n_timesteps, n_features)
            y.shape is (n_samples, n_timesteps)
            Zs.shape is (n_samples)

        L = sum_data log p(y|x)
          = sum_data log 1/Z product_t phi_t_yt_yt-1_xt
          = sum_data sum_t log phi_t_yt_yt-1_xt - log Z

        phi_t_yt_yt-1_xt = exp sum_k theta_k * f_k_yt_yt-1_xt

        Eq. (5.4) - log likelihood with L2 ridge regularization
        L = sum_data sum_t sum_k theta_k * f_k_yt_yt-1_xt
            - sum_data log Z
            - sum_k theta_k^2 / (2 * sigma^2)
        L = sum_data sum_t log phi_t_yt_yt-1_xt
            - sum_data log Z
            - sum_k theta_k^2 / (2 * sigma^2)
        """
        log_likelihood = (np.sum(np.square(self.theta)) / (2 * sigma ** 2)
                          - np.sum(np.log(Zs)))
        for x_sequence, y_sequence in zip(X, y):
            for t, x_t in enumerate(x_sequence):
                log_likelihood += np.log(
                    self.factor(y_sequence[t], y_sequence[t - 1], x_t)
                )
        return log_likelihood

    def dL_dtheta(self,
                  X: List[List[np.ndarray]],
                  y: List[List[str]],
                  alphas: List[np.ndarray],
                  betas: List[np.ndarray],
                  Zs: List[float],
                  sigma: float,
                  batch_size: int,
                  n_samples: int) -> np.ndarray:
        """
        Calculate gradient of log likelihood (dL/dtheta).

        Args:
            X.shape is (n_samples, n_timesteps, n_features)
            y.shape is (n_samples, n_timesteps)
            alphas.shape is (n_samples, n_timesteps, n_labels)
            betas.shape is (n_samples, n_timesteps, n_labels)
            Zs.shape is (n_samples)

        Returns:
            dL.shape is (n_features + n_transitions)

        Adapted from Eq. (5.20)
        dL_i/dtheta_k = sum_data sum_t f_k_yt_yt-1_xt
                        - sum_data sum_t sum_y_y' f_k_y_y'_xt * p(y,y'|x)
                        - (theta_k * batch_size) / (N * sigma^2)
        (Eq. (5.20) was for batch_size = 1,
         so added the batch_size multiplication)

        Eq. (4.19)
        p(yt-1,yt|x) = 1/Z * a_t-1_yt-1 * phi_t_yt_yt-1_xt * b_t_yt
        """
        dL = np.zeros_like(self.theta)
        for s_idx, (x_sequence, y_sequence) in enumerate(zip(X, y)):
            for t, x_t in enumerate(x_sequence):
                dL += self.x_concat(y_sequence[t], y_sequence[t - 1], x_t)
                # Using `sum` instead of `np.sum` because the intent is to do
                # element-wise addition between list of `np.ndarray` to form
                # 1 `np.ndarray`
                dL -= sum(self.x_concat(y_t, y_tlag, x_t)
                          * alphas[s_idx][t, y_tlag]
                          * self.factor(y_t, y_tlag, x_t)
                          * betas[s_idx][t, y_t]
                          / Zs[s_idx]
                          for y_t in self.label_indexes
                          for y_tlag in self.label_indexes)
                dL -= self.theta * batch_size / (n_samples * sigma ** 2)
        return dL
