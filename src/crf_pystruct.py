"""
CRF implemented like `pystruct`
"""
from typing import List

import numpy as np

class ChainCRF():
    """
    Directed linear-chain CRF.
    """
    def __init__(self,
                 *,
                 n_labels: int = None,
                 n_features: int = None,
                 class_weight: List[float] = None) -> None:
        """
        Args:
            n_labels:
                Number of unique labels in `Y`.
                Inferred from data in `initialize` if not provided.
            n_features:
                Number of features in `X`.
                Inferred from data in `initialize` if not provided.
            class_weight:
                Weighting of each class for loss calculation.
                Inferred from data in `initialize` if not provided.
        """
        self.n_labels = n_labels
        self.n_features = n_features
        self.class_weight = class_weight

    def initialize(self,
                   X: List[np.ndarray],
                   Y: List[np.ndarray]) -> None:
        """
        Initialize method that will be called by the learner during `fit`.

        Args:
            X:
                List of numpy array of shape (n_nodes, n_features).
            Y:
                List of numpy array of shape (n_nodes, ).
                The number of unique values should match `n_labels`.
        """
        # Infer `n_features` from data.
        n_features = X[0].shape[1]
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError(f"Expected {self.n_features} features, "
                             f"got {n_features}.")

        # Infer `n_labels` from data.
        possible_states = np.unique(np.hstack(Y))
        n_labels = len(possible_states)
        if self.n_labels is None:
            self.n_labels = n_labels
        elif self.n_labels != n_labels:
            raise ValueError(f"Expected {self.n_labels} labels, "
                             f"got {n_labels}.")

        # Size of `joint_feature` for directed graphs.
        # `n_labels * n_features` is number of parameters for unary potentials
        # `n_labels ** 2` is number of parameters for pair-wise potentials
        # between labels
        self.size_joint_feature = self.n_labels * self.n_features + self.n_labels ** 2

        # Infer `class_weight` from data.
        if self.class_weight is not None:
            if len(self.class_weight) != self.n_labels:
                raise ValueError("`class_weight` must have length of `n_labels`."
                                 f"{self.class_weight=}, {self.n_labels=}")
            self.class_weight = np.array(self.class_weight)
            self.uniform_class_weight = False
        else:
            self.class_weight = np.ones(self.n_labels)
            self.uniform_class_weight = True

    def joint_feature(self,
                      x,
                      y) -> np.ndarray:
        """
        Feature vector associated with instance (x, y).

        Called by `batch_joint_feature`.

        Args:
            x:
            y:
        """

    def batch_joint_feature(self,
                            X,
                            Y):

    def batch_lose(self):

    def batch_loss_augmented_inference(self):
