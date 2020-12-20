"""
`pystruct`-style implementation of the CRF.

Reference: https://github.com/pystruct/pystruct
"""
from typing import List, Tuple

import numpy as np


NEG_INF = -np.inf


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
                List of numpy vector of length `n_nodes`.
                The number of unique values should match `n_labels`.
                Labels should be encoded as integers of 0 to `n_labels - 1`.
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
        # `n_labels * n_features` is number of parameters for unary potentials.
        # `n_labels ** 2` is number of parameters for pair-wise potentials
        # between labels.
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
                      x: np.ndarray,
                      y: np.ndarray) -> np.ndarray:
        """
        Feature vector of length `size_joint_feature`.
        Contains the unary and pairwise features.

        Args:
            x:
                numpy array of shape (n_nodes, n_features).
            y:
                numpy vector of length `n_nodes`.
                Labels should be encoded as integers of 0 to `n_labels - 1`.
        """
        edges = make_chain_edges(x)
        n_nodes = x.shape[0]

        # One-hot encode label for each node.
        y_one_hot = np.zeros((n_nodes, self.n_labels), dtype=np.int)
        y_one_hot[np.arange(n_nodes), y] = 1

        # Pairwise features of shape (n_labels, n_labels).
        # Stores the count of label-label transitions observed in `y`.
        pairwise = y_one_hot[edges[:, 0]].T @ y_one_hot[edges[:, 1]]

        # Unary features of shape (n_labels, n_features).
        # If the features are one-hot encodings, this stores the count
        # of feature-label correlation observed in `x` and `y`.
        unary = y_one_hot.T @ x

        # Unary and pairwise features are ravel into an 1-D feature vector.
        joint_feature_vector = np.hstack([unary.ravel(), pairwise.ravel()])
        return joint_feature_vector


    def batch_joint_feature(self,
                            X: List[np.ndarray],
                            Y: List[np.ndarray]) -> np.ndarray:
        """
        Summed feature vector across data batch of length `size_joint_feature`

        Called by learner during `fit`.

        Args: see `initialize` method.
        """
        summed_joint_feature = np.zeros(self.size_joint_feature)
        for x, y in zip(X, Y):
            summed_joint_feature += self.joint_feature(x, y)
        return summed_joint_feature

    def loss(self,
             y: np.ndarray,
             y_hat: np.ndarray) -> int:
        """
        Hamming loss, i.e. count of nodes where `y != y_hat`.

        Args:
            y: see `joint_feature` method.
            y_hat: Labels prediction. numpy vector of length `n_nodes`.
        """
        if self.class_weight:
            return np.sum(self.class_weight[y] * (y != y_hat))
        return np.sum(y != y_hat)

    def batch_loss(self,
                   Y: List[np.ndarray],
                   Y_hat: List[np.ndarray]) -> List[int]:
        """
        Loss of each row in the data batch.

        Args:
            Y: see `initialize` method.
            Y_hat:
                List of labels prediction,
                which are numpy vectors of length `n_nodes`.
        """
        return [self.loss(y, y_hat) for y, y_hat in zip(Y, Y_hat)]

    def inference(self,
                  x: np.ndarray,
                  w: np.ndarray) -> np.ndarray:
        """
        Inference for x using parameters w.

        Finds (approximately)
        argmin_y_hat w @ joint_feature(x, y_hat) + loss(y, y_hat)

        Returns labels prediction as numpy vector of length `n_nodes`.

        Args:
            x: see `joint_feature` method.
            w:
                Weight parameters for the CRF energy function.
                numpy vector of length `size_joint_feature`.
        """
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(w)

        return inference_viterbi(unary_potentials, pairwise_potentials)
    
    def loss_augmented_inference(self,
                                 x: np.ndarray,
                                 y: np.ndarray,
                                 w: np.ndarray) -> np.ndarray:
        """
        Loss-augmented inference for x relative to y using parameters w.

        Finds (approximately)
        argmin_y_hat w @ joint_feature(x, y_hat) + loss(y, y_hat)

        Returns labels prediction as numpy vector of length `n_nodes`.

        Args:
            x: see `joint_feature` method.
            y:
                Ground truth labels for loss calculation.
                See `joint_feature` method. 
            w: see `inference` method.
        """
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(w)

        if self.class_weight:
            self.loss_augment_unaries(unary_potentials, y, w)

        return inference_viterbi(unary_potentials, pairwise_potentials)

    def batch_loss_augmented_inference(self,
                                       X: List[np.ndarray],
                                       Y: List[np.ndarray],
                                       w: np.ndarray):
        """
        Make inference for each row of data given w as model parameter.

        Called by learner during `fit`.

        Args:
            X: see `initialize` method.
            Y: see `initialize` method.
            w: see `loss_augmented_inference` method.
        """
        return [self.loss_augmented_inference(x, y, w) for x, y in zip(X, Y)]

    def _get_unary_potentials(self,
                              x: np.ndarray,
                              w: np.ndarray) -> np.ndarray:
        """
        Computes unary potentials for given x and w.

        Returns numpy array of shape (n_nodes, n_labels).

        Args:
            x: see `joint_feature` method.
            w: see `loss_augmented_inference` method.
        """
        unary_params = (
            w[:self.n_labels * self.n_features]
            .reshape(self.n_labels, self.n_features)
        )
        return x @ unary_params.T

    def _get_pairwise_potentials(self,
                                 w: np.ndarray) -> np.ndarray:
        """
        Computes pairwise potentials for given w.

        Returns numpy array of shape (n_labels, n_labels).

        Args:
            w: see `loss_augmented_inference` method.
        """
        pairwise_params = (
            w[self.n_labels * self.n_features:]
            .reshape(self.n_labels, self.n_labels)
        )
        return pairwise_params

    def loss_augment_unaries(self,
                             unary_potentials: np.ndarray,
                             y: np.ndarray,
                             w: np.ndarray) -> None:
        """
        Add class weights to unary potentials.

        Only add class weight when the label is not equal to the observed y
        for the node.

        Args:
            unary_potentials: numpy array of shape (n_nodes, n_labels).
            y: see `joint_feature` method.
            w: see `loss_augmented_inference` method.
        """
        n_nodes = unary_potentials.shape[0]
        weight_to_add = np.tile(self.class_weight, (n_nodes, 1))
        mask = (
            np.array([True if label == y[i] else False
                      for i in range(n_nodes)
                      for label in range(self.n_labels)])
            .reshape(n_nodes, self.n_labels)
        )
        weight_to_add[mask] = 0.
        unary_potentials += weight_to_add


def make_chain_edges(x: np.ndarray) -> np.ndarray:
    """
    Linear chain edges of shape (n_nodes - 1, 2).

    Each row denotes an edge connection between two vertexes representing
    the graph.

    For example, if `n_nodes = 3`, this should return:
    ```
    array([[0, 1],
           [1, 2]])
    ```

    Args:
        x:
            numpy array of shape (n_nodes, n_features).
    """
    n_nodes = x.shape[0]
    vertices = np.arange(n_nodes)
    edge_begin = vertices[:-1, np.newaxis]
    edge_end = vertices[1:, np.newaxis]
    edges = np.concatenate([edge_begin, edge_end], axis=1)
    return edges


def inference_viterbi(unary_potentials: np.ndarray,
                      pairwise_potentials: np.ndarray) -> np.ndarray:
    """
    Max-product inference via first-order Viterbi algorithm.

    Returns labels prediction as numpy vector of length `n_nodes`.

    Args:
        unary_potentials: numpy array of shape (n_nodes, n_labels).
        pairwise_potentials: numpy array of shape (n_labels, n_labels).
    """
    n_nodes = unary_potentials.shape[0]
    n_labels = unary_potentials.shape[1]
    max_values = np.zeros((n_nodes, n_labels))
    max_indices = np.zeros((n_nodes, n_labels))

    # Forward recursion.

