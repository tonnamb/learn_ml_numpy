"""
`pystruct`-style implementation of the CRF.

Reference: https://github.com/pystruct/pystruct

Example output:
```
$ make crf
pyvenv/bin/python -i src/crf_pystruct.py
Training for 1,000 iterations
Warning: Maximum number of iterations has been exceeded.
Score after 1,000 iterations: 0.455
                   x                    y               y_pred
          Confidence                 B-NP                 I-NP
                  in                 B-PP                    O
                 the                 B-NP               B-ADJP
               pound                 I-NP                 I-NP
                  is                 B-VP                 I-NP
              widely                 I-VP               B-ADVP
            expected                 I-VP                 I-NP
                  to                 I-VP                 I-NP
                take                 I-VP                 I-NP
             another                 B-NP                 I-NP
               sharp                 I-NP                 I-NP
                dive                 I-NP                 I-NP
                  if               B-SBAR                 B-PP
               trade                 B-NP                 I-NP
             figures                 I-NP                 I-NP
                 for                 B-PP                 B-PP
           September                 B-NP                 I-NP
                   ,                    O                 I-NP
                 due               B-ADJP                 I-NP
                 for                 B-PP                 B-PP
             release                 B-NP                 I-NP
            tomorrow                 B-NP                 I-NP
                   ,                    O                 I-NP
                fail                 B-VP                 I-NP
                  to                 I-VP                 I-NP
                show                 I-VP                    O
                   a                 B-NP               B-ADJP
         substantial                 I-NP                 I-NP
         improvement                 I-NP                 I-NP
                from                 B-PP                 B-PP
                July                 B-NP                 I-NP
                 and                 I-NP                 I-NP
              August                 I-NP                 I-NP
                  's                 B-NP                 I-NP
         near-record                 I-NP                 I-NP
            deficits                 I-NP                 B-PP
                   .                    O                    O
Training for 3,000 iterations
Warning: Maximum number of iterations has been exceeded.
Score after 3,000 iterations: 0.683
Training for 5,000 iterations
Warning: Maximum number of iterations has been exceeded.
Score after 5,000 iterations: 0.785
                   x                    y               y_pred
          Confidence                 B-NP                 B-NP
                  in                 B-PP                 B-PP
                 the                 B-NP                 B-NP
               pound                 I-NP                 I-NP
                  is                 B-VP                 B-VP
              widely                 I-VP                 B-NP
            expected                 I-VP               B-ADVP
                  to                 I-VP                 B-VP
                take                 I-VP                 I-VP
             another                 B-NP                 B-NP
               sharp                 I-NP                 I-NP
                dive                 I-NP                 I-NP
                  if               B-SBAR                 B-NP
               trade                 B-NP                 I-NP
             figures                 I-NP                 I-NP
                 for                 B-PP                 B-PP
           September                 B-NP                 B-NP
                   ,                    O                 I-VP
                 due               B-ADJP                 B-NP
                 for                 B-PP                 B-PP
             release                 B-NP                 B-NP
            tomorrow                 B-NP                 I-NP
                   ,                    O                    O
                fail                 B-VP               I-ADVP
                  to                 I-VP                 B-VP
                show                 I-VP                 I-VP
                   a                 B-NP                 B-NP
         substantial                 I-NP                 I-NP
         improvement                 I-NP                 I-NP
                from                 B-PP                 B-PP
                July                 B-NP                 B-NP
                 and                 I-NP                 I-NP
              August                 I-NP                 I-NP
                  's                 B-NP                 B-NP
         near-record                 I-NP                 I-NP
            deficits                 I-NP                 I-NP
                   .                    O                    O
```

NOTE: The optimization step is still pretty slow as we just using the downhill
simplex algorithm. It takes pretty long (~1000 iterations for 1 hour) to even
train with 100 samples.
"""
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import fmin

from crf import read_conll_2000_chunking, CoNLLChunking


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
                                 f"`class_weight={self.class_weight}`, "
                                 f"`n_labels={self.n_labels}.`")
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

        NOTE: currently not in used.

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
        if self.class_weight is not None:
            return np.sum(self.class_weight[y] * np.not_equal(y, y_hat))
        return np.sum(np.not_equal(y, y_hat))

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

    def max_loss(self,
                 y: np.ndarray) -> float:
        """
        Maximum possible loss on y for macro averages.

        Args:
            y: see `joint_feature` method.
        """
        if self.class_weight is not None:
            return np.sum(self.class_weight[y])
        return float(len(y))

    def inference(self,
                  x: np.ndarray,
                  w: np.ndarray) -> np.ndarray:
        """
        Inference for x using parameters w.

        Finds (approximately)
        argmax_y_hat w @ joint_feature(x, y_hat)

        Called by learner during `fit`.

        Returns labels prediction as numpy vector of length `n_nodes`.

        NOTE: this finds the argmax of the combined potentials, which
        means the 'potentials' are more of a 'likelihood' than a 'potential'
        in the thermodynamic sense.

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
        argmax_y_hat w @ joint_feature(x, y_hat) + loss(y, y_hat)

        Returns labels prediction as numpy vector of length `n_nodes`.

        NOTE: currently not in-used.

        Args:
            x: see `joint_feature` method.
            y:
                Ground truth labels for loss calculation.
                See `joint_feature` method. 
            w: see `inference` method.
        """
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(w)

        if self.class_weight is not None:
            self.loss_augment_unaries(unary_potentials, y, w)

        return inference_viterbi(unary_potentials, pairwise_potentials)

    def batch_loss_augmented_inference(self,
                                       X: List[np.ndarray],
                                       Y: List[np.ndarray],
                                       w: np.ndarray):
        """
        Make inference for each row of data given w as model parameter.

        NOTE: currently not in-used.

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

        NOTE: currently not in-used.

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

    # Forward pass.
    unary_lag = np.tile(unary_potentials[:-1, np.newaxis, :], (1, n_labels, 1))
    unary_lead = np.tile(unary_potentials[1:, :, np.newaxis], (1, 1, n_labels))
    # Transpose is needed to make candidates for each node be transitions that
    # varies in the lag node, and constant in the lead node.
    # For example, A -> A, B -> A, C -> A.
    pairwise = np.tile(pairwise_potentials.T[np.newaxis, :, :], (n_nodes - 1, 1, 1))
    candidates = unary_lag + unary_lead + pairwise

    max_values = np.max(candidates, axis=2)
    max_indices = np.argmax(candidates, axis=2)

    # Backward pass.
    y_pred = np.zeros(n_nodes, dtype=np.intp)
    if n_nodes > 1:
        y_pred[-1] = max_values[n_nodes - 2, :].argmax()
    for node_idx in range(n_nodes - 2, -1, -1):
        y_pred[node_idx] = max_indices[node_idx, y_pred[node_idx + 1]]
    return y_pred


class DownhillSimplexLearner():
    """
    Learn weight parameters using the downhill simplex algothm to minimize
    the loss function.
    """
    def __init__(self, model: ChainCRF):
        """
        Args:
            model: model instance, e.g. ChainCRF instance
        """
        self.model = model
        self.w = None

    def fit(self,
            X: List[np.array],
            Y: List[np.array],
            *,
            maxiter: int = 1000):
        """
        Fit to data.

        Args:
            X: see `ChainCRF.initialize` method.
            Y: see `ChainCRF.initialize` method.
            maxiter: max iterations of the optimization.
        """
        self.model.initialize(X, Y)
        def regularized_loss(w):
            """Function to pass into `scipy.optimize.fmin`"""
            reg_loss = 0
            for x, y in zip(X, Y):
                y_hat = self.model.inference(x, w)
                reg_loss += self.model.loss(y, y_hat)
            reg_loss /= float(len(X))
            reg_loss += np.sum(w ** 2)
            return reg_loss
        self.w = np.random.normal(loc=1.0,
                                  scale=1e-5,
                                  size=self.model.size_joint_feature)
        self.w = fmin(regularized_loss, x0=self.w, maxiter=maxiter)
        return self

    def predict(self,
                X: List[np.array]) -> List[np.array]:
        """
        Predict labels on examples in X.

        Args:
            X: see `ChainCRF.initialize` method.
        """
        return [self.model.inference(x, self.w) for x in X]
    
    def score(self,
              X: List[np.array],
              Y: List[np.array]) -> float:
        """
        Compute score as 1 - loss over whole data set.

        Args:
            X: see `ChainCRF.initialize` method.
            Y: see `ChainCRF.initialize` method.
        """
        losses = self.model.batch_loss(Y, self.predict(X))
        max_losses = [self.model.max_loss(y) for y in Y]
        return 1. - np.sum(losses) / float(np.sum(max_losses))


def sample(data: CoNLLChunking, n: int) -> CoNLLChunking:
    """
    Sample n instances of the data.

    Args:
        data: CoNLL Chunking data set.
        n: Number of instances to sample.
    """
    out = CoNLLChunking()
    for i in range(n):
        out.word.append(data.word[i])
        out.pos.append(data.pos[i])
        out.label.append(data.label[i])
    return out


def prepare(data: CoNLLChunking,
            *,
            vocab_to_int: Dict[str, int] = None,
            pos_to_int: Dict[str, int] = None,
            label_to_int: Dict[str, int] = None,
            ) -> Tuple[List[np.ndarray],
                       List[np.ndarray],
                       Dict[str, int],
                       Dict[str, int],
                       Dict[str, int]]:
    """
    Prepare CoNLL chunking data for the CRF model.

    Args:
        data: CoNLL 2000 Chunking data
        vocab_to_int, pos_to_int, label_to_int:
            When mappings are passed in, it will use it instead of building
            new mappings.

    Returns:
        X:
            Shape of (n_samples, n_nodes, n_features), where the features
            are concatenated one-hot vectors of the word and part-of-speech.
        Y:
            Shape of (n_samples, n_nodes).
        vocab_to_int:
            Mapping of word to index.
        pos_to_int:
            Mapping of part-of-speech to index.
        label_to_int:
            Mapping of label to index.
    """
    if vocab_to_int is None and pos_to_int is None:
        # Build vocab and pos mappings.
        vocab_to_int = {}
        pos_to_int = {}
        idx_feature = 0
        for word_sample, pos_sample in zip(data.word, data.pos):
            for word, pos in zip(word_sample, pos_sample):
                if word not in vocab_to_int:
                    vocab_to_int[word] = idx_feature
                    idx_feature += 1
                if pos not in pos_to_int:
                    pos_to_int[pos] = idx_feature
                    idx_feature += 1

    # Build one-hot vectors
    n_vocab = len(vocab_to_int)
    n_pos = len(pos_to_int)
    n_features = n_vocab + n_pos
    X = []
    for sample, (word_sample, pos_sample) in enumerate(zip(data.word,
                                                           data.pos)):
        X.append(np.zeros((len(word_sample), n_features)))
        for word_idx, (word, pos) in enumerate(zip(word_sample, pos_sample)):
            vocab_int = vocab_to_int[word]
            pos_int = pos_to_int[pos]
            X[sample][word_idx, vocab_int] = 1
            X[sample][word_idx, pos_int] = 1
    if label_to_int is None:
        label_to_int = {}
        idx_label = 0
        for sample_idx, sample in enumerate(data.label):
            for label in sample:
                if label not in label_to_int:
                    label_to_int[label] = idx_label
                    idx_label += 1
    n_labels = len(label_to_int)
    Y = []
    for sample_idx, sample in enumerate(data.label):
        n_nodes = len(sample)
        Y.append(np.zeros(n_nodes, dtype=np.intp))
        for node_idx, label in enumerate(sample):
            label_int = label_to_int[label]
            Y[sample_idx][node_idx] = label_int
    return X, Y, vocab_to_int, pos_to_int, label_to_int


def demo_prediction(data_idx: int,
                    learner: DownhillSimplexLearner,
                    X: List[np.ndarray],
                    sample: CoNLLChunking,
                    int_to_label: Dict[int, str]) -> None:
    """
    Demonstrate a prediction for 1 data point.

    Args:
        data_idx: Index of the 1 data point.
        learner: Trained learner.
        X: see `ChainCRF.initialize` method.
        sample: Sampled CoNLL chunking data set.
        int_to_label: Mapping of integer to label.
    """
    y_pred = learner.predict([X[data_idx]])
    pred_label = [int_to_label[number] for number in y_pred[0]]
    print('{: >20} {: >20} {: >20}'.format('x', 'y', 'y_pred'))
    for x, y, y_pred in zip(sample.word[data_idx],
                            sample.label[data_idx],
                            pred_label):
        print(f'{x: >20} {y: >20} {y_pred: >20}')


if __name__ == "__main__":
    train = read_conll_2000_chunking('data/conll_2000_chunking_train.txt')
    train_sample = sample(train, 10)
    X_train, Y_train, vocab_to_int, pos_to_int, label_to_int = prepare(train_sample)
    crf = ChainCRF()
    learner = DownhillSimplexLearner(crf)

    print('Training for 1,000 iterations')
    learner.fit(X_train, Y_train, maxiter=1000)
    score_1 = learner.score(X_train, Y_train)
    print(f'Score after 1,000 iterations: {score_1:.3f}')

    int_to_label = {number: label for label, number in label_to_int.items()}
    print('Demo prediction:')
    demo_prediction(0, learner, X_train, train_sample, int_to_label)

    print('Training for 3,000 iterations')
    learner.fit(X_train, Y_train, maxiter=3000)
    score_2 = learner.score(X_train, Y_train)
    print(f'Score after 3,000 iterations: {score_2:.3f}')

    print('Training for 5,000 iterations')
    learner.fit(X_train, Y_train, maxiter=5000)
    score_2 = learner.score(X_train, Y_train)
    print(f'Score after 5,000 iterations: {score_2:.3f}')

    print('Demo prediction:')
    demo_prediction(0, learner, X_train, train_sample, int_to_label)
