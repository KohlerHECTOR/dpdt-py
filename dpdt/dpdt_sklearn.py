import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import Interval
from sklearn.tree import DecisionTreeClassifier
from .mdp_utils import backward_induction_multiple_zetas, Action, State
from numbers import Integral, Real


class DPDTree(ClassifierMixin, BaseEstimator):
    """
    Dynamic Proorgramming Decision Tree (DPDTree) classifier.

    Parameters
    ----------
    max_depth : int
        The maximum depth of the tree.
    max_nb_trees : int, default=1000
        The maximum number of trees.
    cart_nodes_list : list of int, default=[3]
        List containing the number of leaf nodes for the CART trees at each depth.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.

    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.

    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    mdp : list of list of State
        The Markov Decision Process represented as a list of lists of states,
        where each inner list contains the states at a specific depth.

    zetas : array-like
        Array of zeta values to be used in the computation.

    trees : dict
        A dictionary representing the tree policies. The keys are tuples representing
        the state observation and depth, and the values are the optimal tree
        for each zeta value.

    init_o : array-like
        The initial observation of the MDP.


    Examples
    --------
    >>> from dpdt import DPDTree
    >>> from sklearn import datasets
    >>> from sklearn.tree import DecisionTreeClassifier
    >>>
    >>> X, y = datasets.load_breast_cancer(return_X_y=True)
    >>>
    >>> clf = DPDTree(max_depth=3)
    >>> clf.fit(X, y)
    >>> print(clf.score(X, y))
    >>>
    >>> clf = DecisionTreeClassifier(max_depth=3)
    >>> clf.fit(X,y)
    >>> print(clf.score(X, y))
    """

    _parameter_constraints = {
        "max_depth": [Interval(Integral, 2, None, closed="left")],
        "max_nb_trees": [Interval(Integral, 1, None, closed="left")],
        "cart_nodes_list": ["array-like"],
    }

    def __init__(self, max_depth=3, max_nb_trees=1000, cart_nodes_list=(3,)):
        self.max_nb_trees = max_nb_trees
        self.max_depth = max_depth
        self.cart_nodes_list = cart_nodes_list

    def build_mdp(self):
        """
        Build the Markov Decision Process (MDP) for the trees.

        This method constructs an MDP using a breadth-first search approach. Each node in the tree represents a state in the MDP,
        and actions are determined based on potential splits from a decision tree classifier.

        1. Initialization:
            - Sets `max_depth` to `self.max_depth + 1`.
            - Creates a `root` state with the concatenated minimum and maximum values of `self.X_` with slight offsets.
            - Initializes a `terminal_state` as an array of zeros with a length of twice the number of features in `self.X_`.

        2. Root State:
            - Initializes the `root` state with all samples (`nz` as an array of `True` values).
            - `deci_nodes` is a list of lists where each inner list holds nodes at a certain depth, starting with the `root`.

        3. Breadth-First Search:
            - Uses a breadth-first search to expand nodes up to the specified `max_depth`.
            - For each node at depth `d`, creates a temporary list `tmp` to store the new nodes created at depth `d + 1`.

        4. Node Expansion:
            - For each node at the current depth, calculates the unique classes and their counts for the samples in the node (`node.nz`).
            - Computes the best possible reward (`rstar`) and the action (`astar`) leading to the next state.
            - If further expansion is possible (i.e., depth budget allows and there are at least two classes), initializes a `DecisionTreeClassifier` to determine the splits.
            - Fits the classifier on the samples in the node, and identifies potential splits (features and thresholds).

        5. Action Creation and Transition:
            - For each split, creates an `Action` and determines the left and right child nodes based on the split.
            - Creates the left and right nodes as new states, and adds transitions to the action.
            - If an action has valid transitions, adds it to the current node.

        6. Depth Advancement:
            - If new nodes are created (`tmp` is not empty), adds them to `deci_nodes`, and increments the depth counter (`d`).
            - If no new nodes are created, the process stops.

        Returns
        -------
        deci_nodes : list
            A list of lists, where each inner list contains the decision nodes at a specific depth of the tree.

        Notes
        -----
        This is an implementation of Algortihm 1 from [1]_ .

        References
        ----------

        .. [1] H. Kohler et. al., "Interpretable Decision Tree Search as a Markov Decision Process" arXiv https://arxiv.org/abs/2309.12701.
        """
        max_depth = self.max_depth + 1
        root = State(
            np.concatenate((self.X_.min(axis=0) - 1e-3, self.X_.max(axis=0) + 1e-3)),
            nz=np.ones(self.X_.shape[0], dtype=np.bool_),
        )
        terminal_state = np.zeros(2 * self.X_.shape[1])
        deci_nodes = [[root]]
        d = 0

        while d < max_depth:
            tmp = []
            for node in deci_nodes[d]:
                obs = node.obs.copy()
                classes, counts = np.unique(self.y_[node.nz], return_counts=True)
                rstar = max(counts) / node.nz.sum() - 1.0
                astar = classes[np.argmax(counts)]
                next_state = State(terminal_state, [0], is_terminal=True)
                next_state.qs = [rstar]
                a = Action(astar)
                a.transition(rstar, 1, next_state)
                node.add_action(a)
                # If there is still depth budget and the current split has more than 1 class:
                if (d + 1) < max_depth and classes.shape[0] >= 2:
                    feat_thresh = []
                    lefts, rights = [], []
                    probas_left, probas_right = [], []
                    if d <= len(self.cart_nodes_list) - 1:
                        clf = DecisionTreeClassifier(
                            max_leaf_nodes=self.cart_nodes_list[d]
                        )
                    else:
                        clf = DecisionTreeClassifier(max_leaf_nodes=2)

                    clf.fit(self.X_[node.nz], self.y_[node.nz])
                    for i in range(len(clf.tree_.feature)):
                        if clf.tree_.feature[i] >= 0:
                            # TODO: try to vectorize this ops.
                            inf = (
                                self.X_[:, clf.tree_.feature[i]]
                                <= clf.tree_.threshold[i]
                            ) * node.nz
                            sup = np.logical_not(inf) * node.nz
                            p_left = inf.sum() / node.nz.sum()
                            p_right = 1 - p_left
                            lefts.append(inf)
                            rights.append(sup)
                            feat_thresh.append(
                                [clf.tree_.feature[i], clf.tree_.threshold[i]]
                            )
                            probas_left.append(p_left)
                            probas_right.append(p_right)

                    for i, split in enumerate(feat_thresh):
                        a = Action(split)
                        feature, threshold = split
                        next_obs_left = obs.copy()
                        next_obs_left[self.X_.shape[1] + feature] = threshold
                        next_obs_right = obs.copy()
                        next_obs_right[feature] = threshold

                        if lefts[i].sum() > 0:
                            next_state_left = State(next_obs_left, lefts[i])
                            a.transition(0, probas_left[i], next_state_left)
                            tmp.append(next_state_left)
                        if rights[i].sum() > 0:
                            next_state_right = State(next_obs_right, rights[i])
                            a.transition(0, probas_right[i], next_state_right)
                            tmp.append(next_state_right)
                        if a.rewards != []:
                            node.add_action(a)

            if tmp != []:
                deci_nodes.append(tmp)
                d += 1
            else:
                break
        return deci_nodes

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the DPDTree classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        X, y = self._validate_data(X, y)
        # We need to make sure that we have a classification task
        check_classification_targets(y)

        # classifier should always store the classes seen during `fit`
        self.classes_ = np.unique(y)

        # Store the training data to predict later
        self.X_ = X
        self.y_ = y

        if self.max_nb_trees < 2:
            self.zetas_ = np.zeros(1)
        else:
            self.zetas_ = np.linspace(-1, 0, self.max_nb_trees)

        print("Building MDP")
        self.mdp_ = self.build_mdp()
        self.init_o_ = self.mdp_[0][0].obs
        print("Backward")
        self.trees_ = backward_induction_multiple_zetas(self.mdp_, self.zetas_)
        # Return the classifier
        return self

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted classes.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, reset=False)
        return self.predict_zeta_(X, -1)

    def predict_zeta_(self, X, zeta_index):
        """
        Predict class for X using a specific zeta index.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        zeta_index : int
            The index of the zeta value to use for prediction.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted classes.
        """
        # X = np.array(X, dtype=np.float64)
        init_a = self.trees_[tuple(self.init_o_.tolist() + [0])][zeta_index]
        y_pred = np.zeros(len(X), dtype=self.y_.dtype)
        for i, x in enumerate(X):
            a = init_a
            o = self.init_o_.copy()
            H = 0
            while isinstance(a, list):  # a is int implies leaf node
                feature, threshold = a
                H += 1
                if x[feature] <= threshold:
                    o[x.shape[0] + feature] = threshold
                else:
                    o[feature] = threshold
                a = self.trees_[tuple(o.tolist() + [H])][zeta_index]
            y_pred[i] = a
        return y_pred

    def average_traj_length_in_mdp_(self, X, y, zeta_index):
        """
        Calculate the average trajectory length in the MDP.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        zeta_index : int
            The zeta value to use for calculation.

        Returns
        -------
        accuracy : float
            The prediction accuracy.
        avg_length : float
            The average trajectory length.
        """
        nb_features = X.shape[1]
        init_a = self.trees_[tuple(self.init_o_.tolist() + [0])][zeta_index]
        lengths = np.zeros(X.shape[0])
        for i, s in enumerate(X):
            a = init_a
            o = self.init_o_.copy()
            H = 0
            while isinstance(a, list):  # a is int implies leaf node
                feature, threshold = a
                H += 1
                if s[feature] <= threshold:
                    o[nb_features + feature] = threshold
                else:
                    o[feature] = threshold
                a = self.trees_[tuple(o.tolist() + [H])][zeta_index]

            lengths[i] = H
        return (
            sum(
                [
                    self.predict_zeta_(X[i].reshape(1, -1), zeta_index)[0] == y[i]
                    for i in range(len(X))
                ]
            )
            / len(X),
            lengths.mean(),
        )

    def get_pareto_front(self, X, y):
        """
        Compute the decision path lengths / test accuracy pareto front of DPDTrees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        scores : array-like of shape (n_samples)
            The test accuracies of the trees.
        decision_path_length : array-like of shape (n_samples)
            The average number of decision nodes traversal in each tree.
        """
        scores = np.zeros(X.shape[0])
        decision_path_length = np.zeros(X.shape[0])
        for z in range(len(self.zetas_)):
            scores[z], decision_path_length[z] = self.average_traj_length_in_mdp_(
                X, y, z
            )
        return scores, decision_path_length
