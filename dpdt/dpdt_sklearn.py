import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context, RegressorMixin, MultiOutputMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils._param_validation import Interval
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from .mdp_utils import backward_induction_multiple_zetas, Action, State
from numbers import Integral, Real


class DPDTreeClassifier(ClassifierMixin, BaseEstimator):
    """
    Dynamic Programming Decision Tree (DPDTree) classifier.

    Parameters
    ----------
    max_depth : int
        The maximum depth of the tree.
    max_nb_trees : int, default=1000
        The maximum number of trees.
    random_state : int, default=42
        Fixes randomness of the classifier. Randomness happens in the calls to cart.
    cart_nodes_list : list of int, default=(3,)
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
    >>> from dpdt import DPDTreeClassifier
    >>> from sklearn import datasets
    >>>
    >>> X, y = datasets.load_breast_cancer(return_X_y=True)
    >>>
    >>> clf = DPDTreeClassifier(max_depth=3, random_state=42)
    >>> clf.fit(X, y)
    >>> print(clf.score(X, y))
    """

    _parameter_constraints = {
        "max_depth": [Interval(Integral, 2, None, closed="left")],
        "max_nb_trees": [Interval(Integral, 1, None, closed="left")],
        "cart_nodes_list": ["array-like"],
        "random_state": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self, max_depth=3, max_nb_trees=1000, cart_nodes_list=(3,), random_state=42
    ):
        # TODO potentially direcltly pass an instantiated CART.
        self.max_depth = max_depth
        self.max_nb_trees = max_nb_trees
        self.cart_nodes_list = cart_nodes_list
        self.random_state = random_state

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
            for node in deci_nodes[d]:# Should be parallel
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
                if (d + 1) < max_depth and rstar < 0:
                    # Get the splits from CART
                    # Note that that 2 leaf nodes means that the split is greedy.
                    if d <= len(self.cart_nodes_list) - 1:
                        clf = DecisionTreeClassifier(
                            max_leaf_nodes=max(2, self.cart_nodes_list[d]),
                            random_state=self.random_state,
                        )
                    # If depth budget reaches limit, get the max entropy split.
                    else:
                        clf = DecisionTreeClassifier(
                            max_leaf_nodes=2, random_state=self.random_state
                        )

                    clf.fit(self.X_[node.nz], self.y_[node.nz])

                    # Extract the splits from the CART tree.

                    masks = clf.tree_.feature >= 0  # get tested features.

                    # Apply mask to features and thresholds to get valid indices
                    valid_features = clf.tree_.feature[masks]
                    valid_thresholds = clf.tree_.threshold[masks]
                    lefts = (
                        self.X_[:, valid_features] <= valid_thresholds
                    )  # is a 2D array with nb CART tree tests columns.
                    rights = np.logical_not(
                        lefts
                    )  # as many rows as data in the whole training set.

                    # Masking data passing threshold and precedent thresholds.
                    lefts *= node.nz[:, np.newaxis]
                    rights *= node.nz[:, np.newaxis]

                    # Compute probabilities
                    p_left = lefts.sum(axis=0) / node.nz.sum()  # summing column values.
                    # In each column (tested features), non-zero values are data indices passing all tests so far in the MDP trajectory.
                    p_right = 1 - p_left

                    feat_thresh = list(
                        zip(valid_features, valid_thresholds)
                    )  # len of the list is nb tests in CART tree.

                    # Precompute next observations for left and right splits
                    next_obs_left = np.tile(obs, (len(feat_thresh), 1))
                    next_obs_right = np.tile(obs, (len(feat_thresh), 1))
                    indices = np.arange(len(feat_thresh))

                    # Fast next obs computations. The next obs in the MDP traj get their bounds updated as the threshold values.
                    next_obs_left[
                        indices, self.X_.shape[1] + valid_features
                    ] = valid_thresholds
                    next_obs_right[indices, valid_features] = valid_thresholds

                    # Create Action objects for each split
                    actions = [Action(split) for split in feat_thresh]

                    # Precompute next states for left and right
                    # There should be a pair of next_states per tested features.
                    next_states_left = [
                        State(next_obs_left[i], lefts[:, i])
                        for i in range(len(valid_features))
                    ]
                    next_states_right = [
                        State(next_obs_right[i], rights[:, i])
                        for i in range(len(valid_features))
                    ]

                    # Perform transitions and append states, the reward is equal to the feature cost.
                    for i in range(len(valid_features)):
                        if lefts[:, i].astype(int).sum() > 0:
                            actions[i].transition(self.feature_costs_[actions[i].action[0]], p_left[i], next_states_left[i])
                            tmp.append(next_states_left[i])

                    for i in range(len(valid_features)):
                        if rights[:, i].astype(int).sum() > 0:
                            actions[i].transition(self.feature_costs_[actions[i].action[0]], p_right[i], next_states_right[i])
                            tmp.append(next_states_right[i])

                    [
                        node.add_action(action)
                        for action in actions
                        if action.rewards != []
                    ]

            if tmp != []:
                deci_nodes.append(tmp)
                d += 1
            else:
                break
        return deci_nodes

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, feature_costs=None):
        """
        Fit the DPDTree classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        feature_costs (optional): list of float, default=None
            List containing the features costs.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        X, y = self._validate_data(X, y)
        if feature_costs:
            assert len(feature_costs) == X.shape[1], "There should be as much feature costs as features."
            assert all([fc >= 1 for fc in feature_costs]), "Feature costs must be greater than 1."
            min_cost, max_cost = min(feature_costs), max(feature_costs)
            if min_cost == max_cost:
                feature_costs = [1 for _ in feature_costs]
            self.feature_costs_ = feature_costs
        else:
            self.feature_costs_= np.ones(X.shape[1])
            
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
            assert len(self.zetas_) == self.max_nb_trees

        
        self.mdp_ = self.build_mdp()
        self.init_o_ = self.mdp_[0][0].obs
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
            while isinstance(a, tuple):  # a is int implies leaf node
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
        lengths, costs = np.zeros(X.shape[0]), np.zeros(X.shape[0])
        for i, s in enumerate(X):
            a = init_a
            o = self.init_o_.copy()
            H = 0
            cost = 0
            while isinstance(a, tuple):  # a is int implies leaf node
                feature, threshold = a
                H += 1
                cost += self.feature_costs_[feature]
                if s[feature] <= threshold:
                    o[nb_features + feature] = threshold
                else:
                    o[feature] = threshold
                a = self.trees_[tuple(o.tolist() + [H])][zeta_index]

            lengths[i] = H
            costs[i] = cost
        return (
            sum(
                [
                    self.predict_zeta_(X[i].reshape(1, -1), zeta_index)[0] == y[i]
                    for i in range(len(X))
                ]
            )
            / len(X),
            lengths.mean(),
            costs.mean()
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
        scores = np.zeros(len(self.zetas_))
        decision_path_length = np.zeros(len(self.zetas_))
        cost = np.zeros(len(self.zetas_))
        for z in range(len(self.zetas_)):
            scores[z], decision_path_length[z], cost[z] = self.average_traj_length_in_mdp_(
                X, y, z
            )
        return scores, decision_path_length, cost
    

class DPDTreeRegressor(RegressorMixin, MultiOutputMixin, BaseEstimator):
    """
    Dynamic Programming Decision Tree (DPDTree) regressor.

    Parameters
    ----------
    max_depth : int
        The maximum depth of the tree.
    max_nb_trees : int, default=1000
        The maximum number of trees.
    random_state : int, default=42
        Fixes randomness of the regressor. Randomness happens in the calls to cart.
    cart_nodes_list : list of int, default=(3,)
        List containing the number of leaf nodes for the CART trees at each depth.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.

    y_ : ndarray, shape (n_samples,) or (n_samples, n_output)
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

    """

    _parameter_constraints = {
        "max_depth": [Interval(Integral, 2, None, closed="left")],
        "max_nb_trees": [Interval(Integral, 1, None, closed="left")],
        "cart_nodes_list": ["array-like"],
        "random_state": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self, max_depth=3, max_nb_trees=1000, cart_nodes_list=(3,), random_state=42
    ):
        # TODO potentially direcltly pass an instantiated CART.
        self.max_depth = max_depth
        self.max_nb_trees = max_nb_trees
        self.cart_nodes_list = cart_nodes_list
        self.random_state = random_state

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
            for node in deci_nodes[d]: # Should be parallel
                obs = node.obs.copy()
                astar = self.y_[node.nz].mean(axis=0)
                rstar = -1 * mean_squared_error(self.y_[node.nz], np.tile(astar, (len(self.y_[node.nz]), 1)))
                next_state = State(terminal_state, [0], is_terminal=True)
                next_state.qs = [rstar]
                a = Action(astar)
                a.transition(rstar, 1, next_state)
                node.add_action(a)
                # If there is still depth budget and the current split has more than 1 class:
                if (d + 1) < max_depth and (rstar < 0 and not np.isnan(rstar)):
                    # Get the splits from CART
                    # Note that that 2 leaf nodes means that the split is greedy.
                    if d <= len(self.cart_nodes_list) - 1:
                        clf = DecisionTreeRegressor(
                            max_leaf_nodes=max(2, self.cart_nodes_list[d]),
                            random_state=self.random_state,
                        )
                    # If depth budget reaches limit, get the max entropy split.
                    else:
                        clf = DecisionTreeRegressor(
                            max_leaf_nodes=2, random_state=self.random_state
                        )

                    clf.fit(self.X_[node.nz], self.y_[node.nz])

                    # Extract the splits from the CART tree.

                    masks = clf.tree_.feature >= 0  # get tested features.

                    # Apply mask to features and thresholds to get valid indices
                    valid_features = clf.tree_.feature[masks]
                    valid_thresholds = clf.tree_.threshold[masks]
                    lefts = (
                        self.X_[:, valid_features] <= valid_thresholds
                    )  # is a 2D array with nb CART tree tests columns.
                    rights = np.logical_not(
                        lefts
                    )  # as many rows as data in the whole training set.

                    # Masking data passing threshold and precedent thresholds.
                    lefts *= node.nz[:, np.newaxis]
                    rights *= node.nz[:, np.newaxis]

                    # Compute probabilities
                    p_left = lefts.sum(axis=0) / node.nz.sum()  # summing column values.
                    # In each column (tested features), non-zero values are data indices passing all tests so far in the MDP trajectory.
                    p_right = 1 - p_left

                    feat_thresh = list(
                        zip(valid_features, valid_thresholds)
                    )  # len of the list is nb tests in CART tree.

                    # Precompute next observations for left and right splits
                    next_obs_left = np.tile(obs, (len(feat_thresh), 1))
                    next_obs_right = np.tile(obs, (len(feat_thresh), 1))
                    indices = np.arange(len(feat_thresh))

                    # Fast next obs computations. The next obs in the MDP traj get their bounds updated as the threshold values.
                    next_obs_left[
                        indices, self.X_.shape[1] + valid_features
                    ] = valid_thresholds
                    next_obs_right[indices, valid_features] = valid_thresholds

                    # Create Action objects for each split
                    actions = [Action(split) for split in feat_thresh]

                    # Precompute next states for left and right
                    # There should be a pair of next_states per tested features.
                    next_states_left = [
                        State(next_obs_left[i], lefts[:, i])
                        for i in range(len(valid_features))
                    ]
                    next_states_right = [
                        State(next_obs_right[i], rights[:, i])
                        for i in range(len(valid_features))
                    ]

                    # Perform transitions and append states, the reward is equal to the feature cost.
                    for i in range(len(valid_features)):
                        if lefts[:, i].astype(int).sum() > 0:
                            actions[i].transition(self.feature_costs_[actions[i].action[0]], p_left[i], next_states_left[i])
                            tmp.append(next_states_left[i])

                    for i in range(len(valid_features)):
                        if rights[:, i].astype(int).sum() > 0:
                            actions[i].transition(self.feature_costs_[actions[i].action[0]], p_right[i], next_states_right[i])
                            tmp.append(next_states_right[i])

                    [
                        node.add_action(action)
                        for action in actions
                        if action.rewards != []
                    ]

            if tmp != []:
                deci_nodes.append(tmp)
                d += 1
            else:
                break
        return deci_nodes

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, feature_costs=None):
        """
        Fit the DPDTree Regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_output)
            The target values.
        feature_costs (optional): list of float, default=None
            List containing the features costs.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)
        self._check_n_features(X, reset=True)

        if feature_costs:
            assert len(feature_costs) == X.shape[1], "There should be as much feature costs as features."
            assert all([fc >= 1 for fc in feature_costs]), "Feature costs must be greater than 1."
            min_cost, max_cost = min(feature_costs), max(feature_costs)
            if min_cost == max_cost:
                feature_costs = [1 for _ in feature_costs]
            self.feature_costs_ = feature_costs
        else:
            self.feature_costs_= np.ones(X.shape[1])
            
        # Store the training data to predict later
        self.X_ = X
        self.y_ = y.astype(float)
        if self.max_nb_trees < 2:
            self.zetas_ = np.zeros(1)
        else:
            self.zetas_ = np.linspace(-1, 0, self.max_nb_trees)
            assert len(self.zetas_) == self.max_nb_trees

        
        self.mdp_ = self.build_mdp()
        self.init_o_ = self.mdp_[0][0].obs
        self.trees_ = backward_induction_multiple_zetas(self.mdp_, self.zetas_)
        # Return the regressor
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
        X = check_array(X)
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
        if self.y_.ndim>1:
            y_pred = np.zeros((len(X), self.y_.shape[1]), dtype=self.y_.dtype)
        else:
            y_pred = np.zeros((len(X)), dtype=self.y_.dtype)
        for i, x in enumerate(X):
            a = init_a
            o = self.init_o_.copy()
            H = 0
            while isinstance(a, tuple):  # a is int implies leaf node
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
        lengths, costs = np.zeros(X.shape[0]), np.zeros(X.shape[0])
        for i, s in enumerate(X):
            a = init_a
            o = self.init_o_.copy()
            H = 0
            cost = 0
            while isinstance(a, tuple):  # a is int implies leaf node
                feature, threshold = a
                H += 1
                cost += self.feature_costs_[feature]
                if s[feature] <= threshold:
                    o[nb_features + feature] = threshold
                else:
                    o[feature] = threshold
                a = self.trees_[tuple(o.tolist() + [H])][zeta_index]

            lengths[i] = H
            costs[i] = cost
        return (
            np.mean(
                [
                    (y[i] - self.predict_zeta_(X[i].reshape(1, -1), zeta_index)[0])**2
                    for i in range(len(X))
                ]
            )
            ,
            lengths.mean(),
            costs.mean()
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
        scores = np.zeros(len(self.zetas_))
        decision_path_length = np.zeros(len(self.zetas_))
        cost = np.zeros(len(self.zetas_))
        for z in range(len(self.zetas_)):
            scores[z], decision_path_length[z], cost[z] = self.average_traj_length_in_mdp_(
                X, y, z
            )
        return scores, decision_path_length, cost
