import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier
from .mdp_utils import backward_induction_multiple_zetas, Action, State


class DPDTree(ClassifierMixin, BaseEstimator):

    def __init__(self, max_depth: int, max_nb_trees: int = 1, cart_nodes_list: list[int] = [3]):
        self.max_nb_trees = max_nb_trees
        self.max_depth = max_depth
        self.cart_nodes_list = cart_nodes_list

        if max_nb_trees < 2:
            self.zetas = np.zeros(1)
        else:
            self.zetas = np.linspace(-1,0, max_nb_trees)


    def build_mdp(self):
        max_depth = self.max_depth + 1
        root = State(np.concatenate((self.X_.min(axis=0) - 1e-3, self.X_.max(axis=0) + 1e-3)), nz=np.ones(self.X_.shape[0], dtype=np.bool_))
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
                    if d <= len(self.cart_nodes_list)-1:
                        clf = DecisionTreeClassifier(max_leaf_nodes=self.cart_nodes_list[d])
                    else:
                        clf = DecisionTreeClassifier(max_leaf_nodes=2)

                    clf.fit(self.X_[node.nz], self.y_[node.nz])
                    for i in range(len(clf.tree_.feature)):
                        if clf.tree_.feature[i] >= 0:
                            # Try to vectorize this ops.
                            inf = (
                                (self.X_[:, clf.tree_.feature[i]] <= clf.tree_.threshold[i]) * node.nz
                            )
                            sup = np.logical_not(inf) * node.nz
                            p_left = inf.sum() / node.nz.sum()
                            p_right = 1 - p_left
                            lefts.append(inf)
                            rights.append(sup)
                            feat_thresh.append([clf.tree_.feature[i], clf.tree_.threshold[i]])
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

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        print("Building MDP")
        self.mdp = self.build_mdp()
        self.init_o = self.mdp[0][0].obs
        print("Backward")
        self.trees = backward_induction_multiple_zetas(self.mdp, self.zetas)
        # Return the classifier
        return self

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        # X = np.array(X, dtype=np.float64)
        init_a = self.trees[tuple(self.init_o.tolist() + [0])][-1]
        y_pred = [0 for _ in X]
        for i, x in enumerate(X):
            a = init_a
            o = self.init_o.copy()
            H = 0
            while isinstance(a, list):  # a is int implies leaf node
                feature, threshold = a
                H += 1
                if x[feature] <= threshold:
                    o[x.shape[0] + feature] = threshold
                else:
                    o[feature] = threshold
                a = self.trees[tuple(o.tolist() + [H])][-1]
            y_pred[i] = a
        return y_pred
    
    def average_traj_length_in_mdp(self, X, y, tree: dict):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.int64)

        nb_features = X.shape[1]
        init_a = tree[tuple(self.init_o.tolist() + [0])]
        lengths = np.zeros(X.shape[0])
        for i, s in enumerate(X):
            a = init_a
            o = self.init_o.copy()
            H = 0
            while isinstance(a, list):  # a is int implies leaf node
                feature, threshold = a
                H += 1
                if s[feature] <= threshold:
                    o[nb_features + feature] = threshold
                else:
                    o[feature] = threshold
                a = tree[tuple(o.tolist() + [H])]

            lengths[i] = H
        return self.score(X,y), lengths.mean()