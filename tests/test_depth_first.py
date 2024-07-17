from dpdt import (
    DPDTreeClassifier,
    DPDTreeRegressor,
    DPDTreeClassifierDepthFirst,
    DPDTreeRegressorDepthFirst,
)
from sklearn.utils.estimator_checks import check_estimator
import pytest
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
import numpy as np


@pytest.mark.parametrize("max_depth", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("max_nb_trees", [1, 20, 50, 100])
@pytest.mark.parametrize("cart_nodes_list", [(3,), (3, 5, 4, 1), (6, 6)])
def test_check_estimator(max_depth, max_nb_trees, cart_nodes_list):
    check_estimator(
        DPDTreeRegressorDepthFirst(max_depth, max_nb_trees, cart_nodes_list)
    )


@pytest.mark.parametrize(
    "n_samples",
    [10, 100],
)
@pytest.mark.parametrize(
    "n_features",
    [5, 200],
)
@pytest.mark.parametrize(
    "centers",
    [2, 4, 6],
)
@pytest.mark.parametrize("max_depth", [2, 4])
@pytest.mark.parametrize("cart_nodes_list", [(3,)])
def test_same_regression(n_samples, n_features, centers, max_depth, cart_nodes_list):
    X = np.random.random(size=(n_samples, n_features))
    y = [[x[0] ** i for i in range(centers)] for x in X]
    clf_dfs = DPDTreeRegressorDepthFirst(max_depth, cart_nodes_list=cart_nodes_list)
    clf_dfs.fit(X, y)
    clf_bfs = DPDTreeRegressor(max_depth=max_depth, cart_nodes_list=cart_nodes_list)
    clf_bfs.fit(X, y)
    assert clf_bfs.trees_ == clf_dfs.trees_


@pytest.mark.parametrize(
    "n_samples",
    [10, 1000],
)
@pytest.mark.parametrize(
    "n_features",
    [5, 500],
)
@pytest.mark.parametrize("max_depth", [2, 4, 6])
@pytest.mark.parametrize("max_nb_trees", [1, 50, 100])
@pytest.mark.parametrize("cart_nodes_list", [(3,), (3, 5, 4, 1), (6, 6)])
def test_dpdt__depth_first_is_same_learning(
    n_samples, n_features, max_depth, max_nb_trees, cart_nodes_list
):
    X, y = make_blobs(n_samples, centers=2, n_features=n_features, random_state=0)
    y = y.reshape(-1, 1)
    clf_dfs = DPDTreeClassifierDepthFirst(
        max_depth, max_nb_trees, cart_nodes_list, random_state=0
    )
    clf_bfs = DPDTreeClassifier(
        max_depth, max_nb_trees, cart_nodes_list, random_state=0
    )

    clf_dfs.fit(X, y)
    clf_bfs.fit(X, y)
    assert clf_dfs.trees_ == clf_bfs.trees_


@pytest.mark.parametrize("max_depth", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("max_nb_trees", [1, 20, 50, 100])
@pytest.mark.parametrize("cart_nodes_list", [(3,), (3, 5, 4, 1), (6, 6)])
def test_check_estimator(max_depth, max_nb_trees, cart_nodes_list):
    check_estimator(
        DPDTreeClassifierDepthFirst(max_depth, max_nb_trees, cart_nodes_list)
    )


@pytest.mark.parametrize(
    "n_samples",
    [10, 1000],
)
@pytest.mark.parametrize(
    "n_features",
    [5, 500],
)
@pytest.mark.parametrize("max_depth", [2, 4, 6])
@pytest.mark.parametrize("max_nb_trees", [1, 50, 100])
@pytest.mark.parametrize("cart_nodes_list", [(3,), (3, 5, 4, 1), (6, 6)])
def test_dpdt_learning(n_samples, n_features, max_depth, max_nb_trees, cart_nodes_list):
    X, y = make_blobs(n_samples, centers=2, n_features=n_features, random_state=0)
    y = y.reshape(-1, 1)
    clf = DPDTreeClassifierDepthFirst(max_depth, max_nb_trees, cart_nodes_list)
    clf.fit(X, y)
    clf.get_pareto_front(X, y)
    clf.predict(X)
    assert clf.score(X, y) >= 0.48


@pytest.mark.parametrize(
    "n_samples",
    [10, 1000],
)
@pytest.mark.parametrize(
    "n_features",
    [5, 500],
)
@pytest.mark.parametrize(
    "centers",
    [2, 4, 6],
)
@pytest.mark.parametrize("max_depth", [2, 4, 6])
@pytest.mark.parametrize("cart_nodes_list", [(3,), (3, 5, 4, 1), (6, 6)])
def test_better_cart(n_samples, n_features, centers, max_depth, cart_nodes_list):
    X, y = make_blobs(n_samples, centers=centers, n_features=n_features, random_state=0)
    y = y.reshape(-1, 1)
    clf = DPDTreeClassifierDepthFirst(max_depth, cart_nodes_list=cart_nodes_list)
    clf.fit(X, y)
    cart = DecisionTreeClassifier(max_depth=max_depth, random_state=clf.random_state)
    cart.fit(X, y)
    assert clf.score(X, y) >= cart.score(X, y)


@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize("costs", [(1, 1, 2), (0, 1, 1, 1, 1)])
def test_feature_costs(costs):
    X, y = make_blobs(100, centers=2, n_features=5, random_state=0)
    y = y.reshape(-1, 1)
    clf = DPDTreeClassifierDepthFirst(4)
    clf.fit(X, y, feature_costs=costs)


@pytest.mark.parametrize("costs", [(1, 1, 1, 1, 1), (10, 1.5, 1, 4, 1)])
def test_feature_costs(costs):
    X, y = make_blobs(100, centers=2, n_features=5, random_state=0)
    y = y.reshape(-1, 1)
    clf = DPDTreeClassifierDepthFirst(4)
    clf.fit(X, y, feature_costs=costs)
