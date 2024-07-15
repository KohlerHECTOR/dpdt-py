from dpdt import DPDTreeRegressor
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils._tags import _safe_tags

import pytest
from sklearn.tree import DecisionTreeRegressor
import numpy as np

@pytest.mark.parametrize("max_depth", [2, 4, 6, 8])
@pytest.mark.parametrize("max_nb_trees", [1, 20, 50, 100])
@pytest.mark.parametrize("cart_nodes_list", [(3,), (3, 3)])
def test_check_estimator(max_depth, max_nb_trees, cart_nodes_list):
    check_estimator(DPDTreeRegressor(max_depth, max_nb_trees, cart_nodes_list))


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
def test_better_cart_multiout(n_samples, n_features, centers, max_depth, cart_nodes_list):
    X = np.random.random(size=(n_samples, n_features))
    y = [[x[0]**i for i in range(centers)] for x in X]
    clf = DPDTreeRegressor(max_depth, cart_nodes_list=cart_nodes_list)
    clf.fit(X, y)
    cart = DecisionTreeRegressor(max_depth=max_depth, random_state=clf.random_state)
    cart.fit(X, y)
    dpdt_score = clf.score(X, y)
    cart_score = cart.score(X, y) 
    assert np.allclose(dpdt_score, cart_score, rtol=1e-5) or dpdt_score >= cart_score

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
    X = np.random.random(size=(n_samples, n_features))
    y = np.array([sum([x[0]**i for i in range(centers)]) for x in X])
    y = y.reshape(-1, 1)
    clf = DPDTreeRegressor(max_depth, cart_nodes_list=cart_nodes_list)
    clf.fit(X, y)
    cart = DecisionTreeRegressor(max_depth=max_depth, random_state=clf.random_state)
    cart.fit(X, y)
    dpdt_score = clf.score(X, y)
    cart_score = cart.score(X, y) 
    assert np.allclose(dpdt_score, cart_score, rtol=1e-5) or dpdt_score >= cart_score



@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize("costs",[(1,1,2), (0,1,1)])
def test_feature_costs(costs):
    X = np.random.random(size=(100, 3))
    y = np.array([sum([x[0]**i for i in range(3)]) for x in X])
    y = y.reshape(-1, 1)
    clf = DPDTreeRegressor(4)
    clf.fit(X, y, feature_costs=costs)


@pytest.mark.parametrize("costs",[(1,1,1), (10,1.5,1)])
def test_feature_costs(costs):
    X = np.random.random(size=(100, 3))
    y = np.array([sum([x[0]**i for i in range(3)]) for x in X])
    y = y.reshape(-1, 1)
    clf = DPDTreeRegressor(4)
    clf.fit(X, y, feature_costs=costs)




