from dpdt import DPDTree
from sklearn.utils.estimator_checks import check_estimator
import pytest
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier

@pytest.mark.parametrize('max_depth', [2, 4, 6, 8, 10])
@pytest.mark.parametrize('max_nb_trees', [1, 20, 50, 100])
@pytest.mark.parametrize('cart_nodes_list', [(3,), (3, 5, 4, 1), (6, 6)])
def test_check_estimator(max_depth, max_nb_trees, cart_nodes_list):
    check_estimator(DPDTree(max_depth, max_nb_trees, cart_nodes_list))



@pytest.mark.parametrize('n_samples', [10, 1000],)
@pytest.mark.parametrize('n_features', [5, 500],)
@pytest.mark.parametrize('max_depth', [2, 4, 6])
@pytest.mark.parametrize('max_nb_trees', [1, 50, 100])
@pytest.mark.parametrize('cart_nodes_list', [(3,), (3, 5, 4, 1), (6, 6)])
def test_dpdt_learning(n_samples, n_features, max_depth, max_nb_trees, cart_nodes_list):
    X, y = make_blobs(n_samples, centers=2, n_features=n_features,
                  random_state=0)
    y = y.reshape(-1, 1)
    clf = DPDTree(max_depth, max_nb_trees, cart_nodes_list)
    clf.fit(X, y)
    clf.get_pareto_front(X, y)
    clf.predict(X)
    assert clf.score(X, y) >= .48


@pytest.mark.parametrize('n_samples', [10, 1000],)
@pytest.mark.parametrize('n_features', [5, 500],)
@pytest.mark.parametrize('centers', [2, 4, 6],)
@pytest.mark.parametrize('max_depth', [2, 4, 6])
@pytest.mark.parametrize('cart_nodes_list', [(3,), (3, 5, 4, 1), (6, 6)])
def test_better_cart(n_samples, n_features, centers, max_depth, cart_nodes_list):
    X, y = make_blobs(n_samples, centers=centers, n_features=n_features,
                  random_state=0)
    y = y.reshape(-1, 1)
    clf = DPDTree(max_depth, cart_nodes_list = cart_nodes_list)
    clf.fit(X, y)
    cart = DecisionTreeClassifier(max_depth=max_depth, random_state = clf.random_state)
    cart.fit(X, y)
    assert clf.score(X, y) >= cart.score(X,y)

