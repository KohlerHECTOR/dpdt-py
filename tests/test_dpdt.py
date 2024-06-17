import numpy as np
from dpdt import DPDTree
from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(DPDTree(2))


def test_dpdt_different_cart_size_mdeium():
    S, Y = np.random.random((100, 10)), np.random.randint(0, 6, 100)
    clf = DPDTree(2, max_nb_trees=5, cart_nodes_list=[16, 8])
    clf.fit(S, Y)
    clf.predict(S)
    clf.score(S, Y)


def test_dpdt_different_cart_size_big():
    S, Y = np.random.random((100, 10)), np.random.randint(0, 6, 100)
    clf = DPDTree(2, max_nb_trees=5, cart_nodes_list=[16, 8, 4])
    clf.fit(S, Y)
    clf.predict(S)
    clf.score(S, Y)


def test_dpdt_different_cart_size_small():
    S, Y = np.random.random((100, 10)), np.random.randint(0, 6, 100)
    clf = DPDTree(2, max_nb_trees=5, cart_nodes_list=[16])
    clf.fit(S, Y)
    clf.score(S, Y)


def test_dpdt_different_data_small():
    S, Y = np.random.random((100, 10)), np.random.randint(0, 6, 100)
    clf = DPDTree(3, max_nb_trees=5, cart_nodes_list=[16, 8])
    clf.fit(S, Y)
    clf.score(S, Y)


def test_dpdt_different_data_medium():
    S, Y = np.random.random((1000, 10)), np.random.randint(0, 6, 1000)
    clf = DPDTree(3, max_nb_trees=5, cart_nodes_list=[16, 8])
    clf.fit(S, Y)
    clf.score(S, Y)


def test_dpdt_different_data_big():
    S, Y = np.random.random((10000, 10)), np.random.randint(0, 6, 10000)
    clf = DPDTree(3, max_nb_trees=5, cart_nodes_list=[16, 8])

    clf.fit(S, Y)
    clf.score(S, Y)


def test_dpdt_different_data_high_dim():
    S, Y = np.random.random((10, 100)), np.random.randint(0, 6, 10)
    clf = DPDTree(3, max_nb_trees=5, cart_nodes_list=[16, 8])

    clf.fit(S, Y)
    clf.score(S, Y)


def test_ultimate():
    S, Y = np.random.random((1000, 50)), np.random.randint(0, 10, 1000)
    clf = DPDTree(4, max_nb_trees=10, cart_nodes_list=[16, 8, 4])

    clf.fit(S, Y)
    clf.score(S, Y)


def test_only_one_class():
    S, Y = np.random.random((1000, 50)), np.random.randint(0, 1, 1000)
    clf = DPDTree(3, max_nb_trees=5, cart_nodes_list=[16, 8])

    clf.fit(S, Y)
    clf.score(S, Y)


def test_real_data():
    S, Y = np.random.random((1000, 50)), np.random.randint(0, 10, 1000)
    clf = DPDTree(3, max_nb_trees=5, cart_nodes_list=[16, 8])

    clf.fit(S, Y)
    clf.score(S, Y)


def test_real_data_default():
    S, Y = np.random.random((1000, 50)), np.random.randint(0, 10, 1000)
    clf = DPDTree(3)

    clf.fit(S, Y)
    clf.score(S, Y)


def test_real_data_one_tree():
    S, Y = np.random.random((1000, 50)), np.random.randint(0, 10, 1000)
    clf = DPDTree(3, max_nb_trees=1, cart_nodes_list=[16, 8])

    clf.fit(S, Y)
    clf.score(S, Y)


def test_pareto_front():
    S, Y = np.random.random((1000, 50)), np.random.randint(0, 10, 1000)
    clf = DPDTree(3, max_nb_trees=4)
    clf.fit(S, Y)
    S, Y = np.random.random((1000, 50)), np.random.randint(0, 10, 1000)
    clf.get_pareto_front(S, Y)
