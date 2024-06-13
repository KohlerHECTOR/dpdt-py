import numpy as np
from dpdt import DPDTree
import json

def get_occupancy_data(test=False):
    # Opening JSON file
    with open("tests/data/big_data/occupancy.json") as json_file:
        data = json.load(json_file)
    if test:
        return np.array(data["Xtest"]), np.array(data["Ytest"])
    else:
        return np.array(data["Xtrain"]), np.array(data["Ytrain"])

def test_dpdt_different_cart_size_mdeium():
    S, Y = np.random.random((100, 10)), np.random.randint(0, 6, 100)
    clf = DPDTree(2, max_nb_trees=5, cart_nodes_list=[16,8])
    clf.fit(S, Y)
    clf.predict(S)
    clf.score(S, Y)
    

def test_dpdt_different_cart_size_big():
    S, Y = np.random.random((100, 10)), np.random.randint(0, 6, 100)
    clf = DPDTree(2, max_nb_trees=5, cart_nodes_list=[16,8, 4])
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
    clf = DPDTree(3, max_nb_trees=5, cart_nodes_list=[16,8])
    clf.fit(S, Y)
    clf.score(S, Y)

def test_dpdt_different_data_medium():
    S, Y = np.random.random((1000, 10)), np.random.randint(0, 6, 1000)
    clf = DPDTree(3, max_nb_trees=5, cart_nodes_list=[16,8])
    clf.fit(S, Y)
    clf.score(S, Y)

def test_dpdt_different_data_big():
    S, Y = np.random.random((10000, 10)), np.random.randint(0, 6, 10000)
    clf = DPDTree(3, max_nb_trees=5, cart_nodes_list=[16,8])

    clf.fit(S, Y)
    clf.score(S, Y)

def test_dpdt_different_data_high_dim():
    S, Y = np.random.random((10, 100)), np.random.randint(0, 6, 10)
    clf = DPDTree(3, max_nb_trees=5, cart_nodes_list=[16,8])

    clf.fit(S, Y)
    clf.score(S, Y)

def test_ultimate():
    S, Y = np.random.random((1000, 50)), np.random.randint(0, 10, 1000)
    clf = DPDTree(4, max_nb_trees=10, cart_nodes_list=[16,8,4])

    clf.fit(S, Y)
    clf.score(S, Y)

def test_only_one_class():
    S, Y = np.random.random((1000, 50)), np.random.randint(0, 1, 1000)
    clf = DPDTree(3, max_nb_trees=5, cart_nodes_list=[16,8])

    clf.fit(S, Y)
    clf.score(S, Y)

def test_real_data():
    S, Y = get_occupancy_data()
    clf = DPDTree(3, max_nb_trees=5, cart_nodes_list=[16,8])

    clf.fit(S, Y)
    clf.score(S, Y)

def test_real_data_default():
    S, Y = get_occupancy_data()
    clf = DPDTree(3)

    clf.fit(S, Y)
    clf.score(S, Y)

def test_real_data_one_tree():
    S, Y = get_occupancy_data()
    clf = DPDTree(3, max_nb_trees=1, cart_nodes_list=[16,8])

    clf.fit(S, Y)
    clf.score(S, Y)