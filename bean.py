import numpy as np
import json
from dpdt import DPDTreeClassifier
from memory_profiler import profile


def get_bean_data(test=False):

    # Opening JSON file
    with open("bean.json") as json_file:
        data = json.load(json_file)
    if test:
        return np.array(data["Xtest"]), np.array(data["Ytest"])
    else:
        return np.array(data["Xtrain"]), np.array(data["Ytrain"])


# @profile
def run(classif):
    classif.fit(X, y)
    return classif


if __name__ == "__main__":
    classif = DPDTreeClassifier(
        max_depth=3, max_nb_trees=1000, cart_nodes_list=(100, 100, 100)
    )
    X, y = get_bean_data()
    run(classif)
