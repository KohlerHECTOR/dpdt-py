from dpdt import DPDTree
from sklearn.tree import DecisionTreeClassifier
import json
import numpy as np
import matplotlib.pyplot as plt
from time import time

def get_occupancy_data(test=False):
    # Opening JSON file
    with open("classification_datasets/avila.json") as json_file:
        data = json.load(json_file)
    if test:
        return np.array(data["Xtest"]), np.array(data["Ytest"])
    else:
        return np.array(data["Xtrain"]), np.array(data["Ytrain"])


def count_avg_test_cart(clf: DecisionTreeClassifier, X):
    node_indicator = clf.decision_path(X)
    return node_indicator.sum(axis=1).mean() - 1


# Train
X, y = get_occupancy_data()
# DPDT
clf_dpdt = DPDTree(max_depth=3, max_nb_trees=20)
clf_dpdt.fit(X, y)
print(clf_dpdt.score(X, y))
#CART
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X,y)
print(clf.score(X, y))

# Test
X_test, y_test = get_occupancy_data(test=True)

t = time()
scores, avg_nb_tests = np.zeros_like(clf_dpdt.zetas), np.zeros_like(clf_dpdt.zetas)
for z in range(len(clf_dpdt.zetas)):
    scores[z], avg_nb_tests[z] = clf_dpdt.average_traj_length_in_mdp(X_test,y_test,zeta=z)
time_pareto_front = time()-t

plt.scatter(scores, avg_nb_tests, label="DPDTrees, time={}".format(round(time_pareto_front,3)), marker="*")


t = time()
path = clf.cost_complexity_pruning_path(X_test, y_test)
ccp_alphas = path.ccp_alphas
scores, avg_nb_tests = np.zeros_like(ccp_alphas), np.zeros_like(ccp_alphas)
for c, ccp_alpha in enumerate(ccp_alphas):
    clf = DecisionTreeClassifier(
        ccp_alpha=ccp_alpha,
        max_depth=3,
    )
    clf.fit(X_test, y_test)
    scores[c], avg_nb_tests[c] = clf.score(X_test, y_test), count_avg_test_cart(clf,X_test)
time_pareto_front = time()-t
plt.scatter(scores, avg_nb_tests, label="CARTccp, time={}".format(round(time_pareto_front,3)), marker="P", alpha=0.5)



plt.xlabel("Test Accuracy")
plt.ylabel("Average decision path length")
plt.title("Pareto Front - Avila")
plt.legend()
plt.savefig("avila.png")

