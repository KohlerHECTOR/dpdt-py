## Installation
```bash
pip install git+https://github.com/KohlerHECTOR/dpdt-py
```


## Quickstart
DPDTree uses the ```scikit-learn``` API. You can find advanced exapmles [here](https://github.com/KohlerHECTOR/dpdt-py/blob/main/examples/).

```python
from dpdt import DPDTree
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


X, y = datasets.load_breast_cancer(return_X_y=True)

# DPDT
clf = DPDTree(max_depth=3)
clf.fit(X, y)
print(clf.score(X, y))

#CART
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X,y)
print(clf.score(X, y))
```