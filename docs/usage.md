## Installation
```bash
pip install git+https://github.com/KohlerHECTOR/dpdt-py.git@v0.1.5
```


## Quickstart
DPDTree uses the ```scikit-learn``` API. You can find advanced exapmles [here](https://github.com/KohlerHECTOR/dpdt-py/blob/main/examples/).

```python
from dpdt import DPDTree
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


X, y = datasets.load_breast_cancer(return_X_y=True)

# DPDT
dpdt = DPDTree(max_depth=3, random_state=42)
dpdt.fit(X, y)

#CART
cart = DecisionTreeClassifier(max_depth=3, random_state=42)
cart.fit(X, y)
assert dpdt.score(X, y) >= cart.score(X, y)
```