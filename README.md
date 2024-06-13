```bash
pip3 install -e .
```

```python
from dpdt import DPDTree
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


X, y = datasets.load_breast_cancer()

# DPDT
clf = DPDTree(max_depth=3)
clf.fit(X, y)
print(clf.score(X, y))

#CART
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X,y)
print(clf.score(X, y))
```
