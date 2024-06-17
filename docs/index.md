# DPDTree

Finding an optimal decision tree for a supervised learning task is a challenging combinatorial problem to solve at scale. DPDT frames the problem as a Markov Decision Problem (MDP). In the MDP, each trajectory corresponds to a sequence of tests on training data until a class is assigned. Potential tests are obtained by calling CART [Breiman, 1984](https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman), and dynamic programming is used to induce the best sequence of tests, i.e the best decision trees. By definition, the train accuracy of DPDT will always be greater than CART. One other property of DPDT is that its trees pareto dominate CART' w.r.t the trade-off between the average decision path length and the test accuracies. See e.g.:
![Avila](examples/avila.png)ithm*.


## User Guide


Information on how to use AdaStop library.

```{toctree}
:maxdepth: 2

user_guide
```


## API

Information on python classes behind AdaStop.

```{toctree}
:maxdepth: 2

api
```
