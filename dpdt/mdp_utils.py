import numpy as np


class State:
    """
    Represents a state in the Markov Decision Process (MDP).

    Parameters
    ----------
    label : array-like
        The observation label for the state.
    nz : array-like of bool
        Boolean array indicating which samples are present in the state.
    is_terminal : bool, default=False
        Indicates if the state is a terminal state.
    """

    __slots__ = ["obs", "actions", "qs", "is_terminal", "nz"]

    def __init__(self, label, nz, is_terminal=False):
        self.obs = label
        self.actions = []
        self.qs = []
        self.is_terminal = is_terminal
        self.nz = nz
        # self.v = 0

    def add_action(self, action):
        """
        Add an action to the state.

        Parameters
        ----------
        action : Action
            The action to be added to the state.
        """
        self.actions.append(action)


class Action:
    """
    Represents an action in the Markov Decision Process (MDP).

    Parameters
    ----------
    action : object
        The action representation (e.g., a split decision).
    """

    __slots__ = ["action", "rewards", "probas", "next_states"]

    def __init__(self, action):
        self.action = action
        self.rewards = []
        self.probas = []
        self.next_states = []

    def transition(self, reward, proba, next_s):
        """
        Add a transition for the action.

        Parameters
        ----------
        reward : float
            The reward associated with the transition.
        proba : float
            The probability of the transition.
        next_s : State
            The next state resulting from the transition.
        """
        self.rewards.append(reward)
        self.probas.append(proba)
        self.next_states.append(next_s)


def backward_induction_multiple_zetas(mdp, zetas):
    """
    Perform backward induction on the MDP for multiple zeta values.

    This function computes the optimal policy for each zeta value by performing
    backward induction on the provided MDP.

    Parameters
    ----------
    mdp : list of list of State
        The Markov Decision Process represented as a list of lists of states,
        where each inner list contains the states at a specific depth.
    zetas : array-like
        Array of zeta values to be used in the computation.

    Returns
    -------
    policy : dict
        A dictionary representing the policy. The keys are tuples representing
        the state observation and depth, and the values are the optimal actions
        for each zeta value.
    """
    policy = dict()
    max_depth = len(mdp)
    for H, d in enumerate(reversed(mdp)):
        for node in d:
            qs = []
            for a in node.actions:
                if not isinstance(a.action, tuple):
                    q_s_a = (
                        np.ones(zetas.shape[0]) * a.rewards[0]
                    )  # here the reward is misclassif cost.
                else:
                    q_s_a = np.zeros(zetas.shape[0])
                    for j, s_next in enumerate(a.next_states):
                        regul = zetas * a.rewards[0]
                        q_s_a += a.probas[j] * (regul + s_next.v)
                qs.append(q_s_a)
            qs = np.asarray(qs)
            argmax_qs = np.argmax(qs, axis=0)
            node.v = np.take_along_axis(qs, argmax_qs[None, :], 0)[0]
            policy[tuple(node.obs.tolist() + [max_depth - H - 1])] = [
                node.actions[k].action for k in argmax_qs
            ]
    return policy
