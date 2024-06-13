import numpy as np
from typing import Union

class State:
    def __init__(self, label: np.ndarray, nz: np.ndarray, is_terminal: bool = False):
        self.obs = label
        self.actions = []
        self.qs = []
        self.is_terminal = is_terminal
        self.nz = nz
        # self.v = 0

    def add_action(self, action):
        self.actions.append(action)

class Action:
    def __init__(self, action: Union[np.ndarray, np.int64]):
        self.action = action
        self.rewards = []
        self.probas = []
        self.next_states = []

    def transition(self, reward: float, proba: float, next_s: State):
        self.rewards.append(reward)
        self.probas.append(proba)
        self.next_states.append(next_s)



def backward_induction_multiple_zetas(mdp: list[list[State]], zetas: np.ndarray):
    policy = dict()
    max_depth = len(mdp)
    for H, d in enumerate(reversed(mdp)):
        for node in d:
            qs = []
            for a in node.actions:
                if not isinstance(a.action, list):
                    q_s_a = np.ones(zetas.shape[0]) * a.rewards[0]
                else:
                    q_s_a = np.zeros(zetas.shape[0])
                    for j, s_next in enumerate(a.next_states):
                        q_s_a += a.probas[j] * (zetas + s_next.v)
                qs.append(q_s_a)
            qs = np.asarray(qs)
            argmax_qs = np.argmax(qs, axis=0)
            node.v = np.take_along_axis(qs, argmax_qs[None, :], 0)[0]
            policy[tuple(node.obs.tolist() + [max_depth - H - 1])] = [
                node.actions[k].action for k in argmax_qs
            ]
    return policy