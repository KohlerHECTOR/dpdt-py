import numpy as np
from binarytree import Node


class InfoNode(Node):
    def __init__(self, feature, threshold, left, right):
        super().__init__("x_" + str(feature) + "≤" + str(round(threshold, 3)))
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class ActionNode(Node):
    def __init__(self, classif):
        self.classif = classif
        super().__init__("C_" + str(classif))


def extract_tree(policy, root, H, zeta):
    nb_feat = root.shape[0] // 2
    a = policy[tuple(root.tolist() + [H])][zeta]
    if isinstance(a, np.int64):
        return ActionNode(a), 1, 0
    else:
        feat, thresh = a
        left = root.copy()
        left[nb_feat + feat] = thresh
        right = root.copy()
        right[feat] = thresh
        child_l, nodes_l, depth_l = extract_tree(policy, left, H + 1, zeta)
        child_r, nodes_r, depth_r = extract_tree(policy, right, H + 1, zeta)
        return (
            InfoNode(feat, thresh, child_l, child_r),
            nodes_l + nodes_r + 1,
            max(depth_l, depth_r) + 1,
        )