import sympy
from zss import simple_distance

from eq.conversion import sympy2zss_module, sequence2model


def convert_pred_sequence_to_eqs(pred_sequences, prints_str=False):
    pred_eq_list = list()
    for pred_sequence in pred_sequences:
        # pred_sequence[0] = '<SOS>', thus skipped
        pred_model = sequence2model(pred_sequence[1:])
        try:
            sympy_eq_str = pred_model.sympy_str()
        except:
            sympy_eq_str = 'nan'

        if prints_str:
            print(sympy_eq_str)

        try:
            pred_eq = sympy.sympify(sympy_eq_str)
            pred_eq = pred_eq.subs(1.0, 1)
        except:
            pred_eq = sympy.nan
        pred_eq_list.append(pred_eq)
    return pred_eq_list


def count_nodes(zss_node):
    if zss_node is None:
        return 0

    count = 1
    for child in zss_node.children:
        count += count_nodes(child)
    return count


def compute_distance(est_eq_tree, gt_eq_tree, normalizes=False):
    edit_dist = simple_distance(est_eq_tree, gt_eq_tree)
    if not normalizes:
        return edit_dist

    num_gt_nodes = count_nodes(gt_eq_tree)
    return min([edit_dist, num_gt_nodes]) / num_gt_nodes


def compute_distances(est_eq_trees, gt_eq_trees, normalizes=False):
    return [compute_distance(est_eq_tree, gt_eq_tree, normalizes)
            for est_eq_tree, gt_eq_tree in zip(est_eq_trees, gt_eq_trees)]


def compute_edit_distance(pred_eqs, true_eqs, normalizes=True):
    edit_dist = 0
    batch_size = len(true_eqs)
    for pred_eq, true_eq in zip(pred_eqs, true_eqs):
        pred_eq_tree = sympy2zss_module(pred_eq.evalf())
        true_eq_tree = sympy2zss_module(true_eq.evalf())
        edit_dist += compute_distance(pred_eq_tree, true_eq_tree, normalizes)
    return edit_dist / batch_size


def compute_edit_distances(pred_eqs, true_eqs, normalizes=True):
    edit_dist_list = list()
    for pred_eq, true_eq in zip(pred_eqs, true_eqs):
        pred_eq_tree = sympy2zss_module(pred_eq.evalf())
        true_eq_tree = sympy2zss_module(true_eq.evalf())
        edit_dist = compute_distance(pred_eq_tree, true_eq_tree, normalizes)
        edit_dist_list.append(edit_dist)
    return edit_dist_list
