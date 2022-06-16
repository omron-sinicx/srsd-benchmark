import graphviz
import sympy
from sympy.utilities.misc import func_name
from torchdistill.common.file_util import make_parent_dirs


def traverse_tree(node, dot, from_idx=None, node_list=None, num_digits=4):
    if node_list is None:
        node_list = list()

    if node.is_number:
        dot.attr('node', shape='box')
        node_label = str(node.evalf(num_digits))
    elif isinstance(node, sympy.Symbol):
        dot.attr('node', shape='doublecircle')
        node_label = str(node)
    else:
        dot.attr('node', shape='ellipse')
        node_label = func_name(node)

    current_idx = len(node_list)
    dot.node(str(current_idx), label=node_label)
    node_list.append(current_idx)
    if from_idx is not None:
        dot.edge(str(from_idx), str(current_idx))

    for child_node in node.args:
        traverse_tree(child_node, dot, current_idx, node_list, num_digits)


def visualize_sympy_as_tree(sympy_eq, output_file_path=None, ext=None, comment=None, label=None):
    if output_file_path is not None:
        make_parent_dirs(output_file_path)

    dot = graphviz.Digraph(comment=comment, format=ext)
    sympy_eq = sympy_eq.evalf()
    dot.attr(label=label)
    traverse_tree(sympy_eq, dot)
    dot.render(filename=output_file_path, cleanup=True, view=False)
