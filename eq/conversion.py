from collections import deque

import sympy
from sympy.utilities.misc import func_name
from zss import Node

from eq.registry import get_node_obj, NODE_WITH_END_TOKEN_SET, END_OF_TREE_TOKEN
from eq.vocabulary import SOS_TOKEN


def preorder_traverse_tree(node, symbol_list, returns_binary_tree=False):
    if node.is_number:
        symbol = 'c'
    elif isinstance(node, sympy.Symbol):
        symbol = str(node)
    else:
        symbol = func_name(node)
        if returns_binary_tree and symbol in NODE_WITH_END_TOKEN_SET:
            symbol = f'Bi{symbol}'

    symbol_list.append(symbol)
    num_children = len(node.args)
    for i, child_node in enumerate(node.args):
        if returns_binary_tree and symbol in NODE_WITH_END_TOKEN_SET and 0 < i < num_children - 1:
            symbol_list.append(symbol)
        preorder_traverse_tree(child_node, symbol_list, returns_binary_tree=returns_binary_tree)
    if not returns_binary_tree and symbol in NODE_WITH_END_TOKEN_SET:
        symbol_list.append(END_OF_TREE_TOKEN)


def sympy2sequence(sympy_eq, returns_binary_tree=False):
    symbol_list = list()
    # Sequence of symbols for preorder (depth-first) traversal
    preorder_traverse_tree(sympy_eq, symbol_list, returns_binary_tree)
    return symbol_list


def sequence2model(symbols, returns_parent_stack=False):
    open_parent_stack = deque()
    root_node = get_node_obj(symbols[0])
    if root_node.can_add_child():
        open_parent_stack.append(root_node)

    num_symbols = len(symbols)
    assert num_symbols > 0, '`symbols` should be a non-empty list'
    if num_symbols == 1:
        if returns_parent_stack:
            return root_node, open_parent_stack
        return root_node

    for i, symbol in enumerate(symbols[1:]):
        if symbol == SOS_TOKEN:
            raise SyntaxError(f'<SOS> should be the first symbol in the sequence, but appeared at {i}')

        parent_node = open_parent_stack.pop()
        if symbol == END_OF_TREE_TOKEN:
            parent_node.finalize_children(symbol)
            continue

        child_node = get_node_obj(symbol)
        parent_node.add_child(child_node)
        if not parent_node.is_ready_to_finalize_children():
            open_parent_stack.append(parent_node)
        if not child_node.is_ready_to_finalize_children():
            open_parent_stack.append(child_node)

    if returns_parent_stack:
        return root_node, open_parent_stack
    return root_node


def model2sequence(model, returns_binary_tree=True):
    sympy_eq_str = model.sympy_str()
    sympy_eq = sympy.simplify(sympy_eq_str)
    return sympy2sequence(sympy_eq.evalf(), returns_binary_tree=returns_binary_tree)


def sympy2zss_module(current_sympy_eq, parent_node=None, node_list=None):
    if node_list is None:
        node_list = list()

    if current_sympy_eq.is_number:
        node_label = 'Const'
    elif isinstance(current_sympy_eq, sympy.Symbol):
        node_label = str(current_sympy_eq)
    else:
        node_label = func_name(current_sympy_eq)

    current_idx = len(node_list)
    current_node = Node(node_label)
    if parent_node is not None:
        parent_node.addkid(current_node)

    node_list.append(current_idx)
    for child_node in current_sympy_eq.args:
        sympy2zss_module(child_node, current_node, node_list)
    return current_node
