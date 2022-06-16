import math
import random

import torch
from torch import nn
from torch.nn.parameter import Parameter

from eq.registry import register_node_class, END_OF_TREE_TOKEN


class SymbolNode(nn.Module):
    global_symbol = None
    requires_end_token = False

    def __init__(self, symbol_str, min_num_children, max_num_children):
        super().__init__()
        self.symbol_str = symbol_str
        self.min_num_children = min_num_children
        self.max_num_children = max_num_children
        self.child_list = None if min_num_children == 0 or (max_num_children is not None and max_num_children == 0) \
            else nn.ModuleList()
        self.has_end_of_tree_token = False

    def can_add_child(self):
        if self.child_list is None:
            return False

        if self.requires_end_token and self.has_end_of_tree_token:
            return False
        return self.max_num_children is None or len(self.child_list) + 1 <= self.max_num_children

    def add_child(self, child_node):
        if self.can_add_child():
            self.child_list.append(child_node)
            return True

        print(f'This node `{self}` cannot have a new child anymore')
        return False

    def is_ready_to_finalize_children(self):
        if self.child_list is None:
            return True

        num_children = len(self.child_list)
        if (self.min_num_children <= num_children and self.max_num_children is None and self.has_end_of_tree_token) \
                or (self.max_num_children is not None and
                    self.min_num_children <= num_children <= self.max_num_children):
            return True
        return False

    def count_learnable_params(self):
        if self.child_list is None:
            return 0
        return sum([child_node.count_learnable_params() for child_node in self.child_list])

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def sympy_str(self):
        raise NotImplementedError()

    def __str__(self):
        return self.global_symbol


@register_node_class
class ConstantNode(SymbolNode):
    global_symbol = 'c'

    def __init__(self, symbol_str):
        super().__init__(symbol_str, min_num_children=0, max_num_children=0)
        init_value = torch.rand(1) * math.pow(10, random.uniform(-1, 1))
        if random.random() < 0.5:
            init_value *= -1.0
        self.coefficient = Parameter(init_value)

    def count_learnable_params(self):
        return 1

    def forward(self, batch):
        return self.coefficient

    def sympy_str(self):
        return str(self.coefficient.item())


@register_node_class
class VariableNode(SymbolNode):
    global_symbol = 'x'

    def __init__(self, symbol_str, variable_index):
        super().__init__(symbol_str, min_num_children=0, max_num_children=0)
        self.variable_index = variable_index

    def forward(self, batch):
        return batch[:, self.variable_index]

    def sympy_str(self):
        return self.global_symbol + str(self.variable_index)

    def __str__(self):
        return self.global_symbol + str(self.variable_index)


class OperationNode(SymbolNode):
    def __init__(self, symbol_str, min_num_children, max_num_children):
        super().__init__(symbol_str, min_num_children=min_num_children, max_num_children=max_num_children)

    def pull_up_child_value(self, batch, child_index):
        return self.child_list[child_index](batch)

    def forward(self, batch):
        raise NotImplementedError()

    def sympy_str(self):
        raise NotImplementedError()


class UnaryOperationNode(SymbolNode):
    def __init__(self, symbol_str):
        super().__init__(symbol_str, min_num_children=1, max_num_children=1)

    def pull_up_child_value(self, batch):
        assert len(self.child_list) == 1
        return self.child_list[0](batch)

    def forward(self, batch):
        raise NotImplementedError()

    def sympy_str(self):
        return self.global_symbol + '(' + self.child_list[0].sympy_str() + ')'


class BinaryOperationNode(SymbolNode):
    def __init__(self, symbol_str):
        super().__init__(symbol_str, min_num_children=2, max_num_children=2)

    def pull_up_child_values(self, batch):
        assert len(self.child_list) == 2
        return self.child_list[0](batch), self.child_list[1](batch)

    def forward(self, batch):
        raise NotImplementedError()

    def sympy_str(self):
        raise NotImplementedError()


class FlexibleOperationNode(SymbolNode):
    requires_end_token = True

    def __init__(self, symbol_str, min_num_children=2):
        super().__init__(symbol_str, min_num_children=min_num_children, max_num_children=None)

    def forward(self, batch):
        raise NotImplementedError()

    def sympy_str(self):
        raise NotImplementedError()

    def finalize_children(self, symbol):
        if symbol == END_OF_TREE_TOKEN:
            self.has_end_of_tree_token = True
            return True
        return False


@register_node_class
class BinaryAddOperationNode(BinaryOperationNode):
    global_symbol = 'BiAdd'

    def __init__(self, symbol_str):
        super().__init__(symbol_str)

    def forward(self, batch):
        left_term, right_term = self.pull_up_child_values(batch)
        return left_term + right_term

    def sympy_str(self):
        return self.child_list[0].sympy_str() + ' + ' + self.child_list[1].sympy_str()


@register_node_class
class BinaryMultiplyOperationNode(BinaryOperationNode):
    global_symbol = 'BiMul'

    def __init__(self, symbol_str):
        super().__init__(symbol_str)

    def forward(self, batch):
        left_term, right_term = self.pull_up_child_values(batch)
        return left_term * right_term

    def sympy_str(self):
        return self.child_list[0].sympy_str() + '*' + self.child_list[1].sympy_str()


@register_node_class
class AddOperationNode(FlexibleOperationNode):
    global_symbol = 'Add'

    def __init__(self, symbol_str):
        super().__init__(symbol_str)

    def forward(self, batch):
        return sum([child_node(batch) for child_node in self.child_list])

    def sympy_str(self):
        return ' + '.join([child_node.sympy_str() for child_node in self.child_list])


@register_node_class
class MultiplyOperationNode(FlexibleOperationNode):
    global_symbol = 'Mul'

    def __init__(self, symbol_str):
        super().__init__(symbol_str)

    def forward(self, batch):
        return math.prod([child_node(batch) for child_node in self.child_list])

    def sympy_str(self):
        return '*'.join([child_node.sympy_str() for child_node in self.child_list])


@register_node_class
class PowerOperationNode(BinaryOperationNode):
    global_symbol = 'Pow'

    def __init__(self, symbol_str):
        super().__init__(symbol_str)

    def forward(self, batch):
        base_input, exponent = self.pull_up_child_values(batch)
        return torch.pow(base_input, exponent)

    def sympy_str(self):
        return self.child_list[0].sympy_str() + '**' + self.child_list[1].sympy_str()


@register_node_class
class LogOperationNode(UnaryOperationNode):
    global_symbol = 'log'

    def __init__(self, symbol_str):
        super().__init__(symbol_str)

    def forward(self, batch):
        result = self.pull_up_child_value(batch)
        return torch.log(result)


@register_node_class
class ExpOperationNode(UnaryOperationNode):
    global_symbol = 'exp'

    def __init__(self, symbol_str):
        super().__init__(symbol_str)

    def forward(self, batch):
        result = self.pull_up_child_value(batch)
        return torch.exp(result)


@register_node_class
class SinOperationNode(UnaryOperationNode):
    global_symbol = 'sin'

    def __init__(self, symbol_str):
        super().__init__(symbol_str)

    def forward(self, batch):
        result = self.pull_up_child_value(batch)
        return torch.sin(result)


@register_node_class
class CosOperationNode(UnaryOperationNode):
    global_symbol = 'cos'

    def __init__(self, symbol_str):
        super().__init__(symbol_str)

    def forward(self, batch):
        result = self.pull_up_child_value(batch)
        return torch.cos(result)


@register_node_class
class TanOperationNode(UnaryOperationNode):
    global_symbol = 'tan'

    def __init__(self, symbol_str):
        super().__init__(symbol_str)

    def forward(self, batch):
        result = self.pull_up_child_value(batch)
        return torch.tan(result)


@register_node_class
class TanHOperationNode(UnaryOperationNode):
    global_symbol = 'tanh'

    def __init__(self, symbol_str):
        super().__init__(symbol_str)

    def forward(self, batch):
        result = self.pull_up_child_value(batch)
        return torch.tanh(result)
