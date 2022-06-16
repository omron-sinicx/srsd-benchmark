from .op import ConstantNode, BinaryAddOperationNode, BinaryMultiplyOperationNode, PowerOperationNode, \
    LogOperationNode, ExpOperationNode, SinOperationNode, CosOperationNode, TanOperationNode, TanHOperationNode


SOS_TOKEN = '<SOS>'
DEFAULT_VOCABULARY = [
    SOS_TOKEN, ConstantNode, BinaryAddOperationNode, BinaryMultiplyOperationNode, PowerOperationNode,
    LogOperationNode, ExpOperationNode, SinOperationNode, CosOperationNode, TanOperationNode, TanHOperationNode
]


class SymbolVocabulary(object):
    def __init__(self, symbols=None, max_num_variables=20, ignored_index=-255):
        if symbols is None:
            symbols = DEFAULT_VOCABULARY

        self.variable_symbols = [f'x{i}' for i in range(max_num_variables)]
        symbols = symbols + self.variable_symbols
        self.symbol2index_map = {s.global_symbol if hasattr(s, 'global_symbol') else s: i
                                 for i, s in enumerate(symbols)}
        self.index2symbol_map = {i: s for s, i in self.symbol2index_map.items()}
        self.variable_indices = self.convert_symbols_to_indices(self.variable_symbols)
        self.ignored_index = ignored_index

    def symbol2index(self, symbol):
        index = self.symbol2index_map.get(symbol, None)
        if index is None:
            raise KeyError(f'symbol `{symbol}` is not defined in vocabulary')
        return index

    def convert_symbols_to_indices(self, symbols):
        index_list = list()
        for symbol in symbols:
            index = self.symbol2index(symbol)
            index_list.append(index)
        return index_list

    def index2symbol(self, index):
        if index == self.ignored_index:
            return None

        symbol = self.index2symbol_map.get(index, None)
        if symbol is None:
            raise KeyError(f'index `{index}` is not defined in vocabulary')
        return symbol

    def convert_indices_to_symbols(self, indices):
        symbol_list = list()
        for index in indices:
            symbol = self.index2symbol(index)
            if symbol is None:
                break
            symbol_list.append(symbol)
        return symbol_list

    def get_unavailable_variable_indices(self, max_num_variables_in_batch):
        return self.variable_indices[max_num_variables_in_batch:]

    def get_size(self):
        return len(self.symbol2index_map)
