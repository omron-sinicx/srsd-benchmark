from collections import OrderedDict

NODE_CLASS_DICT = OrderedDict()
NODE_WITH_END_TOKEN_SET = set()
END_OF_TREE_TOKEN = 'EOT'


def register_node_class(cls):
    key = cls.global_symbol if hasattr(cls, 'global_symbol') else cls.__name__
    if key in NODE_CLASS_DICT:
        raise KeyError(f'A node key `{key}` has been already registered')

    NODE_CLASS_DICT[key] = cls
    if cls.requires_end_token:
        NODE_WITH_END_TOKEN_SET.add(key)
        NODE_WITH_END_TOKEN_SET.add('Bi' + key)
    return cls


def get_node_obj(key, **kwargs):
    if len(kwargs) == 0:
        kwargs = {'symbol_str': key}

    if key in NODE_CLASS_DICT:
        return NODE_CLASS_DICT[key](**kwargs)

    if key.startswith('x') and len(key) > 1 and key[1:].isdigit():
        return NODE_CLASS_DICT[key[0]](variable_index=int(key[1:]), **kwargs)
    raise KeyError(f'`{key}` is not expected as a node key')
