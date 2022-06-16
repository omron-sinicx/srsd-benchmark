from collections import OrderedDict

MODEL_CLASS_DICT = OrderedDict()
MODEL_FUNC_DICT = OrderedDict()


def register_model_class(cls):
    key = cls.__name__
    if key in MODEL_CLASS_DICT:
        raise KeyError(f'A model class key `{key}` has been already registered')

    MODEL_CLASS_DICT[key] = cls
    return cls


def register_model_func(func):
    key = func.__name__
    if key in MODEL_CLASS_DICT:
        raise KeyError(f'A model func key `{key}` has been already registered')

    MODEL_FUNC_DICT[key] = func
    return func


def get_symbolic_regression_model(cls_or_func_name, **kwargs):
    if cls_or_func_name in MODEL_CLASS_DICT:
        return MODEL_CLASS_DICT[cls_or_func_name](**kwargs)
    elif cls_or_func_name in MODEL_FUNC_DICT:
        return MODEL_FUNC_DICT[cls_or_func_name](**kwargs)
    return None
