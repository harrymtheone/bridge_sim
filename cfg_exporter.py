import collections
import types
import typing
from typing import Dict, Any


def get_cfg_attributes(cfg) -> Dict[str, Any]:
    type_hints = typing.get_type_hints(cfg.__class__)

    for attr_name, attr_value in cfg.__dict__.items():
        attr_type = type_hints.get(attr_name, type(attr_value))
        yield attr_name, attr_type, attr_value


def get_type_name(type_hints, registered_cfg_class) -> str:
    if type_hints in [bool, int, float]:
        return type_hints.__name__

    elif type_hints in registered_cfg_class:
        pass

    else:
        raise NotImplementedError


def export_cfg_type_dict(cfg: object, cfg_type_dict: dict):
    for attr_name, attr_type, _ in get_cfg_attributes(cfg):
        if attr_type in [bool, int, float, str]:
            cfg_type_dict[attr_name] = {
                'type_name': attr_type.__name__,
                'type_structure': {}
            }

        elif isinstance(attr_type, typing._LiteralGenericAlias):
            cfg_type_dict[attr_name] = {
                'type_name': 'Literal',
                'type_structure': [*typing.get_args(attr_type)]
            }

        elif attr_type is collections.abc.Callable:
            cfg_type_dict[attr_name] = {
                'type_name': 'Callable',
                'type_structure': {}
            }

        elif attr_type is dict:
            raise ValueError("dict without content annotation is invalid!")

        elif typing.get_origin(attr_type) is dict:
            k, v = typing.get_args(attr_type)
            assert k is str

            cfg_type_dict[attr_name] = {
                'type_name': str(attr_type),
                'type_structure': {},
            }

            if v in [bool, int, float, str]:
                pass


        elif isinstance(attr_type, types.GenericAlias):
            cfg_type_dict[attr_name] = {
                'type_name': attr_type.__name__,
                'type_structure': []
            }

            for arg_type in typing.get_args(attr_type):
                assert arg_type in [bool, int, float, str]
                cfg_type_dict[attr_name]['type_structure'].append(arg_type.__name__)

        elif (typing.get_origin(attr_type) is types.UnionType) or (typing.get_origin(attr_type) is typing.Union):
            cfg_type_dict[attr_name] = {
                'type_name': str(attr_type),
                'type_structure': {}
            }

        elif hasattr(attr_type, 'validate'):
            cfg_type_dict[attr_name] = {
                'type_name': f"{attr_type.__module__}.{attr_type.__name__}",
                'type_structure': {}
            }
            export_cfg_type_dict(getattr(cfg, attr_name), cfg_type_dict[attr_name]['type_structure'])

        else:
            print(attr_name, attr_type)
            raise NotImplementedError
