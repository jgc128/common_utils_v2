from collections import defaultdict


class Registrable(object):
    """
    Based on AllenNLP
    https://github.com/allenai/allennlp/blob/74634e34145cb6b54ec16c4cc404eba5deed63d5/allennlp/common/registrable.py#L17-L73
    """
    _registry = defaultdict(dict)

    @classmethod
    def register(cls, name):
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass):
            if name in registry:
                message = f'Name `{name}` already in use for {cls.__name__} -> {registry[name].__name__}'
                raise ValueError(message)

            registry[name] = subclass
            return subclass

        return add_subclass_to_registry

    @classmethod
    def by_name(cls, name):
        registry = Registrable._registry[cls]
        if name not in registry:
            message = f'Name `{name}` is not registered for {cls.__name__}'
            raise ValueError(message)

        return registry.get(name)

    @classmethod
    def get_name(cls, obj):
        subclass = type(obj)
        registry = Registrable._registry[cls]
        name = [n for n, c in registry.items() if c == subclass]

        if len(name) == 0:
            message = f'Class `{subclass.__name__}` is not registered for {cls.__name__}'
            raise ValueError(message)

        name = name[0]
        return name
