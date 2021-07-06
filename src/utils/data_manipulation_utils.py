import collections.abc

def flatten(d: collections.abc.MutableMapping, parent_key='', sep='_'):
    """Flatten a nested dictionary
    Args:
        d (collections.abc.MutableMapping): input dictionary
        parent_key (str):
        sep (str):
    Returns:
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
