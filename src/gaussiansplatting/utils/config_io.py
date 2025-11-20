from arguments import GroupParams


def recursive_args_from_pg(a: "GroupParams"):
    args = {}
    for k, v in vars(a).items():
        if isinstance(v, GroupParams):
            args.update(recursive_args_from_pg(v))
        else:
            args[k] = v
    return args
