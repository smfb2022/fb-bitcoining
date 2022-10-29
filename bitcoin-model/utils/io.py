import yaml


def load_yaml(config_path = './config/btc-config.yaml'):
    """_summary_
    Returns:
        _type_: _description_
    """
    #config_path = './config/btc-config.yaml'
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)

    return params
