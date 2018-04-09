import json
import munch


def all_vars():
    all_var = dict()
    all_var['Din'] = 750
    all_var['batch_size'] = 4
    all_var['num_tr_points'] = 300
    all_var['channels'] = [6, 7, 8, 9, 10, 11]
    # all_var['channels'] = [4, 6, 7, 8, 9, 10]
    # all_var['channels'] = [7, 8]
    return all_var


def process_config(config_file):
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    config = munch.Munch(config_dict)
    return config
