
def config_choice(model_name):
    if model_name == 'falcon':
        choices = {'gamma_param': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280]}
    else:
        if model_name == 'supervised':
            return {'class': ['linear']}
        choices = {'learning_rate': [0.001, 0.01, 0.1, 1, 10],
                   'cb_type': ['dr', 'ips', 'mtr']}

        if model_name == 'bag':
            choices.update({'bag': [2, 4, 8, 16]})
        elif model_name == 'eps-greedy':
            choices.update({'epsilon': [0, 0.02, 0.05, 0.1]})
        elif model_name == 'regcbopt':
            choices.update({'regcbopt': [None], 'mellowness': [1e-1, 1e-2, 1e-3]})
        elif model_name == 'cover':
            choices.update({'cover': [1, 4, 8], 'psi': [0, 0.01, 0.1, 0.5]})
    return choices

#update cmd for vw command line
def config_to_pair(cmd, config_param):
    for key in config_param.keys():
        cmd += ['--{}'.format(key)]
        if config_param[key] is not None:
            cmd += [str(config_param[key])]
    return cmd
    
