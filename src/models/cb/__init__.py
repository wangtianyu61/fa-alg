FALCON_FAIR = False
def config_choice(model_name):
    
    if model_name == 'falcon':
        if FALCON_FAIR == False:
            choices = {'gamma_param': [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
                        'action_fair_type': [None],
                        'loss_fair_type': [None]}
        else:
            choices = {"gamma_param": [50, 100, 150, 200, 250, 300],
                        'action_fair_type': [None, 'individual-parity', 'group-action-parity'],
                        'loss_fair_type': [None, 'history-group-weight']}
    elif model_name == 'cftl':
        return {'class': ['logistic']}

    else:
        if model_name == 'supervised':
            return {'class': ['linear']}
        choices = {'learning_rate': [0.001, 0.01, 0.1, 1],
                   'cb_type': ['dr', 'ips', 'mtr']}

        if model_name == 'bag':
            choices.update({'bag': [2, 4, 8, 16]})
        elif model_name == 'eps-greedy':
            choices.update({'epsilon': [0, 0.02, 0.05, 0.1]})
        elif model_name == 'regcbopt':
            choices.update({'regcbopt': [None], 'mellowness': [1e-1, 1e-2, 1e-3]})
        elif model_name == 'cover':
            choices.update({'cover': [1, 4, 8], 'psi': [0, 0.1, 0.5]})
    return choices

#update cmd for vw command line
def config_to_pair(cmd, config_param):
    for key in config_param.keys():
        cmd += ['--{}'.format(key)]
        if config_param[key] is not None:
            cmd += [str(config_param[key])]
    return cmd
    
