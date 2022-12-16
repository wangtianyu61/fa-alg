#(TODO): add how to incorporate fairness into the design of FALCON
from src.datasets.tabular import ADULT, BRFSS, CANDC, COMPAS, GERMAN
from src.datasets.tabular import get_dataset_config
import os, subprocess
import numpy as np
import pandas as pd

from src.models.cb.base_cb import base_cb
from src.models.cb.FALCON import FALCON
from src.models.cb import config_choice, config_to_pair, FALCON_FAIR
from src.utils import generator
from src import IND_TIME_WINDOW

from src.eval_metric_cb import offline_eval_metric_cb
from src.utils.joint_regret import output_extraction

action_type = 'group-action-parity'
def run_falcon_fair(action_type = 'group-action-parity', dataset = 'adult'):
    dataset = dataset
    action_type = action_type
    n_actions = 2
    csv_path = 'src/datasets/' + dataset + '/' + dataset + '_' + str(n_actions) + '.csv'
    vw_path = 'src/datasets/' + dataset + '/' + dataset + '_' + str(n_actions) + '.vw.gz'
    output_path = 'results/' + dataset 
    loop_num = 10
    dataset_kwargs = {'root_dir': 'src/datasets'}
    filepath = dataset_kwargs['root_dir'] + '/' + dataset
    dataset_config = get_dataset_config(dataset, **dataset_kwargs)
    all_results = []
    linear_param = list(np.array(range(10)) / 10)
    print(linear_param)
    feature_param = ['linear', 'quad', 'log', 'cube']
    for gamma_param in [100]:
        for ind_parity in feature_param + linear_param:    
            for j in range(loop_num):
                sample_falcon = FALCON(csvpath = csv_path, gamma_param = gamma_param, 
                                        group = dataset_config.sens, action_parity = ind_parity, ind_parity = ind_parity)
                sample_falcon.learn_schedule(action_fair_type = action_type)
                eval_model = offline_eval_metric_cb(group = sample_falcon.group, context = sample_falcon.context_all,
                                                        action = sample_falcon.chosen_action_all.astype(int), loss = sample_falcon.loss_all)
                config_result = {'gamma': gamma_param, "fair_type": action_type, 
                                'ind_parity': ind_parity, 'idx': j}
                eval_model.group_loss_parity()
                eval_model.group_action_parity()
                eval_model.individual_loss_parity()
                config_result.update(eval_model.summary_loss)
                eval_model.offline_data.to_csv(os.path.join(output_path + '/' + dataset + '_new', dataset + '_' + str(action_type) + '_' + str(ind_parity) + '_' + str(j) + '.csv'))
                all_results.append(config_result)

    df = pd.DataFrame(all_results)
    output_file = os.path.join(output_path, dataset + '_summarynew_' + action_type + '.csv')
    if os.path.exists(output_file) == False:
        df.to_csv(output_file, index = None)
    else:
        #already exists one file
        df_raw = pd.read_csv(output_file)
        df_concat = pd.concat([df_raw, df])
        df_concat.to_csv(output_file, index = None)

#intervention on the worst case group
def run_falcon_fair_loss(dataset = 'adult'):
    dataset = dataset
    n_actions = 2
    csv_path = 'src/datasets/' + dataset + '/' + dataset + '_' + str(n_actions) + '.csv'
    vw_path = 'src/datasets/' + dataset + '/' + dataset + '_' + str(n_actions) + '.vw.gz'
    output_path = 'results/' + dataset 

    dataset_kwargs = {'root_dir': 'src/datasets'}
    filepath = dataset_kwargs['root_dir'] + '/' + dataset
    dataset_config = get_dataset_config(dataset, **dataset_kwargs)
    all_results = []
    loop_num = 10
    for gamma_param in [50, 100, 150, 200, 250, 300]:
        for loss_type in ['minmax-weight']:
            #add GBR 
            for model_class in ['linear']:
                if model_class == 'GBR' and loss_type == 'minmax-weight':
                    continue
                for j in range(loop_num):
                    sample_falcon = FALCON(csvpath = csv_path, gamma_param = gamma_param, 
                                                group = dataset_config.sens, funclass = model_class)
                    sample_falcon.learn_schedule(loss_fair_type = loss_type)
                    eval_model = offline_eval_metric_cb(group = sample_falcon.group, context = sample_falcon.context_all,
                                                                action = sample_falcon.chosen_action_all.astype(int), loss = sample_falcon.loss_all)
                    config_result = {'gamma': gamma_param, 'fair_loss_type': loss_type}
                    eval_model.group_loss_parity()
                    eval_model.group_action_parity()
                    eval_model.individual_loss_parity()
                    config_result.update(eval_model.summary_loss)
                    all_results.append(config_result)
    
    df = pd.DataFrame(all_results)
    output_file = os.path.join(output_path, dataset + '_summary_loss_linear.csv')
    #df.to_csv(output_file, index = None)
    if os.path.exists(output_file) == False:
        df.to_csv(output_file, index = None)
    else:
        #already exists one file
        df_raw = pd.read_csv(output_file)
        df_concat = pd.concat([df_raw, df])
        df_concat.to_csv(output_file, index = None)
            



