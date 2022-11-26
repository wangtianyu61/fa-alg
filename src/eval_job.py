from src.datasets.tabular import ADULT, BRFSS, CANDC, COMPAS, GERMAN
from src.datasets.tabular import get_dataset_config
import os, subprocess
import numpy as np
import pandas as pd

from src.models.cb.base_cb import base_cb
from src.models.cb.FALCON import FALCON
from src.models.cb import config_choice, config_to_pair
from src.utils import generator

from src.eval_metric_cb import offline_eval_metric_cb
from src.utils.joint_regret import output_extraction

VW = 'vw'

#(TODO): remind to incorporate the multi-action case...
def run_jobs():
    model_classes = ['falcon', 'bag', 'regcbopt', 'eps-greedy', 'cover']

    dataset = ADULT
    n_actions = 2
    csv_path = 'src/datasets/' + dataset + '/' + dataset + '_' + str(n_actions) + '.csv'
    vw_path = 'src/datasets/' + dataset + '/' + dataset + '_' + str(n_actions) + '.vw.gz'
    output_path = 'results/' + dataset 

    dataset_kwargs = {'root_dir': 'src/datasets'}
    filepath = dataset_kwargs['root_dir'] + '/' + dataset
    dataset_config = get_dataset_config(dataset, **dataset_kwargs)

    all_config_results = []
    for model_name in ['eps-greedy']:
        #model_name = 'falcon'
        
        #config_result = {'model': model_name, 'config': str(config_param)}
        #obtain the config from that model class
        all_params = config_choice(model_name)
        cmd = [VW, vw_path, '-b', '24', '--progress', '1']
        for idx, config_param in enumerate(generator(all_params)):
            if model_name == 'falcon':
                #run the algorithm
                config_param.update({'idx': idx})
                sample_falcon = FALCON(csvpath = csv_path, gamma_param = config_param['gamma_param'], group = dataset_config.sens)
                sample_falcon.learn_schedule()

                eval_model = offline_eval_metric_cb(group = sample_falcon.group, context = sample_falcon.context_all,
                                                    action = sample_falcon.action_all, loss = sample_falcon.loss_all)
            
            elif model_name in ['bag', 'regopt', 'eps-greedy', 'cover']:
                #config_param = {'bag': 4, 'idx': idx}
                #call the command
                cmd = [VW, vw_path, '-b', '24', '--progress', '1']
                cmd += ['--cbify', str(n_actions)]
                #(TODO): add hyperparameter config for each algorithm in VW for command line
                cmd = config_to_pair(cmd, config_param)

                print(cmd)
                config_param.update({'idx': idx})
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                #output = os.system('vw ' + ds + ' -b 24 --oaa 2') 
                output = str(output, encoding = 'utf-8')
                loss_all, action_all = output_extraction(output)
                base_cb_model = base_cb(csvpath = csv_path, dataset_class = 'multiclass', 
                                funclass = 'linear', group_name = ['sex', 'race'])
                eval_model = offline_eval_metric_cb(group = base_cb_model.group, context = base_cb_model.context_all,
                                                    action = action_all, loss = loss_all)


            else:
                #supervised oracle for comparison
                #(TODO): will implement later
                pass
            config_result = {'model': model_name, 'config': str(config_param)}
            eval_model.group_loss_parity()
            eval_model.group_action_parity()
            config_result.update(eval_model.summary_loss)
            eval_model.offline_data.to_csv(os.path.join(output_path, dataset + '_' + model_name + '_' + str(idx) + '.csv'))
            
            all_config_results.append(config_result)

    df = pd.DataFrame(all_config_results)
    output_file = os.path.join(output_path, dataset + '_summary.csv')
    if os.path.exists(output_file) == False:
        df.to_csv(output_file)
    else:
        #already exists one file
        df_raw = pd.read_csv(output_file)
        df_concat = pd.concat([df_raw, df])
        df_concat.to_csv(output_file)
