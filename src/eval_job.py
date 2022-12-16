from src.datasets.tabular import ADULT, BRFSS, CANDC, COMPAS, GERMAN
from src.datasets.tabular import get_dataset_config
import os, subprocess
import numpy as np
import pandas as pd

from src.models.cb.base_cb import base_cb
from src.models.cb.FALCON import FALCON
from src.models.cb.CFTL import CFTL
from src.models.cb import config_choice, config_to_pair, FALCON_FAIR
from src.utils import generator
from src import IND_TIME_WINDOW

from src.eval_metric_cb import offline_eval_metric_cb
from src.utils.joint_regret import output_extraction

VW = 'vw'

#(TODO): remind to incorporate the multi-action case...
def run_jobs():
    model_classes = ['cftl', 'falcon', 'bag', 'regcbopt', 'eps-greedy', 'cover', 'supervised']

    dataset = CANDC
    n_actions = 2
    csv_path = 'src/datasets/' + dataset + '/' + dataset + '_' + str(n_actions) + '.csv'
    vw_path = 'src/datasets/' + dataset + '/' + dataset + '_' + str(n_actions) + '.vw.gz'
    output_path = 'results/' + dataset 

    dataset_kwargs = {'root_dir': 'src/datasets'}
    filepath = dataset_kwargs['root_dir'] + '/' + dataset
    dataset_config = get_dataset_config(dataset, **dataset_kwargs)

    all_config_results = []
    for model_name in model_classes:
    #['regcbopt', 'eps-greedy', 'cover', 'supervised', 'falcon', 'bag']:
        #model_name = 'falcon'
        
        #config_result = {'model': model_name, 'config': str(config_param)}
        #obtain the config from that model class
        all_params = config_choice(model_name)
        print(model_name, all_params)
        cmd = [VW, vw_path, '-b', '24', '--progress', '1']
        for idx, config_param in enumerate(generator(all_params)):
            print(config_param)
            if model_name == 'falcon':
                #run the algorithm
                config_param.update({'idx': idx})
                sample_falcon = FALCON(csvpath = csv_path, gamma_param = config_param['gamma_param'], group = dataset_config.sens,
                                         action_parity = 0.5, ind_parity = 0.5)
                sample_falcon.learn_schedule(action_fair_type = config_param['action_fair_type'],
                                            loss_fair_type = config_param['loss_fair_type'])
                eval_model = offline_eval_metric_cb(group = sample_falcon.group, context = sample_falcon.context_all,
                                                    action = sample_falcon.chosen_action_all.astype(int), loss = sample_falcon.loss_all)
            
            elif model_name == "cftl":
                config_param.update({'idx': idx})
                sample_cftl = CFTL(csv_path, group=dataset_config.sens)
                sample_cftl.fit()
                sample_cftl.eval()
                eval_model = offline_eval_metric_cb(group=sample_cftl.group, context=sample_cftl.context_all,
                                                    action=sample_cftl.chosen_action_all.astype(int), loss=sample_cftl.loss_all)

            else:
                #call the VW model
                if model_name in ['bag', 'regcbopt', 'eps-greedy', 'cover']:
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
                                    funclass = 'linear', group_name = dataset_config.sens)
                    eval_model = offline_eval_metric_cb(group = base_cb_model.group, context = base_cb_model.context_all,
                                                        action = action_all, loss = loss_all)


                elif model_name  == 'supervised':
                    #supervised oracle for comparison
                    #(TODO): we can directly change to sklearn realization...
                    cmd = [VW, vw_path, '-b', '24', '--progress', '1', '--oaa', str(n_actions)]
                    print(cmd)
                    config_param = {'linear': 0}
                    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                    #output = os.system('vw ' + ds + ' -b 24 --oaa 2') 
                    output = str(output, encoding = 'utf-8')
                    loss_all, action_all = output_extraction(output)
                    base_cb_model = base_cb(csvpath = csv_path, dataset_class = 'multiclass', 
                                    funclass = 'linear', group_name = dataset_config.sens)
                    print(len(base_cb_model.group))
                    eval_model = offline_eval_metric_cb(group = base_cb_model.group, context = base_cb_model.context_all,
                                                        action = action_all, loss = loss_all)

                else:
                    raise NotImplementedError
            config_result = {'model': model_name, 'config': str(config_param)}
            eval_model.group_loss_parity()
            eval_model.group_action_parity()
            eval_model.individual_loss_parity()
            config_result.update(eval_model.summary_loss)
            if FALCON_FAIR == False:

                eval_model.offline_data.to_csv(os.path.join(output_path, dataset + '_' + model_name + '_' + str(idx) + '.csv'))
            else:
                
                eval_model.offline_data.to_csv(os.path.join(output_path, dataset + '_' + model_name + '_' + str(idx) + '_fair.csv'))
            all_config_results.append(config_result)

    df = pd.DataFrame(all_config_results)
    if FALCON_FAIR == False:
        output_file = os.path.join(output_path, dataset + '_summary.csv')
    else:
        output_file = os.path.join(output_path, dataset + '_summary_fair.csv')
    if os.path.exists(output_file) == False:
        df.to_csv(output_file)
    else:
        #already exists one file
        df_raw = pd.read_csv(output_file)
        df_concat = pd.concat([df_raw, df])
        df_concat.to_csv(output_file, index = None)


if __name__ == "__main__":
    run_jobs()