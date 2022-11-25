import src.datasets.tabular
import src.datasets.utils
from src.datasets.tabular import get_dataset_config
import vowpalwabbit
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_dataset(dataset="adult", root="src/datasets"):
     ## get data from csv file
    dataset_kwargs = {'root_dir': root}
    dataset_config = get_dataset_config(dataset, **dataset_kwargs)
    dset = src.datasets.utils.get_dataset(dataset_config)
    return dset


def to_vw_format(context, actions, cb_label=None):
    if cb_label is not None:
        chosen_action, cost, prob = cb_label
    res = "shared |"
    res += ' '.join(
                "{}:{:.6f}".format(j, val) 
                for j, val in enumerate(context) if val != 0
                # zip(list(context.index), list(context))
            ) + '\n'
    
    for action in actions:
        if cb_label is not None and action == chosen_action:
            res += "0:{}:{} ".format(cost, prob)
        res += "|Action a={} \n".format(action)
    # Strip the last newline
    return res[:-1]


def get_action(model, context, actions):
    vw_context = to_vw_format(context, actions)
    pmf = model.predict(vw_context)
    pmf = [p/sum(pmf) for p in pmf]
    action = np.random.choice(actions, p=pmf)
    prob = np.array(pmf)[np.array(actions)==action][0]
    return action, prob


def train(dataset, model):
    cumcost = 0
    records = {}
    reward = []

    ## preprocess and split dataset
    X_tr, y_tr, _, _, _, _ =  dataset.get_data()
    actions = [1, 2]
    
    for i in range(X_tr.shape[0]):
        """
        sample is like: 
        action:cost:prob|feat1:val1 feat2:val2
        """
        feat, action_true = X_tr.loc[i], y_tr.loc[i] + 1

        # predict action-prob
        action_pred, prob = get_action(model, feat, actions)     
        records.update({i:{"pred":action_pred, "prob": prob, "true":action_true}})

        # get cost
        cost = -1 if action_true == action_pred else 0
        cumcost += cost
        reward.append(-1 * cumcost / (i+1))

        # train
        vw_format = model.parse(
            to_vw_format(feat, actions, (action_true, -1, 1)),
            vowpalwabbit.LabelType.CONTEXTUAL_BANDIT,
        )   
        model.learn(vw_format)

    return reward, records
        

def plot_ctr(reward):
    plt.plot(range(10, len(reward)), reward[10:])
    plt.xlabel("num_iterations", fontsize=14)
    plt.ylabel("reward", fontsize=14)
    plt.ylim([0, 1])
    plt.show()


def main():
    dset = get_dataset()

    vw = vowpalwabbit.Workspace("--cb_explore 2 --bag 4")

    reward, records = train(dset, vw)
    plot_ctr(reward)
    return records



if __name__ == "__main__":
    main()