from src.datasets.tabular import get_dataset_config
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
import torch 
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from src.models.cb.base_cb import base_cb


class CFTL(base_cb):

    def __init__(self, csvpath, dataset_class="oml", funclass="linear", group=None, K=60):
        self.K = K
        base_cb.__init__(self, csvpath, dataset_class, funclass, group)

    def get_bounds(self, X_k, x_t, Y_k, lambd = 0.5, gamma=0.5):
        n, d = np.shape(X_k)
        index = faiss.IndexFlatL2(d)
        index.add(np.ascontiguousarray(X_k))
        D, I = index.search(x_t, n)
        U = min(Y_k[I[0]] + lambd * D[0] + gamma)
        L = max(Y_k[I[0]] - lambd * D[0] - gamma)
        return L, U 

    def fit(self):
        
        X, y = self.context_all, self.action_all
        K = self.K
        self.y_logit = np.ones(len(y)) * 0.5
        model = LogisticRegression().fit(X[ : K], y[ : K])
        for t in tqdm(range(K, len(X))):
            
            x_t, y_t = X[ : t], y[ : t]
            y_pred_t = model.predict_proba(X[[t], :])[0, 1]
            self.y_logit[t] = np.log(y_pred_t / (1 - y_pred_t)) # log odds

            idx_start = max(t - K, 0)
            X_k = X[idx_start : t - 1]
            Y_k = self.y_logit[idx_start : t - 1]
            
            l, u = self.get_bounds(X_k.astype("float32"), X[[t], :].astype("float32"), Y_k)
            self.y_logit[t] = np.clip(self.y_logit[t], a_min=l, a_max=u)
            if t % 1000 == 0:
                model = LogisticRegression(max_iter=500).fit(x_t, y_t)
            

    def eval(self, plot=False):
        
        y = self.action_all
        input = torch.tensor(self.y_logit).float()
        target = torch.tensor(y).float()
        loss = nn.BCEWithLogitsLoss(reduction="none")(input, target)
        loss = loss.detach().numpy()
        self.loss_all = np.cumsum(loss) / np.arange(1, len(loss) + 1)
        self.chosen_action_all = (1 / (1 + np.exp(-self.y_logit)) > 0.5).astype(int)
        if plot:
            pd.Series(self.loss_all).plot(title="Training Loss", ylabel="Loss", xlabel="Steps")
        

            

if __name__ == "__main__":
    # dset = get_dataset(root="3_fairness/fa-alg/downloads")
    # X_tr, y_tr, _, _, _, _ =  dset.get_data()
    csv_path = '3_fairness/fa-alg/downloads/adult/adult_2.csv'
    dataset_kwargs = {'root_dir': 'src/datasets'}
    dataset_config = get_dataset_config("adult", **dataset_kwargs)
    model = CFTL(csv_path, group=dataset_config.sens)
    model.fit()
    model.eval(plot=True)
