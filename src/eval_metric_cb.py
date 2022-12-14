import pandas as pd
import numpy as np
from src import C1, C2, IND_TIME_WINDOW

class offline_eval_metric_cb:
    def __init__(self, group, context, action, loss, prob = None):
        """
        each parameter above is a T-dimension vector, collected from the online algorithm
        (currently in the static version...)
        group: the t-th incoming individual group, each in [m];
        context: the t-th incoming individual feature (excluding group info), each is R_{dx} vector;
        action: the t-th individual's action chosen by the agent, each in [K];
        loss: the t-th loss (one step, not cummulative up to now) obtained from that individual, each in R;
        prob: the chosen action prob (propensity score), each in Delta_{K};
        """
        self.group = np.array(group)
        self.individual_num = len(group)
        self.group_num = len(set(group))
        self.context = np.array(context)
        self.action = np.array(action)
        self.action_num = len(set(action))
        self.loss = np.array(loss)
        self.prob = np.array(prob)
        #create a dataframe including {g_t, a_t, r_t} t = 1^T
        self.offline_data = pd.DataFrame({"group": group, "action": action, "step_loss": loss})
        self.summary_loss = {}
        self.expected_action_parity_disable = True
    def group_loss_parity(self):
        cumu_group_loss = 0.001 * np.ones((self.individual_num, self.group_num))
        cumu_group_num = 0.001 * np.ones((self.individual_num, self.group_num))
        #parity between avg loss
        loss_parity = np.zeros(self.individual_num)

        #initialization
        cumu_group_loss[0][self.group[0]] = self.loss[0]
        cumu_group_num[0][self.group[0]] = 1
        loss_parity[0] = self.loss[0]
        for i in range(1, self.individual_num):
            for k in range(self.group_num):
                cumu_group_loss[i][k] = cumu_group_loss[i - 1][k]
                cumu_group_num[i][k] = cumu_group_num[i - 1][k]

            cumu_group_loss[i][self.group[i]] += self.loss[i]
            cumu_group_num[i][self.group[i]] += 1
            #compute the loss parity at each time t
            avg_loss = [cumu_group_loss[i][k] / cumu_group_num[i][k] for k in range(self.group_num)]
            avg_loss.sort()
            loss_parity[i] = avg_loss[-1] - avg_loss[0]
        cumu_group_loss = cumu_group_loss.T
        cumu_group_num = cumu_group_num.T
        cumu_loss = np.sum(cumu_group_loss, axis = 0)
        
        self.offline_data['cumu_loss'] = cumu_loss
        self.summary_loss = {'cumu_loss': cumu_loss[-1], 'avg_loss': cumu_loss[-1] / self.individual_num}
        for k in range(self.group_num):
            self.offline_data['cumu_loss_' + str(k)] = list(cumu_group_loss[k])
            self.summary_loss['cumu_loss_' + str(k)] = cumu_group_loss[k][-1]
            self.summary_loss['avg_loss_' + str(k)] = cumu_group_loss[k][-1] / cumu_group_num[k][-1]
            self.offline_data['cumu_num_' + str(k)] = list(cumu_group_num[k])
        self.offline_data['loss_parity'] = list(loss_parity)
        self.summary_loss['loss_parity'] = loss_parity[-1]
        return self.summary_loss

    
    def group_action_parity_computation(self, prefix = 'realized'):
        """
        realized action parity: from the realized action; 
        expected action parity: from the expected action simplex.
        """
        ## 0.001 prevents the case where 0 is the denominator
        avg_action_num = [0.001] * self.action_num
        group_action_num = 0.001 * np.ones((self.group_num, self.action_num))
        ## initialization
        avg_action_num[self.action[0]] += 1
        group_action_num[self.group[0]][self.action[0]] += 1
        max_group_action_parity = np.zeros(self.individual_num)
        for i in range(1, self.individual_num):
            ## update num
            if prefix == 'realized':
                avg_action_num[self.action[i]] += 1
                group_action_num[self.group[i]][self.action[i]] += 1
            elif prefix == 'expected':
                #see from the prob table
                for j in range(self.action_num):
                    avg_action_num[j] += self.prob[i][j]
                    group_action_num[self.group[i]][j] += self.prob[i][j]
            else:
                raise NotImplementedError
            ## update parity directly
            group_action_parity = np.zeros(self.group_num)
            for k in range(self.group_num):
                group_action_parity[k] = np.sum([abs(avg_action_num[j]/(i + 1) - group_action_num[k][j]/sum(group_action_num[k])) 
                                                for j in range(self.action_num)])
            max_group_action_parity[i] = max(group_action_parity)
        #print(avg_action_num, group_action_num)
        self.offline_data['group_action_parity_' + prefix] = list(max_group_action_parity)
        self.summary_loss['group_action_parity_' + prefix] = max_group_action_parity[-1]
        return self.summary_loss

    def group_action_parity(self):
        self.group_action_parity_computation('realized')
        if self.expected_action_parity_disable == False:
            self.group_action_parity_computation('expected')
        
    def individual_loss_parity_computation(self, prefix = 'realized', order = None):
        """
        order: one-side or two-side (i.e. full time or hindsight), default None means two-side, i.e. the action for one individual is fair to both former and latter.
        time_window: default None mean compute the unfairness from the start time until now, otherwise only compute K-step unfairness (moving window)
        action_distance encoding (realized): if action are different, label action_distance to be 1 otherwise 0.
        =sum indicate function {d(a_1, a_2) >= c1 * d(x1, x_2) + c2}
        """
        loss_parity = np.zeros(self.individual_num) 
        c1 = C1
        c2 = C2
        time_window = IND_TIME_WINDOW

        if order == None:
            for t1 in range(1, self.individual_num):
                loss_parity[t1] = loss_parity[t1 - 1]
                if time_window == None:
                    time_window = t1
                for t2 in range(max(0, t1 - time_window), t1):
                    if self.action[t1] != self.action[t2]:
                        #compute the feature distance to see if there is some unfairness
                        #print(c1 * np.linalg.norm(self.context[t1] - self.context[t2]) + c2)
                        if c1 * np.linalg.norm(self.context[t1] - self.context[t2]) + c2 <= 1:
                            #print(c1 * np.linalg.norm(self.context[t1] - self.context[t2]) + c2)
                            loss_parity[t1] += 1
            self.offline_data['individual_loss_parity_' + prefix] = list([loss_parity[i] / (min(IND_TIME_WINDOW, i + 1) * (i + 1)) for i in range(self.individual_num)])
            self.summary_loss['individual_loss_parity_' + prefix] = loss_parity[-1] / (IND_TIME_WINDOW * self.individual_num)
        else:
            raise NotImplementedError


       
          

    def individual_loss_parity(self, order = None):
        #involve the interaction with the context...
        self.individual_loss_parity_computation('realized', order)
