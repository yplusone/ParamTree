from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import *
from scipy.stats import chi2
from scipy.linalg import sqrtm

from util.util import *
from feature.feature import *
from util.MyLinearModel import MyLinearRegression


LEAF_MIN_SAMPLE_NUM = 7
NUMERIC_BRANCH_NUM = 3
np.seterr(divide='ignore', invalid='ignore')

class Node():
    beta = np.array(pd.read_csv('./model/beta.csv'))
    def __init__(self, nodeid,dataset, node_features, leaf_features, types, min_size, trim, alpha, operator, modelname,coefs,
                 check_dim,checked_dims,node_class='TerminalNode', filter=[]):
        self.operator = operator
        self.modelname = modelname
        self.dataset = dataset
        self.node_features = node_features
        self.leaf_features = leaf_features
        self.types = types
        self.node_feature_num = len(node_features)
        self.leaf_feature_num = len(leaf_features)
        self.min_size = min_size
        self.trim = trim
        self.node_class = node_class
        self.children = []
        self.buckets = []
        self.alpha = alpha
        self.filters = filter
        self.checked_dims = checked_dims
        self.mape_score = 100
        self.status = True
        self.default_model = coefs
        self.nodeid = nodeid
        if len(self.dataset) > len(leaf_features) + 1 and len(node_features) != 0:
            self.add_one(data=[],check_dim=check_dim)
        else:
            self.model = self.default_model

    def estfun(self, obj):
        ans = (obj['residuals'].reshape(-1, 1) * np.hstack((np.ones(len(obj['x'])).reshape(-1, 1), obj['x']))).astype(
            float)
        # ans = ((np.log(((obj['y_pred']+1)/(obj['y_true']+1)).astype(float))/(obj['y_pred']+1)).reshape(-1,1)*np.hstack((np.ones(len(obj['x'])).reshape(-1,1),obj['x']))).astype(float)
        return ans

    def supLM(self, x, k, tlambda):
        ## use Hansen (1997) approximation
        m = Node.beta.shape[1] - 1
        if tlambda < 1:
            tau = tlambda
        else:
            tau = 1 / (1 + np.sqrt(tlambda))
        beta_ = Node.beta[(k - 1) * 25:(k * 25), :]
        dummy = beta_[:, 0:m].dot(np.power(x, np.array([t for t in range(m)])))
        dummy = dummy * (dummy > 0)
        pp = np.log(chi2.pdf(dummy, beta_[:, m]))
        if tau == 0.5:
            p = np.log(chi2.pdf(x, k))
        elif tau <= 0.01:
            p = pp[25]
        elif tau >= 0.49:
            p = np.log((np.exp(np.log(0.5 - tau) + pp[1]) + np.exp(np.log(tau - 0.49) + np.log(chi2.pdf(x, k)))) * 100)
        else:
            taua = (0.51 - tau) * 50
            tau1 = int(np.floor(taua))
            p = np.log(np.exp(np.log(tau1 + 1 - taua) + pp[tau1 - 1]) + np.exp(np.log(taua - tau1) + pp[tau1]))

        return p

    def mob_fit_fluctests(self, x, y, minsplit, trim, partvar):
        lr = MyLinearRegression()
        lr.fit(x, y)
        filter = abs(np.array(lr.coef_)) > 1e-12
        x = x[:, filter]
        if len(x[0]) == 0:
            return {}
        lr = MyLinearRegression()
        lr.fit(x, y)
        obj = {}
        obj['model'] = lr
        obj['residuals'] = np.array(y - lr.predict(x))
        obj['y_true'] = np.array(y)
        obj['y_pred'] = np.array(lr.predict(x))
        obj['x'] = x

        ## set up return values
        m = len(partvar[0])
        n = len(partvar)
        pval = np.zeros(m)
        stat = np.zeros(m)
        ifac = [False for _ in range(m)]

        # ## extract estimating functions
        process = self.estfun(obj)
        k = len(process[0])

        # ## scale process
        process = process / np.sqrt(n)
        J12 = sqrtm(process.T.dot(process))
        try:
            np.linalg.pinv(J12)
        except:
            return {}
        process = (np.linalg.pinv(J12).dot(process.T)).T

        # ## select parameters to test
        tfrom = int(trim) if trim > 1 else int(np.ceil(n * trim))
        tfrom = max(tfrom, minsplit)
        to = n - tfrom
        tlambda = ((n - tfrom) * to) / (tfrom * (n - to))

        diff = self.partvar_diff(partvar)

        # ## compute statistic and p-value for each ordering
        for i in range(m):
            if diff[i] == False:
                stat[i] = 0
                pval[i] = 100
                continue
            pvi = partvar[:, i]
            if type(pvi[0]) == str or len(set(pvi)) < 10:  # or len(set(pvi))<len(pvi)/5
                proci = process[np.argsort(pvi), :]
                ifac[i] = True

                # # re-apply factor() added to drop unused levels
                pvi = pvi[np.argsort(pvi)]
                # # compute segment weights
                count_info = Counter(pvi)
                segweights = {}  ## tapply(ww, pvi, sum)/n
                for key in count_info.keys():
                    segweights[key] = count_info[key] / n
                # # compute statistic only if at least two levels are left
                if len(segweights) < 2:
                    stat[i] = 0
                    pval[i] = 100
                else:
                    tsum = 0
                    for j in range(k):
                        df = pd.DataFrame({'proci': np.real(proci[:, j]), 'pvi': pvi}).groupby('pvi').sum()
                        for key in segweights.keys():
                            tsum += np.power(float(df.loc[key]), 2) / segweights[key]
                    stat[i] = tsum
                    pval[i] = np.log(chi2.pdf(stat[i], k * (len(segweights) - 1)))
            else:

                oi = np.argsort(pvi)
                proci = process[oi, :]
                proci = np.cumsum(proci, axis=0)
                if tfrom < to:
                    xx = np.sum(np.power(proci, 2), axis=1)
                    xx = xx[tfrom:to]
                    tt = np.array([t for t in range(tfrom, to)]) / n
                    stat[i] = np.real(max(xx / (tt * (1 - tt))))
                else:
                    stat[i] = 0

                if tfrom < to:
                    pval[i] = self.supLM(stat[i], k, tlambda)
                else:
                    pval[i] = 100
        # ## select variable with minimal p-value
        try:
            best = np.nanargmin(pval)
        except:
            best = -1
        rval = {}
        rval['pval'] = np.exp(pval)
        rval['stat'] = stat
        rval['best'] = best

        # print("pval:%.6f"%(np.exp(pval[best])))
        return rval

    def add_types(self, types):
        self.types = types

    def add_one(self, data,check_dim):

        if self.node_class == "TerminalNode":
            self.terminal_add_one(data,check_dim)
        elif self.node_class == "InnerNode":
            self.inner_add_one(data,check_dim)

    def inner_add_one(self, data,check_dim):
        if self.types[self.index]['type'] == 'numeric':
            data_pool = [[] for _ in range(len(self.children))]
            for i in range(len(data)):
                id = -1
                for idx,bucket in enumerate(self.buckets):
                    if round(data[i, self.index], 4) >bucket[0] and round(data[i, self.index], 4)<=bucket[1]:
                        id = idx
                        break
                if id == -1:
                    raise Exception("Wrong")
                data_pool[id].append(data[i,:])
        else:
            data_pool = [[] for _ in range(len(self.children))]
            for i in range(len(data)):
                id = -1
                for idx, bucket in enumerate(self.buckets):
                    if data[i, self.index] in bucket:
                        id = idx
                        break
                if id == -1:
                    self.buckets[-1].append(data[i, self.index])
                    id = idx
                    # raise Exception("Wrong")
                data_pool[id].append(data[i, :])

        for idx,item in enumerate(data_pool):
            if len(item):
                self.children[idx].add_one(np.array(item),check_dim)

    def partvar_diff(self, partvar):
        res = [False for _ in range(len(partvar[0]))]
        for i in range(len(partvar[0])):
            if len(set(partvar[:, i])) > 1:
                res[i] = True
        return res
    def get_cparam_id(self,cparam):
        for id, c in enumerate(self.node_features):
            if cparam == c:
                return id
        return -1

    def terminal_add_one(self, data,check_dim):
        if check_dim == 'End':
            self.model = self.to_terminal(self.dataset)
            return
        if len(data) != 0:
            if len(self.dataset) == 0:
                self.dataset = data
            else:
                self.dataset = np.vstack((self.dataset, data))
        if len(self.dataset) == 0:
            return
        if len(self.dataset) <= self.min_size or self.node_feature_num == 0:
            self.model = self.to_terminal(self.dataset)
            return
        if sum(self.partvar_diff(self.dataset[:, :len(self.node_features)])) == 0:
            self.model = self.to_terminal(self.dataset)
            return

        rval = self.mob_fit_fluctests(self.dataset[:, len(self.node_features):-1], self.dataset[:, -1],
                                      minsplit=self.min_size, trim=self.trim,
                                      partvar=self.dataset[:, :len(self.node_features)])
        if not len(rval):
            self.model = self.to_terminal(self.dataset)
            return
        if check_dim!=None:
            self.checked_dims.add(check_dim)
            check_id = self.get_cparam_id(check_dim)
            candidate = [check_id]
        else:
            candidate = np.argsort(rval['pval'])
        if len(rval) == 0 or rval['best'] == -1:
            self.model = self.to_terminal(self.dataset)
            return
        ignore_index = [item['index'] for item in self.filters]
        for i in candidate:
            if i in ignore_index:
                continue
            if rval['pval'][i] < self.alpha:
                if sum(self.partvar_diff(self.dataset[:, i].reshape(-1, 1))) == 0:
                    self.model = self.to_terminal(self.dataset)
                    return
                res = self.split(rval, i)
                if res == None or len(res)<2:
                    continue
                else:
                    # print(f"Begin split on {check_dim}")
                    if check_dim != None:
                        check_dim = 'End'

                    buckets = []
                    for idx,item in enumerate(res):
                        if item['value'] == None:
                            print("here")
                        # print(f"Leaf Node Sample Num: {len(item['dataset'])}")
                        leaf =  Node(self.nodeid+[idx],np.array(item['dataset']), self.node_features, self.leaf_features, self.types, self.min_size,
                                     self.trim, self.alpha, self.operator,self.modelname,filter=self.filters + [
                            {'index': item['index'], "value": item['value'],
                             "type": self.types[item['index']]['type'],"name":self.node_features[item['index']],}],coefs=self.default_model,check_dim=check_dim,checked_dims=self.checked_dims)
                        self.children.append(leaf)
                        buckets.append(item['value'])
                    self.buckets = buckets
                    self.index = i
                    self.node_class = 'InnerNode'
                    del (self.dataset)
                    break
            else:
                break
        if self.node_class == 'TerminalNode':
            self.model = self.to_terminal(self.dataset)

    def get_group_mse(self, group):
        group_array = np.array(group)
        size = float(len(group))
        if size <= 2:
            return 0
        score = self.metric(group_array[:, self.node_feature_num:-1], group_array[:, -1])
        return score

    def get_group_msle(self, group):
        group_array = np.array(group)
        size = float(len(group))
        if size <= 2:
            return 0
        y_true = group_array[:, -1]
        model = self.to_terminal(group)
        coef = np.array(model[0])
        intercept = model[1]
        result = group_array[:,self.node_feature_num:-1].astype(float).dot(coef.T) + intercept
        # k = min(5, max(y_true))
        # filter = np.where(y_true >= k)
        # mape = abs(np.array(y_true)[filter]-np.array(result)[filter])/np.array(y_true)[filter]
        # return np.mean(sorted(mape)[int(-len(mape)/4):])*100
        # return mean_squared_log_error(y_true,result,squared = False)
        return rsquared(result, y_true)

    def groupscore_metric(self, groups):
        n_instances = float(sum([len(group) for group in groups]))
        mse = 0.0
        for group in groups:
            size = float(len(group))
            score = self.get_group_mse(group)
            mse += score * (size / n_instances)
        return mse

    def metric(self, x, y_true):
        lr = MyLinearRegression()
        # lr = BayesianRidge(fit_intercept=False)
        lr.fit(x, y_true)
        result = lr.predict(x)

        return rms(y_true, result)  # mape(y_true,result)  #mean_squared_log_error(y_true,result,squared = False)

    def get_group_rscore(self,group):
        if not len(group):
            return 1
        lr = MyLinearRegression()
        # lr = BayesianRidge(fit_intercept=False)
        X = np.array(group)[:,self.node_feature_num:-1]
        y = np.array(group)[:,-1]
        lr.fit(X, y)
        return lr.score(X,y)

    def get_node_predict_mape(self, group):
        group_array = np.array(group)
        size = float(len(group))
        if size <= 2:
            return 0
        y_true = group_array[:, -1]
        result = []
        for item in group_array:
            result.append(self.get_terminal_predict(item[:-1]))

        return mape(y_true, result)

    def get_node_predict_mape_list(self, group):
        group_array = np.array(group)
        size = float(len(group))
        if size <= 2:
            return 0
        y_true = group_array[:, -1]
        result = []
        for item in group_array:
            result.append(self.get_terminal_predict(item[:-1]))

        res = [abs(y_true[i]-result[i])/y_true[i] for i in range(len(y_true))]
        return res

    def test_split_numeric(self, index, branch_num, dataset):
        res = {}
        if branch_num<NUMERIC_BRANCH_NUM:
            leaf_min = LEAF_MIN_SAMPLE_NUM
        else:
            leaf_min = np.floor(len(dataset)/branch_num)
        data = []
        last_row_value = -np.Inf
        bucket_idx = 0
        for row in dataset:
            if row[index] == last_row_value:
                data.append(row)
            elif len(data)<leaf_min:
                data.append(row)
                last_row_value = row[index]
            else:
                res[bucket_idx] = data
                data = []
                bucket_idx += 1
        if len(data):
            res[bucket_idx] = data
        return res

    def test_split_numeric_two(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] <= value:
                left.append(row)
            else:
                right.append(row)
        return np.array(left), np.array(right)

    def test_split_string(self, index, dataset):
        res = {}
        for row in dataset:
            if row[index] not in res.keys():
                res[row[index]] = []
            res[row[index]].append(row)
        return res

    def test_split_string_two(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] == value:
                left.append(row)
            else:
                right.append(row)
        return np.array(left), np.array(right)

    def recursive_split(self,index,data):
        b_index, b_value, b_score, b_groups = 999, 999, 999999999, None
        if self.types[index]['type'] == 'numeric':
            unique_val = set([round(a, 4) for a in data[:, index]])
            b_score = self.get_group_mse(data)
            for val in unique_val:
                groups = self.test_split_numeric_two(index, val, data)

                if len(groups[0]) < self.min_size or len(groups[1]) < self.min_size:
                    continue
                mse = self.groupscore_metric(groups)
                if mse < b_score:
                    b_index, b_value, b_score, b_groups = index, val, mse, groups
        else:
            unique_val = set(data[:, index])
            b_score = self.get_group_mse(data)
            for val in unique_val:
                groups = self.test_split_string_two(index, val, data)

                if len(groups[0]) < self.min_size or len(groups[1]) < self.min_size:
                    continue
                # if self.get_group_rscore(groups[0]) < 0.5 or self.get_group_rscore(groups[1]) < 0.5:
                #     continue
                mse = self.groupscore_metric(groups)
                if mse < b_score:
                    b_index, b_value, b_score, b_groups = index, val, mse, groups

        if b_groups == None:
            return None

        if len(b_groups[0]) == 0 or len(b_groups[1]) == 0:
            return None

        else:
            return {'index': b_index, 'value': b_value, 'left': b_groups[0], 'right': b_groups[1]}

    def split_attribute(self,index,groups):
        data_left = groups['left']
        rval = self.mob_fit_fluctests(data_left[:, len(self.node_features):-1], data_left[:, -1],
                                      minsplit=self.min_size, trim=self.trim,
                                      partvar=data_left[:, :len(self.node_features)])

        if len(rval) and rval['pval'][index] < self.alpha:
            left_groups = self.recursive_split(index, data_left)
            if left_groups != None:
                yield left_groups['left']
                yield left_groups['right']
        else:
            yield data_left
        data_right = groups['right']
        rval = self.mob_fit_fluctests(data_right[:, len(self.node_features):-1], data_right[:, -1],
                                      minsplit=self.min_size, trim=self.trim,
                                      partvar=data_right[:, :len(self.node_features)])

        if len(rval) and rval['pval'][index] < self.alpha:
            right_groups = self.recursive_split(index, data_right)
            if right_groups != None:
                yield right_groups['left']
                yield right_groups['right']
        else:
            yield data_right

    def split(self,rval,index):
        b_index, b_value, b_score, b_groups = 999, 999, 999999999, None
        if self.types[index]['type'] == 'numeric':
            data = self.dataset
            groups = self.recursive_split(index, data)
            new_data = []
            if groups != None:
                new_data = list(self.split_attribute(index,groups))
            l_bound = -np.inf
            res = []
            for item in new_data:
                res.append({"index":index,"value":[l_bound,max(item[:,index])],"dataset":item})
                l_bound = max(item[:, index])
            if len(res):
                res[-1]['value'][1] = np.inf
        else:
            data = self.dataset
            groups = self.recursive_split(index, data)
            new_data = []
            if groups != None:
                new_data = list(self.split_attribute(index, groups))
            res = []
            for item in new_data:
                values = list(set(item[:,index]))
                res.append({"index": index, "value": values, "dataset": item})
        return res

    def to_terminal(self, group):
        group_array = np.array(group)

        X = group_array[:, len(self.node_features):-1]
        y = group_array[:, -1]

        lr = MyLinearRegression()
        lr.fit(X,y.reshape(-1,1))
        coef = lr.coef_
        intercept = lr.intercept_

        return [coef, intercept]

    def get_terminal_predict(self, group):
        group = np.array(group)
        coef = np.array(self.model[0])
        intercept = self.model[1]
        res = group[self.node_feature_num:].astype(float).dot(coef.T) + intercept
        default_res = group[self.node_feature_num:].astype(float).dot(np.array(self.default_model[0]).T) + self.default_model[1]
        if self.get_group_rscore(self.dataset)<0.1:
            # X = self.dataset[:, len(self.node_features):-1]
            # y = self.dataset[:, -1]
            # coef = [0.01, 0.005, 0.01, 1, 4, 0.005, 0]
            # y_p = X.dot(np.array(coef).T)
            # lr = LinearRegression(fit_intercept=False)
            # lr.fit(y_p.reshape(-1,1),y.reshape(-1,1))
            # scale = lr.coef_[0][0]
            # res = group[self.node_feature_num:].astype(float).dot(np.array(coef).T)*scale
            return default_res

        if default_res < 1  or res < 1:
            return default_res
            # print(group[self.parentop_index])
            # return default_res
        # if default_res < 0.001:
        #     return default_res

        return res

    def plot_terminal(self, dirname="", i=0):
        group_array = np.array(self.dataset)
        size = float(len(group_array))
        y_true = group_array[:, -1]
        y_pred = []
        for item in group_array:
            y_pred.append(self.get_terminal_predict(item[:-1]))
        plt.figure()
        plt.scatter(y_pred, y_true, s=1)
        plt.xlabel('predict_time')
        plt.ylabel('actual_time')

        line = f"rms:{round(self.get_group_mse(self.dataset), 2)},mape:{round(self.get_node_predict_mape(self.dataset), 2)},msle:{round(self.get_group_msle(self.dataset), 2)}"
        plt.title(line)
        x0 = [t for t in range(int(np.max(y_pred)) + 2)]
        y0 = [t for t in x0]
        plt.plot(x0, y0, 'r')
        # plt.show()
        if dirname != "":
            plt.savefig(dirname + f"{str(i)}.png", dpi=150)
        else:
            plt.show()

    def normalizeValue(self,sample_set):
        sample=sample_set[0]
        normalized_value_list=[]
        types=[]
        for i,f_name in enumerate(self.node_features):
            normalized_list=[]
            type={}
            type["feature_name"]=f_name
            value_list=sample_set[:,i]
            value_set=Counter(value_list)
            type["value_set"] = value_set
            type["value_set_len"]=len(value_set)
            data=sample[i]
            if isinstance(sample[i],str):
                type["type"]="category"
                if(len(value_set)==1):
                    normalized_list=[0 for j in range(len(value_list))]
                else:
                    max_key = max(value_set, key = value_set.get)
                    normalized_list=[0 if j==max_key else 0.5 for j in value_list]
            else:
                type["type"]="numeric"
                if(len(value_set)==1):
                    normalized_list=[0 for j in range(len(value_list))]
                else:
                    mean = np.mean(value_list)
                    std = np.std(value_list)
                    normalized_list=[abs((j-mean)/std) for j in value_list]
            normalized_value_list.append(normalized_list)
            types.append(type)
        dim_info = types
        return dim_info,np.array(normalized_value_list)




