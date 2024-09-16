import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from doepy import build
from sklearn.preprocessing import LabelEncoder
from SALib.sample import saltelli
from SALib.analyze import sobol
import warnings
import random

from database_util.database_info import *
from database_util.db_connector import *
from feature.plan import *
from feature.feature import *
from .ps_rbf import RBF
from util.util import rsquared,get_plantrees
from feature.infos import all_cparams
from .lhs import LHS



warnings.filterwarnings("ignore")
np.random.seed(1)
pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)

rs_data_dir = './data/rsdata/'
model_dir = './saved_models/rsmodel.pickle'
class Knob_rs:
    def __init__(self,db_name='tpcds',load = True):
        self.scheme_info = Database_info(db_name)
        self.plan_tool = Plan_class(self.scheme_info)
        self.feature_tool = FeatureExtract()
        self.encoders = {}
        # self.data = self.load_lhs_data(file_name)
        self.rs_models = {}
        if load and os.path.exists(model_dir):
            file = open(model_dir, "rb")
            self.rs_models = pickle.load(file)
            print(f"Model load from {model_dir}")
        else:
            self.init_models()
            file = open(model_dir, "wb")
            pickle.dump(self.rs_models, file)
            print(f"Model saved in {model_dir}")

    def load_lhs_data(self,file_name):
        opdatas = {"Seq Scan": [],
                   "Index Scan": [],
                   "Index Only Scan": [],
                   "Sort": [],
                   "Hash Join": [],
                   "Nested Loop": [],
                   "Merge Join": [],
                   "Aggregate": [],
                   }

        def get_op_info(plan, config):
            if 'Plans' in plan.keys():
                for item in plan['Plans']:
                    get_op_info(item, config)
            if plan['Node Type'] in opdatas.keys():
                temp_config = config.copy()
                res = self.plan_tool.get_op_info(plan, execute=True)
                feat = self.feature_tool.get_model_raw_feature(res, plan['Node Type'], 'runtime_cost')
                for item in ['Nt', 'No', 'Ni', 'Ns', 'Nr', 'Np', 'Nm', 'y']:
                    feat.pop(item)
                feat.update(temp_config)
                if plan['Node Type'] in ["Seq Scan", "Index Scan", "Index Only Scan"]:
                    feat['actual'] = plan['Actual Total Time']
                    feat['estimate'] = plan['Total Cost']
                elif plan['Node Type'] in ['Sort', 'Aggregate']:
                    feat['actual'] = plan['Actual Total Time'] - plan['Plans'][0]['Actual Total Time']
                    feat['estimate'] = plan['Total Cost'] - plan['Plans'][0]['Total Cost']
                elif plan['Node Type'] in ['Hash Join', 'Merge Join', 'Nested Loop']:
                    feat['actual'] = plan['Actual Total Time'] - plan['Plans'][0]['Actual Total Time'] - \
                                            plan['Plans'][1]['Actual Total Time']
                    feat['estimate'] = plan['Total Cost'] - plan['Plans'][0]['Total Cost'] - plan['Plans'][1][
                        'Total Cost']

                opdatas[plan['Node Type']].append(feat)
            else:
                pass
                # print(plan['Node Type'])
        plan_trees = get_plantrees([file_name],subplan=True)

        for item in plan_trees:
            res_dict = item['planinfo']
            config = item['config'] if 'config' in item.keys() else {}
            get_op_info(res_dict['Plan'], config)

        return opdatas

    def get_float_df(self,df,op):
        self.encoders[op] = {}
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.encoders[op][col] = le
        df['scale'] = df['actual'] / df['estimate']
        return df

    def get_name_to_index(self,name,names):
        for idx,n in enumerate(names):
            if n==name:
                return idx
        return -1

    def get_sensitivity(self,op,filters,checked_dims=None,knob_change=False):
        bounds = self.rs_models[op].bounds.T
        for filter in filters:
            if filter['type']!='numeric':
                continue
            idx = self.get_name_to_index(filter['index'],self.rs_models[op].xcolumns)
            bounds[idx] = filter['value']
            if bounds[idx][1]<=bounds[idx][0]:
                bounds[idx][1] = bounds[idx][0] +max(min(abs(bounds[idx][0])/10,1),0.1)
        if checked_dims == None:
            checked_dims = [filter['name'] for filter in filters]
        new_bounds = []
        keys = []
        for idx,key in enumerate(self.rs_models[op].xcolumns):
            if bounds[idx][1] > bounds[idx][0]:
                keys.append(key)
                new_bounds.append(bounds[idx])
            # else:
            #     print("here")
        problem = {
            'num_vars': len(keys),
            'names': keys,
            'bounds': new_bounds
        }
        try:
            param_values = saltelli.sample(problem, 10,calc_second_order=False)
        except:
            print(traceback.print_exc())
            print("here")

        samples = np.zeros((len(param_values),len(self.rs_models[op].xcolumns)))
        i = 0
        for idx,key in enumerate(self.rs_models[op].xcolumns):
            if bounds[idx][1] > bounds[idx][0]:
                samples[:,idx] = param_values[:,i]
                i += 1
            else:
                samples[:,idx] = bounds[idx][0]

        y = self.rs_models[op].interp(samples)
        Si = sobol.analyze(problem, y.reshape(-1),calc_second_order = False)
        s = pd.Series(Si['S1'],index = keys).sort_values(ascending=False)
        #return random.choice(s.index)
        for idx in s.index:
            if not knob_change:
                if idx not in checked_dims and idx not in dbms_cparams:
                    return idx
            else:
                if idx not in checked_dims:  #idx in all_cparams[op]['dbms'] and
                    return idx
        return -1

    def init_models(self):
        for op in ["Seq Scan", "Index Scan", "Index Only Scan",'Hash Join', 'Merge Join', 'Nested Loop','Sort', 'Aggregate']:
            print(op)
            self.build_rs(op)

    def large_bounds(self,bounds,sample,origin_bounds):
        old_cost = 1
        out_bounds = {}
        area_percent = 1
        for col in bounds.keys():
            old_cost *= (bounds[col][1] - bounds[col][0]) / (origin_bounds[col][1] - origin_bounds[col][0])
            if sample[col]<=bounds[col][0]:

                out_bounds[col] = [sample[col],bounds[col][1]]
                area_percent *= (bounds[col][1]-sample[col])/(origin_bounds[col][1]-origin_bounds[col][0])
            elif sample[col]>bounds[col][1]:
                out_bounds[col] = [bounds[col][0],sample[col]]
                area_percent *= (sample[col] - bounds[col][0]) / (origin_bounds[col][1] - origin_bounds[col][0])
            else:
                out_bounds[col] = bounds[col]
                area_percent *= (bounds[col][1] - bounds[col][0]) / (origin_bounds[col][1] - origin_bounds[col][0])
        if old_cost <1e-12:
            cost = 1
        else:
            cost = area_percent/old_cost
        return cost,out_bounds,area_percent




    def test_fuzzy_area(self,op,models):
        lhs_bounds = {}
        for idx,bound in enumerate(models[0].bounds.T):
            lhs_bounds[models[0].xcolumns[idx]] = bound

        samples = build.space_filling_lhs(
            lhs_bounds,
            num_samples=10000
        )
        pred = [models[t].interp(np.array(samples)) for t in range(len(models))]
        diff = np.array(pred).std(axis=0)/abs(np.array(pred).mean(axis=0))
        diff_n = (diff-min(diff))/(max(diff)-min(diff))
        sort_indices = np.argsort(diff.reshape(-1))[::-1]

        roi_bound = 1e-4
        for j in range(5):
            new_bounds = {}
            count = 0
            now_area_percent = 0
            for i in sort_indices:
                # initialize bounds
                if not len(new_bounds):
                    for col in samples.columns:
                        new_bounds[col] = [samples[col].iloc[i],samples[col].iloc[i]+1e-8]
                else:
                    cost,out_bounds,area_percent = self.large_bounds(new_bounds,samples.iloc[i],lhs_bounds)
                    profit = diff_n[i][0]
                    roi = profit/cost
                    # print(profit,roi,area_percent)
                    if roi > roi_bound:
                        count += 1
                        now_area_percent = area_percent
                        # print(area_percent)
                        new_bounds = out_bounds
                if now_area_percent>1e-2:
                    break
            if count<10:
                roi_bound /= 10
            else:
                break

        print(count,now_area_percent)
        return new_bounds

    def load_files(self,op,al=True):

        file_name = f'{rs_data_dir}{op}.txt'
        df = self.get_float_df(pd.DataFrame(self.load_lhs_data(file_name)[op]),op)
        file_name = f'{al_data_dir}{op}.txt'
        if al and os.path.exists(file_name):
            data = self.load_lhs_data(file_name)[op]
            if len(data):
                df_al = self.get_float_df(pd.DataFrame(data),op)
                df = df.append(df_al)
        df = df[df['actual']>1]
        return df

    def test_score(self,important_dim_set,op,al=True,plot = False):
        file_name = f'{test_data_dir}{op}.txt'
        df = self.get_float_df(pd.DataFrame(self.load_lhs_data(file_name)[op]), op)
        df = df[df['actual'] > 1]
        df_train = self.load_files(op,al)
        rbf = RBF('linear', 1)
        rbf.fit(df_train[important_dim_set], df_train['scale'])
        y_pred = rbf.interp(np.array(df[important_dim_set]))
        fit_score = rsquared(y_pred.reshape(-1), df['scale'])
        if plot:
            plt.scatter(y_pred, df['scale'])
            plt.show()
            plt.title(op)
        print(fit_score)

    def build_rs(self,op):
        important_dim_set = all_cparams
        file_name = f'{rs_data_dir}{op}.txt'
        df = self.get_float_df(pd.DataFrame(self.load_lhs_data(file_name)[op]),op)
        rbf = RBF('Gaussian',1)
        df_t = df.groupby(important_dim_set,as_index=False).mean()
        n_sample = len(df_t)
        rbf.fit(df_t[important_dim_set].iloc[:int(n_sample*0.9)],df_t['scale'].iloc[:int(n_sample*0.9)])
        # y_pred = rbf.interp(np.array(df_t[important_dim_set])[int(n_sample*0.9):])
        # plt.scatter(y_pred,df_t['scale'][int(n_sample*0.9):])
        # plt.show()
        # rbf.plot_2d(axes = [0,1])
        # fit_score = rsquared(y_pred.reshape(-1),df_t['scale'][int(n_sample*0.9):])
        # print(f"{op} Fit score:%.2f"%(fit_score))
        self.rs_models[op] = rbf
        return rbf

    def test_rs(self,op):
        important_dim_set = all_cparams[op]['dbms']+all_cparams[op]['query']
        df = self.load_files(op)
        df_t = df.groupby(important_dim_set,as_index=False).mean()
        n_sample = len(df_t)
        test_df = df_t.iloc[int(n_sample * 0.9):]
        df_t = df_t.iloc[:int(n_sample * 0.9)]
        for _ in range(1):
            rbf = RBF('linear', 1)
            df_h = df_t.sample(frac=0.6)
            rbf.fit(df_h[important_dim_set], df_h['scale'])
            y_pred = rbf.interp(np.array(test_df[important_dim_set]))
            plt.scatter(y_pred,test_df['scale'])
            plt.show()
            # rbf.plot_2d(axes = [0,1])
            fit_score = rsquared(y_pred.reshape(-1),test_df['scale'])
            print(f"{op} Fit score:%.2f"%(fit_score))

        # self.rs_models[op] = rbf
        return rbf

if __name__ == '__main__':

    db_name = 'tpcds'
    kb = Knob_rs(db_name,load=False)
    # rbf = kb.test_rs('Index Only Scan')
    # print(rbf)
    # for op in ['Sort']:
    #     print(op)
    #     kb.active_learning_rs(op)

    # for op in ["Seq Scan","Index Scan",'Index Only Scan','Sort','Hash Join', 'Merge Join', 'Nested Loop', 'Aggregate']:
    #     important_dim_set = all_cparams[op]['dbms'] + all_cparams[op]['query']
    #     print(op)
    #     kb.test_score(important_dim_set,op,al=True,plot=True)


    # kb.build_rs(op="Sort")
    for op in ["Index Scan", "Index Only Scan",'Hash Join', 'Merge Join', 'Nested Loop','Sort', 'Aggregate']:
        kb.build_rs(op = op)
    filters = [{'index': 'IndexCorrelation', 'value': -0.0034, 'sign': '>', 'type': 'numeric'}]
    kb.get_sensitivity(op="Index Scan",filters = filters)

