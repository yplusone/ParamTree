import json
import numpy as np
from database_util.db_connector import *
from model.query_inference import QueryInference,get_plantrees
import random
from model.caculate_rparams import Calculate_5params
from database_util.database_info import *
from sklearn.linear_model import LinearRegression

def get_rparams_and_scale(scheme_info,files):
    data = []
    for file_name in files:
        with open(file_name, 'r') as f:
            data += f.readlines()
    plan_trees = []

    for item in data:
        try:
            plan_json = json.loads(item.strip())
            plan_trees.append(plan_json['planinfo'])

        except:
            pass
    rparam_tool = Calculate_5params(plan_trees, scheme_info)
    coef = rparam_tool.coef
    scale = rparam_tool.scale
    return np.array([np.array(coef), 0],dtype=object),scale

def get_scale(files):
    data = []
    for file_name in files:
        with open(file_name, 'r') as f:
            data += f.readlines()
    plan_trees = [json.loads(item.strip())['planinfo'] for item in data]
    pred = []
    true = []

    def get_op_info(plan):
        if 'Plans' in plan.keys():
            for item in plan['Plans']:
                get_op_info(item)
        if plan['Node Type'] in ['Seq Scan']:
            pred.append(plan['Total Cost'])
            true.append(plan['Actual Total Time'])
        else:
            pass

    for plan in plan_trees:
        get_op_info(plan['Plan'])
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.array(true).reshape(-1,1),np.array(pred).reshape(-1,1))
    # print(lr.score())
    return lr.coef_[0][0]

def get_tuned_rparam_cost(db_name,train_files,test_plans,subplan,args):
    scheme_info = Database_info(db_name)
    coefs,scale = get_rparams_and_scale(scheme_info, train_files)
    # print(coefs)
    tool = QueryInference(db_name, scale=scale, coefs = coefs,args = args,load=False,load_model_name=None)
    # for test_file in test_files:
    y_pred = tool.predict(test_plans)
    y_true = [t['planinfo']['Plan']['Actual Total Time'] for t in test_plans]
    lr = LinearRegression(fit_intercept=False, positive=True)
    lr.fit(np.array(y_pred).reshape(-1, 1), np.array(y_true).reshape(-1, 1))
    y_pred = list(lr.predict(np.array(y_pred).reshape(-1, 1)).reshape(-1))
    return y_pred,y_true

def get_default_rparam_cost(db_name,test_plans,subplan,args):
    coef = [0.01,0.005,0.01,1,4,0.005,0]
    coefs = np.array([np.array(coef), 0],dtype=object)

    tool = QueryInference(db_name, scale=1000, coefs = coefs,args =  args,load=False,load_model_name=None)
    y_pred = tool.predict(test_plans)
    lr = LinearRegression()
    # for test_file in test_files:

    y_true = [t['planinfo']['Plan']['Actual Total Time'] for t in test_plans]
    lr.fit(np.array(y_pred).reshape(-1,1), np.array(y_true).reshape(-1,1))
    y_pred = lr.predict(np.array(y_pred).reshape(-1,1))

    # res = get_test_results(y_pred, y_true)
    return list(y_pred.reshape(-1)),list(y_true)
