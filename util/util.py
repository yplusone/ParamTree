import numpy as np
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,mean_squared_log_error
import json
import queue
import re

def rsquared(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mean = np.mean(y_true)
    SST = np.sum(np.square(y_true - mean)) + np.exp(-8)
    SSReg = np.sum(np.square(y_true - y_pred))
    score = 1 - SSReg / SST
    return score

def mape(Y_actual, Y_Predicted):
    k = min(1, max(Y_actual))
    filter = np.where(Y_actual >= k)
    mape = np.median(np.abs(
        (np.array(Y_actual)[filter] - np.array(Y_Predicted)[filter]) / (np.array(Y_actual)[filter]))) * 100
    return mape

def qerror(y_pred, y_true,percent=50,score = False):
    if len(y_pred) != len(y_true):
        raise ValueError("Dimension Wrong")
    r = []
    for i in range(len(y_pred)):
        r.append(max((y_true[i]+1e-6) / (y_pred[i]+1e-6), (y_pred[i]+1e-6) / ((y_true[i])+1e-6)))
    return r

def rms(Y_actual, Y_Predicted):
    return mean_squared_error(Y_Predicted,Y_actual, squared=False)

def deal_plan(plan):
    add_parent_info(plan)
    add_initplan_info(plan)

def add_parent_info(plan):
    if not plan:
        return
    plans = queue.Queue()
    plans.put(plan['Plan'])
    plan['Plan']['parent'] = 'None'
    while not plans.empty():
        tt = plans.get()
        parentop = tt['Node Type']
        if 'Plans' in tt.keys():
            for item in tt['Plans']:
                item['parent'] = parentop
                plans.put(item)


def add_initplan_info(plan):
    def get_nodes(node):
        if 'Plans' in node.keys():
            for t in node['Plans']:
                for x in get_nodes(t):
                    yield x
        yield node
    plans = list(get_nodes(plan['Plan']))
    init_plan = {}
    cte_plan = {}
    for tt in plans:
        if 'Subplan Name' in tt.keys():
            if tt['Parent Relationship'] in ['InitPlan'] and 'InitPlan' in tt['Subplan Name']:
                key = re.findall(r'\$\d+', tt['Subplan Name'])[0]
                init_plan[key] = tt
            elif tt['Parent Relationship'] in ['InitPlan'] and 'CTE' in tt['Subplan Name']:
                key = tt['Subplan Name'].split(" ")[1]
                cte_plan[key] = tt
                # if 'Filter'
    for tt in plans:
        # add(tt)
        if 'Actual Total Time' in tt.keys() and tt['Actual Total Time'] != 0 and 'Filter' in tt.keys():
            keys = list(init_plan.keys())
            for key in keys:
                if key in tt['Filter']+",".join(tt['Output']):
                    if 'InitPlan' not in tt.keys():
                        tt['InitPlan'] = [init_plan[key]]
                        init_plan.pop(key)
                    else:
                        flag = False
                        for t in tt['InitPlan']:
                            if t['Subplan Name'] == init_plan[key]['Subplan Name']:
                                flag = True
                        if flag == False:
                            tt['InitPlan'].append(init_plan[key])
                            init_plan.pop(key)
        if 'Actual Total Time' in tt.keys() and tt['Actual Total Time'] != 0 and tt['Node Type'] == "CTE Scan":
            if tt['CTE Name'] in cte_plan.keys():
                if 'InitPlan' not in tt.keys():
                    tt['InitPlan'] = [cte_plan[tt['CTE Name']]]
                    cte_plan.pop(tt['CTE Name'])
                else:
                    flag = False
                    for t in tt['InitPlan']:
                        if t['Subplan Name'].split(" ")[1] == tt['CTE Name']:
                            flag = True
                    if flag == False:
                        tt['InitPlan'].append(cte_plan[tt['CTE Name']])
                        cte_plan.pop(tt['CTE Name'])
def has_subplan(plan_tree):
    def get_nodes(node):
        if 'Plans' in node.keys():
            for t in node['Plans']:
                for x in get_nodes(t):
                    yield x
        yield node
    nodes = list(get_nodes(plan_tree['Plan']))
    for node in nodes:
        if 'Subplan Name' in node.keys():
            return True
    return False

def actual_rows_modify(plan_tree):
    def get_info(node,father):
        if node['Node Type'] in ['Index Scan','Index Only Scan'] and node['Actual Loops'] > 1:
            node['Actual Rows'] = father['Actual Rows']/node['Actual Loops']
        if 'Plans' in node.keys():
            for t in node['Plans']:
                get_info(t,node)

    if 'Plans' in plan_tree.keys():
        for t in plan_tree['Plans']:
            get_info(t, plan_tree)
    return

def get_plantrees(file_names,subplan):
    data = []
    for file_name in file_names:
        with open(file_name, 'r') as f:
            data += f.readlines()
    plan_trees = []
    for item in data:
        if 'Result' in item:
            continue
        try:
            plan_json = json.loads(item.strip())
        except:
            pass
        if not plan_json['planinfo'] or 'Plan' not in plan_json['planinfo'].keys():
            continue
        if plan_json['planinfo']['Plan']['Actual Rows'] == 0:
            continue
        if not subplan and has_subplan(plan_json['planinfo']):
            continue
        if not plan_json['planinfo']:
            continue
        add_parent_info(plan_json['planinfo'])
        add_initplan_info(plan_json['planinfo'])
        actual_rows_modify(plan_json['planinfo']['Plan'])
        if 'config' in plan_json.keys() and 'setting' in plan_json.keys():
            plan_json['config'].update(plan_json['settings'])
        elif 'setting' in plan_json.keys():
            plan_json['config'] = plan_json['setting']
        plan_trees.append(plan_json)

    return plan_trees

def filter_plan_trees(plans):
    res = []
    for plan_json in plans:
        if plan_json['planinfo']['Plan']['Shared Read Blocks']!= 0 :
            # print(plan_json['planinfo']['Plan']['I/O Read Time']/plan_json['planinfo']['Plan']['Shared Read Blocks'])
            if plan_json['planinfo']['Plan']['I/O Read Time']/plan_json['planinfo']['Plan']['Shared Read Blocks'] > 0.002:
                continue
        res.append(plan_json)
    return res
def get_test_results(y_pred,y_true):
    if len(y_pred) != len(y_true):
        raise ValueError("Dimension Wrong")
    r = []
    for i in range(len(y_pred)):
        r.append(max(y_true[i]/(y_pred[i]+1e-6),y_pred[i]/(y_true[i]+1e-6)))
    return {
        "mean":round(np.mean(r),2),
        "50%r": round(np.percentile(r, 50),2),
        "90%r": round(np.percentile(r, 90),2),
        "95%r": round(np.percentile(r, 95),2),
        "99%r": round(np.percentile(r, 99),2),
        "max": round(max(r),2),
    }

def get_r_results(r):
    return {
        "mean":round(np.mean(r),2),
        "50%r": round(np.percentile(r, 50),2),
        "90%r": round(np.percentile(r, 90),2),
        "95%r": round(np.percentile(r, 95),2),
        "99%r": round(np.percentile(r, 99),2),
        "max": round(max(r),2),
    }

def get_operator_num(plan_trees):
    ops = ["Seq Scan", "Index Scan", "Index Only Scan",'Sort','Hash Join','Merge Join', 'Nested Loop','Aggregate']
    def get_op(plan):
        if 'Plans' in plan.keys():
            for item in plan['Plans']:
                for x in get_op(item):
                    yield x
        yield plan['Node Type']
    res = {op:0 for op in ops}
    for plan in plan_trees:
        for op in get_op(plan):
            if op in ops:
                res[op]+=1
    return res

def print_pred_tree(plan):

    def print_tree(plan,idx):
        if "Startup Predict" not in plan.keys():
            plan["Startup Predict"] = 0
            plan["Total Predict"] = 0
        print(idx*3*"*"+str({"name":plan['Node Type'],
               "Startup Predict":round(plan["Startup Predict"],2),
               "Startup True":round(plan["Actual Startup Time"],2),
               "Total Predict":round(plan["Total Predict"],2),
               "Total True":round(plan["Actual Total Time"],2),
               "Actual Rows":plan['Actual Rows'],
               "Loops":plan['Actual Loops']}))

        if 'Plans' in plan.keys():
            for item in plan['Plans']:
                print_tree(item,idx+1)
                if 'InitPlan' in item.keys():
                    for t in item['InitPlan']:
                        print_tree(t,idx+1)


    print_tree(plan,1)
    return

def deal_config_value(value):
    if value == "on":
        return 1
    if value == "off":
        return 0
    try:
        return float(value)
    except:
        return value

def print_args(parser,opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)