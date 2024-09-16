import json
from tqdm import tqdm
import pickle
import logging
import numpy as np
import random

from .paramtree import Models
from database_util.database_info import *
from database_util.db_connector import *
from feature.plan import *
from query_gen.bucket import Bucket
from .active_learning import ActiveLearningTool
from feature.infos import dbms_cparams
from util.util import *

class QueryInference:
    def __init__(self,db_name,scale,coefs,args,load,load_model_name,node_features = None):
        self.leaf_op = ['Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Heap Scan', 'CTE Scan']
        self.join_op = ["Hash Join", "Merge Join", "Nested Loop"]
        self.db_name = db_name
        self.scheme_info = Database_info(db_name)
        self.plan_tool = Plan_class(self.scheme_info)
        self.scale = scale
        self.coef = coefs
        self.bucket_info = Bucket(db_name)
        if load_model_name == "":
            load_path = f"./saved_models/{db_name}.pickle"
        else:
            load_path = f"./saved_models/{load_model_name}.pickle"
        self.mobtrees = Models(db_name,scale= self.scale)
        if load == True and not os.path.exists(load_path):
            raise Exception('Model Not Found')
        if load == True and os.path.exists(load_path):
            file = open(load_path,"rb")
            self.mobtrees.models = pickle.load(file)
            print(f"Model load from {load_path}")
        else:
            self.mobtrees.init_models(coefs, batch_size=args.batch_size, min_size=args.leaf_num, node_features=node_features)

    def load_model(self,name):
        load_path = f"./saved_models/{name}.pickle"
        file = open(load_path, "rb")
        self.mobtrees.models = pickle.load(file)
        print(f"Model load from {load_path}")

    def shift_db(self,db_name):
        self.scheme_info = Database_info(db_name)
        self.plan_tool = Plan_class(self.scheme_info)

    def fit(self,files,input='file',clear_buffer=True,train_samples=2000):
        if input == "file":
            plan_trees = get_plantrees(files,subplan=False)
        else:
            plan_trees = files
        plan_trees = plan_trees[:train_samples]
        for index, item in enumerate(plan_trees):
            if 'config' not in item.keys():
                item['config'] ={}
            res = self.predict(item, learn=True)
        if clear_buffer:
            self.mobtrees.clear_buffer()
        return

    def save_model(self,save_name = ""):
        if save_name == "":
            save_path = f"./saved_models/{self.db_name}.pickle"
        else:
            save_path = f"./saved_models/{save_name}.pickle"
        file = open(save_path, "wb")
        pickle.dump(self.mobtrees.models, file)
        print(f"Model saved in {save_path}")
        return

    def get_histogram(self,plans,operator,cparam):
        def get_cparam_value(node):
            if node['Node Type'] == operator:
                info = self.plan_tool.get_op_info(node,execute=True)
                yield info[cparam]
            if 'Plans' in node.keys():
                for item in node['Plans']:
                    item['parent'] = node['Node Type']
                    for x in get_cparam_value(item):
                        yield x
        def find_bucket_item(buckets, value):
            if type(value) == str:
                for idx in range(len(buckets)):
                    if value in buckets[idx]['range']:
                        return idx
                for idx in range(len(buckets)):
                    if 'others' in buckets[idx]['range']:
                        return idx
            else:
                for idx in range(len(buckets)):
                    if idx == 0 and value >= buckets[idx]['range'][0] and value <= buckets[idx]['range'][1]:
                        return idx
                    elif value > buckets[idx]['range'][0] and value <= buckets[idx]['range'][1]:
                        return idx

            return None
        buckets = self.bucket_info.op_buckets[operator][cparam]
        hist = [{'range':item['range'],'count':1} for item in buckets]
        all_count = len(hist)
        for plan in plans:
            if cparam in dbms_cparams:
                value = deal_config_value(plan['config'][cparam])
                id = find_bucket_item(buckets, value)
                if id == None:
                    print("here")
                hist[id]['count'] += 1
                all_count += 1
            else:
                for value in get_cparam_value(plan['planinfo']['Plan']):
                    id = find_bucket_item(buckets,value)
                    if id == None:
                        print("here")
                    hist[id]['count']+=1
                    all_count += 1
        for item in hist:
            item['prob'] = max(5,item['count'])/all_count
            # item['prob'] = 1/len(hist)
        return hist

    def train_model_actively(self,args,run):
        run_count = 0
        def test():
            mean_qerror = []
            runned_queries = []
            for idx,test_file in enumerate(args.workload):
                test_plans = get_plantrees([test_file],args.subplan)
                if len(test_plans)>500:
                    test_plans = test_plans[:500]
                y_pred = self.predict(test_plans)
                y_true = [t['planinfo']['Plan']['Actual Total Time'] for t in test_plans]
                for id,plan in enumerate(test_plans):
                    plan['predict'] = y_pred[id]
                    if max(y_pred[id]/y_true[id],y_true[id]/y_pred[id]) > args.qerror_threshold:
                        runned_queries.append(plan)
                res = get_test_results(y_pred, y_true)
                mean_qerror.append(res['mean'])
                print(res)
                if run:
                    run.track(res['50%r'], name=test_file, step=run_count, context={"subset": "train"})
            return np.mean(mean_qerror),runned_queries

        def test_buffer(buffer):
            new_buffer = []
            rqs = []
            for plan in buffer:
                y_pred = self.predict([plan])
                y_true = [plan['planinfo']['Plan']['Actual Total Time']]
                plan['predict'] = y_pred[0]
                rq = get_test_results(y_pred, y_true)
                rqs.append(round(rq['mean'],2))
                if rq['mean'] > args.qerror_threshold:
                    new_buffer.append(plan)
            return new_buffer

        last_mean_qerror,runned_queries = test()
        active_learning_tool = ActiveLearningTool(self.db_name,self.scheme_info,args.save_model_name,args.knob_change)
        workload_plans = []
        for test_file in args.workload:
            workload_plans += get_plantrees([test_file],args.subplan)
        while len(workload_plans)<args.buffer_size*50:
            workload_plans += workload_plans
        random.shuffle(workload_plans)
        buffer = []

        sample_num = 0
        for plan in workload_plans:
            y_pred = self.predict([plan])
            y_true = [plan['planinfo']['Plan']['Actual Total Time']]
            rq = get_test_results(y_pred, y_true)
            if run_count > 100 or sample_num>2000: # Terminate Program
                break
            # runned_queries.append(plan)
            if rq['mean']>args.qerror_threshold:
                buffer.append(plan)
            if len(buffer)>=args.buffer_size:  #Triggered Expansion

                for step in tqdm(range(100)):
                    test_buffer(buffer)
                    run_count += 1
                    operator,check_dim,node,queries = self.mobtrees.check_node(active_learning_tool, runned_queries,args.random_ratio)
                    if not operator or check_dim == -1:
                        continue
                    print(f"step:{step},operator:{operator},check_cparam:{check_dim}")
                    hist = self.get_histogram(runned_queries,operator,check_dim)

                    # queries = []
                    feats,num = active_learning_tool.getdata_with_filter(operator, check_dim, queries, hist,node,MIN_SAMPLE_NUM=args.sample_num_per_expansion)
                    
                    if not len(feats):
                        continue
                    print(f"add {len(feats['runtime_cost'])} samples into ParamTree for {operator}")
                    self.mobtrees.models[operator]['runtime_cost'].fit_one(np.array(feats['runtime_cost'],dtype=object),check_dim)
                    if len(feats['startup_cost']):
                        self.mobtrees.models[operator]['startup_cost'].fit_one(
                            np.array(feats['startup_cost'], dtype=object),check_dim)
                    buffer = test_buffer(buffer)
                    mean_qerror,runned_queries = test()
                    sample_num += num
                    print(f"Add {sample_num} Queries")
                    if mean_qerror<last_mean_qerror:
                        print("Model Improved, Qerror: %.2f, Model Saved"%mean_qerror)
                        last_mean_qerror = mean_qerror
                        self.save_model(args.save_model_name)
                    print("run_count")
                    self.save_model(args.save_model_name+"_last")
                    if len(buffer)<args.buffer_size:
                        print("Buffer Not Full")
                        print("Improve %d sqls"%(args.buffer_size-len(buffer)))
                        break


    def predict_op(self, plan, config, Left_start_time, Left_Total_time, Right_start_time, Right_Total_time, learn):
        operator = plan['Node Type']
        res = self.plan_tool.get_op_info(plan, execute=True)
        res.update(config)
        startup_time, total_time,nodeid = self.mobtrees.predict_data(res, Left_start_time, Left_Total_time, Right_start_time,
                                                              Right_Total_time)
        plan['Startup Predict'] = startup_time
        plan['Total Predict'] = total_time
        plan['nodeid'] = nodeid
        # if not learn:
        #     print(f"{operator}:actual:{plan['Actual Total Time']},pred:{total_time},loops:{res['Loops']}")
        # if learn and operator in ['Seq Scan','Index Scan','Index Only Scan','Hash Join','Nested Loop','Merge Join','Aggregate','Sort']:
        #     if plan['Shared Read Blocks']>0 and plan['I/O Read Time'] / plan['Shared Read Blocks']>self.coef[0][3]*400:
        #         pass
        #     else:
        #         self.mobtrees.add_data(res)
        if learn and res['never_executed']==False:
            self.mobtrees.add_data(res)
        return startup_time, total_time

    def predict_tree(self, plan_tree, config, learn):
        operator = plan_tree['Node Type']
        if operator not in self.leaf_op and 'Plans' not in plan_tree.keys():
            print("here")
        child_nodes = []
        if 'Plans' in plan_tree.keys():
            for t in plan_tree['Plans']:
                if 'Subplan Name' not in t.keys():
                    child_nodes.append(t)
        subplan_start_time, subplan_total_time = 0, 0
        if 'Plans' in plan_tree.keys():
            for item in plan_tree['Plans']:
                if 'Subplan Name' in item.keys() and item['Parent Relationship'] == "SubPlan":
                    subplan_left_start_time, subplan_left_total_time = self.predict_tree(item, config, learn)
                    item['Startup Predict'] = subplan_left_start_time
                    item['Total Predict'] = subplan_left_total_time
                    if plan_tree['Actual Loops'] != 0:
                        ratio = item['Actual Loops'] / plan_tree['Actual Loops']
                        subplan_start_time += subplan_left_start_time * ratio
                        subplan_total_time += subplan_left_total_time * ratio
        if operator in self.leaf_op:

            if 'InitPlan' in plan_tree.keys() and len(plan_tree['InitPlan']):
                for item in plan_tree['InitPlan']:
                    subplan_left_start_time, subplan_left_total_time = self.predict_tree(item,config,learn)
                    item['Startup Predict'] = subplan_left_start_time
                    item['Total Predict'] = subplan_left_total_time
                    if plan_tree['Actual Loops'] != 0:
                        ratio = item['Actual Loops'] / plan_tree['Actual Loops']
                        subplan_start_time += subplan_left_start_time*ratio
                        subplan_total_time += subplan_left_total_time*ratio

            Left_start_time, Left_Total_time = self.predict_op(plan_tree, config, subplan_start_time, subplan_total_time, 0, 0, learn)
            plan_tree['Startup Predict'] = Left_start_time
            plan_tree['Total Predict'] = Left_Total_time
            return Left_start_time, Left_Total_time
        elif operator in self.join_op:

            if 'InitPlan' in plan_tree.keys() and len(plan_tree['InitPlan']):
                for item in plan_tree['InitPlan']:
                    subplan_left_start_time, subplan_left_total_time = self.predict_tree(item,config,learn)
                    item['Startup Predict'] = subplan_left_start_time
                    item['Total Predict'] = subplan_left_total_time
                    if plan_tree['Actual Loops'] != 0:
                        ratio = item['Actual Loops'] / plan_tree['Actual Loops']
                        subplan_start_time += subplan_left_start_time * ratio
                        subplan_total_time += subplan_left_total_time * ratio

            Left_start_time, Left_Total_time = self.predict_tree(child_nodes[0], config, learn)
            Right_start_time, Right_Total_time = self.predict_tree(child_nodes[1], config, learn)
            Left_start_time, Left_Total_time = self.predict_op(plan_tree, config, Left_start_time, Left_Total_time, Right_start_time,
                                       Right_Total_time, learn)
            plan_tree['Startup Predict'] = Left_start_time+subplan_start_time
            plan_tree['Total Predict'] = Left_Total_time + subplan_total_time
            return Left_start_time+subplan_start_time,Left_Total_time + subplan_total_time
        elif operator == 'Hash':
            Left_start_time, Left_Total_time = self.predict_tree(plan_tree['Plans'][0], config, learn)
            plan_tree['Startup Predict'] = Left_start_time
            plan_tree['Total Predict'] = Left_Total_time
            return Left_start_time, Left_Total_time
        elif operator in ['Append','Merge Append']:
            Left_start_time, Left_Total_time = 0,0
            for item in plan_tree['Plans']:
                start_time,total_time = self.predict_tree(item, config, learn)
                Left_start_time += start_time
                Left_Total_time += total_time
            plan_tree['Startup Predict'] = Left_start_time
            plan_tree['Total Predict'] = Left_Total_time
            return Left_start_time,Left_Total_time
        elif operator == "Materialize":
            Left_start_time, Left_Total_time = self.predict_tree(plan_tree['Plans'][0], config, learn)
            start_time, Total_time = self.predict_op(plan_tree, config, Left_start_time ,
                                                               Left_Total_time, 0, 0, learn)
            if plan_tree['Actual Loops'] == 0:
                return 0,0
            return start_time+Left_start_time/plan_tree['Actual Loops'],Total_time+Left_Total_time/plan_tree['Actual Loops']
        else:

            if 'InitPlan' in plan_tree.keys() and len(plan_tree['InitPlan']):
                for item in plan_tree['InitPlan']:
                    subplan_left_start_time, subplan_left_total_time = self.predict_tree(item,config,learn)
                    item['Startup Predict'] = subplan_left_start_time
                    item['Total Predict'] = subplan_left_total_time
                    if plan_tree['Actual Loops'] != 0:
                        ratio = item['Actual Loops'] / plan_tree['Actual Loops']
                        subplan_start_time += subplan_left_start_time * ratio
                        subplan_total_time += subplan_left_total_time * ratio

            Left_start_time, Left_Total_time = self.predict_tree(child_nodes[0], config, learn)
            child_nodes[0]['Startup Predict'] = Left_start_time
            child_nodes[0]['Total Predict'] = Left_Total_time
            Left_start_time, Left_Total_time = self.predict_op(plan_tree, config, Left_start_time+subplan_start_time, Left_Total_time+subplan_total_time, 0, 0, learn)
            plan_tree['Startup Predict'] = Left_start_time
            plan_tree['Total Predict'] = Left_Total_time
            return Left_start_time,Left_Total_time

    def predict(self, plan_json, learn=False):
        if type(plan_json) == dict:
            start_time, Total_time = self.predict_tree(plan_json['planinfo']['Plan'], plan_json['config'], learn)
            return Total_time
        elif type(plan_json) == list and len(plan_json)!=0:
            res = []
            for id,item in enumerate(plan_json):
                if 'template' not in item.keys():
                    item['template'] = 0

                if 'config' not in item.keys():
                    item['config'] = {}
                
                start_time, Total_time = self.predict_tree(item['planinfo']['Plan'], item['config'], learn)
                res.append(Total_time)
            return res
