import pickle

from .paramtree import Models
from database_util.database_info import *
from database_util.db_connector import *
from feature.plan import *
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

    def fit(self,files,input='file',clear_buffer=True):
        if input == "file":
            plan_trees = get_plantrees(files,subplan=False)
        else:
            plan_trees = files
        plan_trees = plan_trees[:2000]
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

    def predict_op(self, plan, config, Left_start_time, Left_Total_time, Right_start_time, Right_Total_time, learn):
        operator = plan['Node Type']
        res = self.plan_tool.get_op_info(plan, execute=True)
        res.update(config)
        startup_time, total_time,nodeid = self.mobtrees.predict_data(res, Left_start_time, Left_Total_time, Right_start_time,
                                                              Right_Total_time)
        plan['Startup Predict'] = startup_time
        plan['Total Predict'] = total_time
        plan['nodeid'] = nodeid
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
            result = []
            s = set()
            for id,item in enumerate(plan_json):
                if 'template' not in item.keys():
                    item['template'] = 0
                # if item['template'] in [42]:
                #     print("here")
                    # deal_plan(item['planinfo'])
                # if 'template2' in item['query']:
                #     print("here")
                # print(item['template'])

                if 'config' not in item.keys():
                    item['config'] = {}
                
                start_time, Total_time = self.predict_tree(item['planinfo']['Plan'], item['config'], learn)
                # print_pred_tree(item['planinfo']['Plan'])
                # print(f"******{item['planinfo']['Plan']['Actual Total Time']}")
                # if item['template'] not in s:
                #     print("*"*30)
                #     print(item['template'])
                #     try:
                #         print_pred_tree(item['planinfo']['Plan'])
                #     except:
                #         # start_time, Total_time = self.predict_tree(item['planinfo']['Plan'], item['config'], learn)
                #         print_pred_tree(item['planinfo']['Plan'])
                #     s.add(item['template'])
                # if id == 2:
                #     print_pred_tree(item['planinfo']['Plan'])
                #     self.predict_tree(item['planinfo']['Plan'], item['config'], learn)
                # if max(Total_time/item['planinfo']['Plan']['Actual Total Time'],item['planinfo']['Plan']['Actual Total Time']/Total_time) > 50:
                #     print_pred_tree(item['planinfo']['Plan'])
                #     start_time, Total_time = self.predict_tree(item['planinfo']['Plan'], item['config'], learn)
                #     print("-"*100)

                # result.append({"template":item['template'],
                #               "Qerror":max(Total_time/item['planinfo']['Plan']['Actual Total Time'],item['planinfo']['Plan']['Actual Total Time']/Total_time),
                #               "Running Time":item['planinfo']['Plan']['Actual Total Time']})
                res.append(Total_time)
            # df = pd.DataFrame(result)
            # print(df.groupby(['template']).mean())
            # df = df.groupby(['tempslate']).mean()
            # df.to_excel('./tempfile/tpcds_template_pt.xlsx')
            return res
