# Split a dataset based on an attribute and an attribute value
# from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, Ridge
from matplotlib import pyplot as plt
from tqdm import tqdm

from util.util import *
from feature.feature import *
from database_util.db_connector import *
from database_util.database_info import *
from .node import Node
np.seterr(divide='ignore', invalid='ignore')
import random

from feature.infos import all_cparams


class MobTree():
    def __init__(self, operator, modelname,node_features, leaf_features,scale,coefs,min_size=30, trim = 0.1,alpha = 0.01):
        self.operator = operator
        self.modelname = modelname
        self.min_size = min_size
        self.trim = trim
        self.depth = 1
        self.node_features = node_features
        self.leaf_features = leaf_features
        self.alpha = alpha
        self.types = {}
        self.coefs = coefs
        self.root = Node([],[],self.node_features,self.leaf_features,self.types,self.min_size,self.trim,self.alpha,operator,modelname,coefs=coefs,check_dim=None,checked_dims=set())
        self.node_performance = []
        self.scale = scale
        self.predict_history = []
        self.checked_dimensions = []

    def clear_predict_history(self):
        self.predict_history = []

    def get_predict_history_performance(self):
        if not len(self.predict_history):
            return 1
        return mean_squared_error(np.array(self.predict_history)[:,0],np.array(self.predict_history)[:,1],squared=True)

    def append_predict_history(self,pred,true):
        self.predict_history.append([pred,true])

    def plot_terminal(self, y_pred, y_true):
        plt.scatter(y_pred, y_true, s=1)
        plt.xlabel('predict_time')
        plt.ylabel('actual_time')
        x0 = [t for t in range(int(np.max(y_pred)) + 2)]
        y0 = [t for t in x0]
        plt.plot(x0, y0, 'r')
        plt.show()

    # Build a decision tree
    def tree_add_one(self, train,check_dim):
        ## filter time<0.1
        res = []
        for t in train:
            if t[-1] > 0:
                res.append(t)
        train = np.array(res)
        if self.root == None:
            self.root = Node([],train,self.node_features,self.leaf_features,self.types,self.min_size,self.trim,self.alpha,self.operator,self.modelname,coefs=self.coefs,check_dim=check_dim,checked_dims=set())
        elif isinstance(self.root,Node):
            if len(self.root.types)==0:
                self.root.add_types(self.types)
            if len(train):
                self.root.add_one(train,check_dim)
        return self.root
    def free_grow(self,node):
        if node.node_class == 'InnerNode':
            for child in node.children:
                self.free_grow(child)
        else:
            node.terminal_add_one([],check_dim=None)

    def get_column_type(self, dataset, weights=None):
        types = {}
        for t in range(len(dataset[0])):
            if isinstance(dataset[0][t],str):
                value = set(dataset[:,t])
                types[t]={"type":"category","value":value}
            else:
                types[t]={"type":"numeric"}
        return types

    def fit_one(self, train,check_dim = None, weight=None):
        if not len(train):
            return
        if len(self.types) == 0:
            self.types = self.get_column_type(train, weight)

        self.root = self.tree_add_one(train,check_dim)

    def predict(self, X_test):
        result = []
        for i in range(len(X_test)):
            result.append(self.__predict(self.root, X_test[i]))
        return result

    def __predict(self, node, row):
        if node.node_class == 'InnerNode':
            id = -1
            if node.types[node.index]['type'] == 'numeric':
                for idx,bucket in enumerate(node.buckets):
                    if round(row[node.index], 4) >bucket[0] and round(row[node.index], 4)<=bucket[1]:
                        id = idx
                        break
                if id == -1:
                    raise Exception("Wrong")
                return self.__predict(node.children[idx],row)
            else:
                for idx,bucket in enumerate(node.buckets):
                    if row[node.index] in bucket[0]:
                        id = idx
                        break
                if id == -1:
                    id = np.random.choice(np.arange(len(node.buckets)))
                    # raise Exception("Wrong")
                return self.__predict(node.children[id],row)
        else:
            return node.get_terminal_predict(row)

    def get_pred_node_id(self, X_test):
        result = []
        for i in range(len(X_test)):
            result.append(self.__get_nodeid(self.root, X_test[i]))
        return result

    def __get_nodeid(self, node, row):
        if node.node_class == 'InnerNode':
            id = -1
            if node.types[node.index]['type'] == 'numeric':
                for idx,bucket in enumerate(node.buckets):
                    if round(row[node.index], 4) >bucket[0] and round(row[node.index], 4)<=bucket[1]:
                        id = idx
                        break
                if id == -1:
                    raise Exception("Wrong")
                return self.__get_nodeid(node.children[idx],row)
            else:
                for idx,bucket in enumerate(node.buckets):
                    if row[node.index] in bucket[0]:
                        id = idx
                        break
                if id == -1:
                    id = np.random.choice(np.arange(len(node.buckets)))
                    # raise Exception("Wrong")
                return self.__get_nodeid(node.children[id],row)
        else:
            return node.nodeid

    def get_leaf_params(self, X_test):
        result = []
        for i in range(len(X_test)):
            result.append(self.__get_leaf_params(self.root, X_test[i]))
        return result

    def __get_leaf_params(self, node, row):
        if node.node_class == 'InnerNode':
            if self.types[node.index]['type'] == 'numeric':
                if round(row[node.index],4) <= node.value:
                    return self.__get_leaf_params(node.left, row)
                else:
                    return self.__get_leaf_params(node.right, row)
            else:
                if row[node.index] == node.value:
                    return self.__get_leaf_params(node.left, row)
                else:
                    return self.__get_leaf_params(node.right, row)
        else:
            return node.model

    def score(self, X_test):
        pred = self.predict(X_test[:, :-1])
        return rsquared(pred, X_test[:, -1])

    def get_all_filters(self):
        return self.get_subtree_filters(self.root)

    def get_subtree_filters(self,node):
        filters = []
        if node.node_class == 'InnerNode':
            filters += [node.index]
            filters += self.get_subtree_filters(node.left)
            filters += self.get_subtree_filters(node.right)
        return filters


    def check(self,active_learning_tool):
        nodes = self.check_leaves(self.root,active_learning_tool)
        return nodes

    def check_leaves(self,node,active_learning_tool):
        nodes = []
        if node.node_class == 'InnerNode':
            for node_t in node.children:
                nodes += self.check_leaves(node_t,active_learning_tool)
        else:
            nodes = [node]
        return nodes


    def plot_tree(self,node,depth=0):
        if depth == 0:
            self.plot_index = 0
        if node.node_class == "TerminalNode":
            node.plot_terminal()
            self.plot_index += 1
        else:
            for child in node.children:
                self.plot_tree(child,self.plot_index)

    # Print a decision tree
    def print_tree(self, node, depth=0,items  = ['Nt','No','Ni','Ns','Nr','Np','Nm']):
        if node.node_class == 'InnerNode':
            for idx,child in enumerate(node.children):
                print('%s[%s in %s]' % ((depth * ' ', (self.node_features[node.index]), str(node.buckets[idx]))))
                self.print_tree(child, depth + 1,items = items)
        else:
            params = node.model
            text = ""
            for index,item in enumerate(params[0]):
                text += str(item*1000)
                text += " * "
                text += items[index]
                text += "+"
            # print('%s[%.3f * Nt + %.3f * No + %.3f * Ni + %.3f * Ns + %.3f * Nr + %.3f * Np + %.3f * Nm + %.3f]' % ((depth * ' ', params[0][0]*1000,params[0][1]*1000,params[0][2]*1000,params[0][3]*1000,params[0][4]*1000,params[0][5]*1000,params[0][6]*1000,params[1]*1000)))
            print('%s%s'%(depth * ' ',text))
            if len(params) >=7:
                print("%s[Depth:%d, Rms: %.3f, Mape: %.3f,Sample number: %d]"%(depth * ' ',depth,node.get_group_mse(node.dataset),node.get_node_predict_mape(node.dataset), len(node.dataset)))

    def get_subtree_mape_score(self,node):
        if node.node_class == 'InnerNode':
            count_left,score_sum_left = self.get_subtree_mape_score(node.left)
            count_right,score_sum_right = self.get_subtree_mape_score(node.right)
            return count_left+count_right,(score_sum_left*count_left+score_sum_right*count_right)/(count_left+count_right)
        else:
            return len(node.dataset),node.mape_score
import json

class Models():
    def __init__(self,db_name,scale,mode = "Dafault_Setting",config = None):
        # print("Create Normal Mob Tree.")
        self.feature_tool = FeatureExtract()
        self.scheme_info = Database_info(db_name)
        self.models = {}
        self.temp_pool = {}
        self.train_num = {}
        self.scale = scale
        self.check_num = 0
        for operator in self.feature_tool.features.keys():
            self.models[operator] = {}
            self.temp_pool[operator] = {}
            self.train_num[operator] = 0

    def clear_mobtrees_predict_history(self):
        for operator in self.feature_tool.features.keys():
            for modelname in self.feature_tool.features[operator].keys():
                self.models[operator][modelname].clear_predict_history()

    def get_models_mape_score(self):
        res = {}
        for operator in self.feature_tool.features.keys():
            if len(self.feature_tool.features[operator].keys())==2:
                count_s,mape_score_s = self.models[operator]['startup_cost'].get_subtree_mape_score(self.models[operator]['startup_cost'].root)
                count_r, mape_score_r = self.models[operator]['runtime_cost'].get_subtree_mape_score(
                    self.models[operator]['runtime_cost'].root)
                res[operator] = (mape_score_s*(count_s+1)+mape_score_r*(count_r+1))/(count_s+count_r+2)
            else:
                count, mape_score = self.models[operator]['runtime_cost'].get_subtree_mape_score(
                    self.models[operator]['runtime_cost'].root)
                res[operator] = mape_score
        return res

    def predict_data(self,data,Left_start_time,Left_Total_time, Right_start_time,Right_Total_time):
        operator = data['name']
        nodeinfo = {"runtime_nodeid":[],"startup_nodeid":[],
                    'runtime_true':0,'runtime_pred':0,
                    'startup_true':0,'startup_pred':0,}
        if data['never_executed'] == True:
            return 0,0,nodeinfo
        data = [data]
        if operator in ['Seq Scan','Index Scan','Index Only Scan']:
            startup_time = 0
            modelname = "runtime_cost"
            dataset = self.get_model_data(data,operator,modelname)
            total_time = self.models[operator][modelname].predict(dataset[:,:-1])[0]
            nodeinfo['runtime_nodeid'] = self.models[operator][modelname].get_pred_node_id(dataset[:,:-1])[0]
            nodeinfo['runtime_true'] = dataset[0, -1]
            nodeinfo['runtime_pred'] = total_time
            self.models[operator]["runtime_cost"].append_predict_history(total_time,dataset[:,-1][0])
            if data[0]['Loops'] != 0:
                total_time = total_time / data[0]['Loops']
                startup_time = startup_time / data[0]['Loops']

            return Left_start_time+startup_time,Left_Total_time+total_time,nodeinfo

        elif operator in ['Sort','Hash Join']:
            modelname = "startup_cost"
            dataset = self.get_model_data(data,operator,modelname)
            startup_time = self.models[operator][modelname].predict(dataset[:,:-1])[0]
            nodeinfo['startup_true'] = dataset[0,-1]
            nodeinfo['startup_nodeid'] = self.models[operator][modelname].get_pred_node_id(dataset[:, :-1])[0]
            nodeinfo['startup_pred'] = startup_time
            self.models[operator]["startup_cost"].append_predict_history(startup_time, dataset[:, -1][0])
            runtime_true = dataset[0,-1]
            modelname = "runtime_cost"
            dataset = self.get_model_data(data,operator,modelname)
            total_time = self.models[operator][modelname].predict(dataset[:,:-1])[0]
            nodeinfo['runtime_nodeid'] = self.models[operator][modelname].get_pred_node_id(dataset[:, :-1])[0]
            nodeinfo['runtime_true'] = dataset[0, -1]
            nodeinfo['runtime_pred'] = total_time
            self.models[operator]["runtime_cost"].append_predict_history(total_time, dataset[:, -1][0])

            if data[0]['Loops'] != 0:
                total_time = total_time / data[0]['Loops']
                startup_time = startup_time / data[0]['Loops']
            if operator == 'Hash Join':
                # print(startup_time)
                if data[0]['RightRows'] == 0 and data[0]['LeftOp'] == 'Seq Scan':
                    startup_time += Right_Total_time + Left_start_time
                    total_time += startup_time
                else:
                    startup_time += Right_Total_time #+ Left_start_time
                    total_time += startup_time + Left_Total_time
            else:
                startup_time += Left_Total_time
                total_time += startup_time
            return startup_time,total_time,nodeinfo

        elif operator == "Nested Loop":
            startup_time = Left_start_time + Right_start_time
            modelname = "runtime_cost"
            dataset = self.get_model_data(data,operator,modelname)
            total_time = self.models[operator][modelname].predict(dataset[:,:-1])[0]
            nodeinfo['runtime_nodeid'] = self.models[operator][modelname].get_pred_node_id(dataset[:, :-1])[0]
            nodeinfo['runtime_true'] = dataset[0, -1]
            nodeinfo['runtime_pred'] = total_time
            self.models[operator]["runtime_cost"].append_predict_history(total_time, dataset[:, -1][0])
            if data[0]['Loops'] != 0:
                total_time = total_time / data[0]['Loops']
            if data[0]['RightOp'] in ['Sort']:
                total_time += Left_Total_time + Right_Total_time
            else:
                total_time += Left_Total_time + Right_start_time + Right_start_time*(data[0]['LeftRows']-1)+data[0]['LeftRows']*(Right_Total_time-Right_start_time)
            # total_time += startup_time
            return startup_time,total_time,nodeinfo
        
        elif operator == "Merge Join":
            startup_time = Left_start_time + Right_start_time 
            modelname = "runtime_cost"
            dataset = self.get_model_data(data,operator,modelname)
            total_time = self.models[operator][modelname].predict(dataset[:,:-1])[0]
            nodeinfo['runtime_nodeid'] = self.models[operator][modelname].get_pred_node_id(dataset[:, :-1])[0]
            nodeinfo['runtime_true'] = dataset[0, -1]
            nodeinfo['runtime_pred'] = total_time
            self.models[operator]["runtime_cost"].append_predict_history(total_time, dataset[:, -1][0])
            if data[0]['Loops'] != 0:
                total_time = total_time / data[0]['Loops']
            total_time += Left_Total_time + Right_Total_time 
            return startup_time,total_time,nodeinfo
        
        elif operator == "Aggregate":
            modelname = "startup_cost"
            dataset = self.get_model_data(data,operator,modelname)
            startup_time = self.models[operator][modelname].predict(dataset[:,:-1])[0]
            nodeinfo['startup_true'] = dataset[0,-1]
            nodeinfo['startup_nodeid'] = self.models[operator][modelname].get_pred_node_id(dataset[:, :-1])[0]
            nodeinfo['startup_pred'] = startup_time
            self.models[operator]["startup_cost"].append_predict_history(startup_time, dataset[:, -1][0])
            modelname = "runtime_cost"
            dataset = self.get_model_data(data,operator,modelname)
            total_time = self.models[operator][modelname].predict(dataset[:,:-1])[0]
            nodeinfo['runtime_nodeid'] = self.models[operator][modelname].get_pred_node_id(dataset[:, :-1])[0]
            nodeinfo['runtime_true'] = dataset[0, -1]
            nodeinfo['runtime_pred'] = total_time
            self.models[operator]["runtime_cost"].append_predict_history(total_time, dataset[:, -1][0])

            if data[0]['Loops'] != 0:
                total_time = total_time/data[0]['Loops']
                startup_time = startup_time/data[0]['Loops']
            if data[0]['Strategy'] == 'Plain':
                startup_time += Left_Total_time
                total_time += startup_time
            elif data[0]['Strategy'] == 'Sorted':
                startup_time += Left_start_time
                total_time += Left_Total_time
            elif data[0]['Strategy'] in ['Hashed','Mixed']:
                startup_time += Left_Total_time
                total_time += startup_time
            return startup_time,total_time,nodeinfo
        elif operator == "Materialize":
            return 0,(data[0]['Total Cost']-data[0]['Left Total Cost'])/self.scale,nodeinfo
        else:
            if 'Left Total Cost' not in data[0].keys():
                if operator == "CTE Scan":
                    return data[0]['Startup Cost']/self.scale*10+Left_start_time,(data[0]['Total Cost'])/self.scale*10+Left_Total_time,nodeinfo
                else:
                    return data[0]['Startup Cost']/self.scale+Left_start_time,(data[0]['Total Cost'])/self.scale+Left_Total_time,nodeinfo

            if data[0]['Startup Cost']>=data[0]['Left Total Cost']:
                startup_time = Left_Total_time + (data[0]['Startup Cost']-data[0]['Left Total Cost'])/self.scale
                total_time = startup_time + (data[0]['Total Cost']-data[0]['Startup Cost'])/self.scale
            else:
                startup_time = Left_start_time + (data[0]['Startup Cost']-data[0]['Left Startup Cost'])/self.scale
                total_time = Left_Total_time + (data[0]['Total Cost']-data[0]['Left Total Cost'])/self.scale
            return startup_time,total_time,nodeinfo

    def add_datas(self,datas):
        for data in datas:
            self.add_data(data)

    def add_data(self,data):

        operator = data['name']
        if operator not in self.feature_tool.features.keys():
            return
        self.train_num[operator] += 1
        for modelname in self.feature_tool.features[operator].keys():
            if modelname in self.temp_pool[operator].keys():
                self.temp_pool[operator][modelname].append(data)
            else:
                self.temp_pool[operator][modelname] = [data]
            if len(self.temp_pool[operator][modelname])>=self.batch_size:
                dataset = self.get_model_data(self.temp_pool[operator][modelname],operator,modelname)
                self.models[operator][modelname].fit_one(dataset)
                self.temp_pool[operator][modelname] = []
        
    def clear_buffer(self):
        print("begin clear buffer")
        for operator in tqdm(self.feature_tool.features.keys()):
            for modelname in self.temp_pool[operator].keys():
                dataset = self.get_model_data(self.temp_pool[operator][modelname],operator,modelname)
                self.models[operator][modelname].fit_one(dataset)
                self.temp_pool[operator][modelname] = []

    def init_models(self,coefs,min_size=10,batch_size=5000,node_features = None):
        self.batch_size = batch_size
        for operator in self.feature_tool.features.keys():
            for modelname in self.feature_tool.features[operator].keys():
                if node_features == None:
                    node_features = self.feature_tool.features[operator][modelname]['node_features']
                self.models[operator][modelname]= MobTree(operator = operator, modelname = modelname,min_size=min_size, node_features=node_features,
                                    leaf_features=self.feature_tool.leaf_features,scale=self.scale,alpha = 0.05, trim = 0.1,coefs=coefs)



    def check(self,active_learning_tool,buffer):

        # check_operator = ""
        # max_q = 1
        # q_info = {}
        # op_cand = []
        # for operator in self.models.keys():
        #     for model_name in self.models[operator].keys():
        #         if op_num[operator] == 0:
        #             continue
        #         q = np.log(self.models[operator][model_name].get_predict_history_performance()*op_num[operator])
        #         if 'Scan' in operator:
        #             q*=2
        #         if operator not in q_info.keys():
        #             q_info[operator] = q
        #         elif q>q_info[operator]:
        #             q_info[operator] = q
        #
        # random.seed(time.time())
        # op_sort = np.argsort([q_info[op] for op in q_info.keys()])
        # weights = np.zeros(len(op_sort))
        # for i in range(len(op_sort)):
        #     weights[op_sort[i]] = i
        responsibility,sqls = self.get_responsibility(buffer)
        op_responsibility = {op:[] for op in responsibility.keys()}
        # check_operator = random.choices(list(q_info.keys()),k = 1,weights = weights)[0]
        for op in responsibility.keys():
            for t in responsibility[op].keys():
                op_responsibility[op] += responsibility[op][t]
        random.seed(time.time())
        op_sort = np.argsort([op_responsibility[t] for t in op_responsibility.keys()])
        weights = np.zeros(len(op_sort))
        for i in range(len(op_sort)):
            weights[op_sort[i]] = i+1
        check_operator = random.choices(list(responsibility.keys()),k=1,weights=weights)[0]
        # check_operator = random.choices(list(op_responsibility.keys()), k=1, weights=[np.mean(op_responsibility[t]) for t in op_responsibility.keys()])[0]
        nodes = []
        for model_name in self.models[check_operator].keys():
            nodes += self.models[check_operator][model_name].check(active_learning_tool)
        vote_res = {}
        for node in nodes:
            check_dim = active_learning_tool.get_sensitivity(op=check_operator,filters=node.filters)
            if check_dim not in vote_res.keys():
                vote_res[check_dim] = 1
            else:
                vote_res[check_dim] += 1
        check_dim = sorted(vote_res.items(), key=lambda i: i[1], reverse=True)[0][0]
        return check_operator,check_dim

    def get_responsibility(self,buffer):
        ops = ["Seq Scan", "Index Scan", "Index Only Scan",'Hash Join', 'Merge Join', 'Nested Loop','Sort', 'Aggregate']

        def flatten(plan_trees):

            def get_op(plan):
                if 'Plans' in plan.keys():
                    for item in plan['Plans']:
                        for x in get_op(item):
                            yield x
                yield plan

            res = {op: [] for op in ops}
            for plan in plan_trees:
                all_time = plan['planinfo']['Plan']['Actual Total Time']
                plan_qerror = max(all_time/plan['predict'],plan['predict']/all_time)
                query = {'sql':plan['query'],'template':plan['template']}
                for item in get_op(plan['planinfo']['Plan']):
                    item['All Time'] = all_time
                    item['query'] = query
                    item['qerror'] = plan_qerror
                    if item['Node Type'] in ops:
                        res[item['Node Type']].append(item)

            return res

        op_items = flatten(buffer)
        responsibility = {op: {} for op in ops}
        sqls = {op: {} for op in ops}
        check_operator = ""
        for op in op_items.keys():
            for item in op_items[op]:
                if 'nodeid' not in item.keys():
                    continue

                if item['All Time']<0.1:
                    rs = 0
                else:
                    rs = abs(item['nodeid']['runtime_true']-item['nodeid']['runtime_pred']) / item['All Time'] * item['qerror'] *2#* item['qerror']
                key = tuple(item['nodeid']['runtime_nodeid'])
                if key in responsibility[op].keys():
                    responsibility[op][key].append(rs)
                    sqls[op][key].append({'sql':item['query']['sql'],'template':item['query']['template'],'qerror':item['qerror']})
                else:
                    responsibility[op][key] = [rs]
                    sqls[op][key] = [{'sql':item['query']['sql'],'template':item['query']['template'],'qerror':item['qerror']}]
        return responsibility,sqls

    def check_node(self,active_learning_tool,buffer,random_ratio):

        responsibility,sqls = self.get_responsibility(buffer)
        ops = list(responsibility.keys())
        op_res = {t:[] for t in ops}
        for op in responsibility.keys(): 
            for key in responsibility[op].keys():
                op_res[op] += responsibility[op][key]
        random.seed(time.time())

        while True:
            if np.random.random()<1-random_ratio:
                check_operator = random.choices(ops,k=1,weights=[max(0.1,np.median(op_res[op])) for op in ops])[0]
            else:
                check_operator = random.choices(ops, k=1)[0]

            new_responsibility = []
            for key in responsibility.keys():
                if key.split(",")[0] == check_operator:
                    for nodeid in responsibility[key].keys():
                        new_responsibility.append({"op":key,'nodeid':nodeid,'rs':np.mean(responsibility[key][nodeid])})
            if len(new_responsibility) == 0:
                continue
            else:
                break
        self.check_num += 1
        id = random.choices(np.arange(len(new_responsibility)), k=1, weights=[item['rs'] for item in new_responsibility])[0]
        selected_item = new_responsibility[id]
        node = self.models[check_operator]['runtime_cost'].root
        max_responsibility_nodeid = selected_item['nodeid']
        for i in max_responsibility_nodeid:
            node = node.children[i]
        selected_node = node
        # sort_id = np.argsort([t['rs'] for t in new_responsibility],)[::-1]
        # for id in sort_id:
        #     item = new_responsibility[id]
        #     node = self.models[check_operator]['runtime_cost'].root
        #     max_responsibility_nodeid = item['nodeid']
        #     for i in max_responsibility_nodeid:
        #         node = node.children[i]
        #     if len(node.checked_dims)<len(all_cparams):
        #         selected_node = node
        #         selected_item = item
        #         break
        if selected_node:
            sqls = sqls[selected_item['op']][selected_item['nodeid']]
            if len(sqls)>20:
                # sql_res.sort(key=lambda x: x['qerror'])
                # sql_res = sql_res[-20:]
                ids = random.choices(np.arange(len(sqls)), k=20,
                                    weights=[item['qerror'] for item in sqls])
                sql_res = [sqls[t] for t in ids]
            else:
                sql_res = sqls
            check_dim = active_learning_tool.get_sensitivity(op=check_operator, filters=selected_node.filters,
                                                             checked_dims=list(selected_node.checked_dims))

            return check_operator,check_dim,selected_node,sql_res
        else:
            return None,None,None,None

    def get_model_raw_data(self,data,operator,model):
        dataset = []
        for item in data:
            if not isinstance(item,dict):
                plan_json = json.loads(item.strip())
            else:
                plan_json = item
            if plan_json['name'] == operator:
                feat = self.feature_tool.get_model_raw_feature(plan_json,operator,model)
                feat.update(plan_json)
                dataset.append(feat)
        return pd.DataFrame(dataset)

    def get_model_data(self,data,operator,model):
        features = self.feature_tool.features[operator][model]['node_features']+self.feature_tool.leaf_features+self.feature_tool.target_feature
        dataset = []
        for item in data:
            if not isinstance(item,dict):
                plan_json = json.loads(item.strip())
            else:
                plan_json = item
            if plan_json['name'] == operator:
                feat = self.feature_tool.get_model_raw_feature(plan_json,operator,model)
                dataset.append([feat[feature] for feature in features])
        if len(dataset)==0:
            return []

        return np.array(dataset,dtype=object)

    def get_predict_result(self,dataset):
        result = self.tree.predict(dataset[:, :-1])
        test_mape = mape(dataset[:, -1],result)
        test_rms = rms(dataset[:, -1],result)
        return test_mape,test_rms
