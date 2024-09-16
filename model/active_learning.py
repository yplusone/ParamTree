import numpy as np
import random
import json
from tqdm import tqdm
import traceback
import copy

from database_util.db_connector import *
from feature.feature import FeatureExtract
from feature.plan import *
from recommendation.knob_rs import Knob_rs
from query_gen.workload_modify import SqlModify
from database_util.database_info import *
from query_gen.bucket import Bucket
from feature.infos import dbms_cparams,db_info
from util.util import get_plantrees,deal_plan
from database_util.knob import *


pool_file_path = f"./data/temporary/query_gen_pool/"
# MIN_SAMPLE_NUM = 100

class ActiveLearningTool():
    def __init__(self,db_name, scheme_info,test_name,knob_change):
        self.feature_tool = FeatureExtract()
        db_info['pg']['db_name'] = db_name
        self.db_connector = Postgres_Connector(server=db_info['server'], pg = db_info['pg'], ssh = db_info['ssh'])
        knobs = load_knobs_from_json_file()
        self.db_connector.initial_tunning_knobs(knobs)
        self.knob_change = knob_change
        self.scheme_info = scheme_info
        self.plan_tool = Plan_class(self.scheme_info)
        self.kb = Knob_rs()
        self.sqlmodify_tool = SqlModify(self.db_connector,scheme_info)
        self.checked_dims = {t:[] for t in self.feature_tool.features.keys()}
        self.bucket_info = Bucket(db_name)
        dir_path = "./data/benchmark/learnedsqlgen/"
        train_files = [
            f"{dir_path}/{self.db_connector.db_name}_pool_scan.txt",
            f"{dir_path}/{self.db_connector.db_name}_pool_join.txt",]
        self.pool = self.init(train_files)
        # self.file_path = f"{pool_file_path}/db_{self.db_connector.db_name}_{self.db_connector.server.replace('.', '_')}"
        self.file_path = f"{pool_file_path}/db_{test_name}_47_96_181_98"
        self.exec_pool = {}
        for op in self.feature_tool.features.keys():
            for model_name in self.feature_tool.features[op].keys():
                if op not in self.exec_pool.keys():
                    self.exec_pool[op] = {}
                self.exec_pool[op][model_name] = []
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
            f = open(f"{self.file_path}/run_sqls.txt", 'w')
            f.close()
        self.exec_pool = self.get_pool([f"{self.file_path}/run_sqls.txt"],pop = False)


    def get_sensitivity(self,op,filters,checked_dims=None):
        if checked_dims:
            return self.kb.get_sensitivity(op, filters, checked_dims,self.knob_change)
        else:
            return self.kb.get_sensitivity(op,filters,self.checked_dims[op],self.knob_change)

    def init(self,train_files):
        return self.get_pool(train_files)

    def get_pool(self,file_names,pop=True):
        plan_trees = get_plantrees(file_names,True)
        ops = ["Seq Scan","Index Scan","Index Only Scan","Sort","Hash Join","Nested Loop","Merge Join","Aggregate"]
        if pop:
            opdatas = {op:[] for op in ops}
        else:
            opdatas = {op:{'runtime_cost':[]} for op in ops}
            opdatas['Sort']['startup_cost'] = []
            opdatas['Aggregate']['startup_cost'] = []
            opdatas['Hash Join']['startup_cost'] = []
        def get_op_info(plan):
            if 'Plans' in plan.keys():
                for item in plan['Plans']:
                    get_op_info(item)
            if plan['Node Type'] in opdatas.keys():
                res = self.plan_tool.get_op_info(plan, execute=True)
                feat = self.feature_tool.get_model_raw_feature(res, plan['Node Type'], 'runtime_cost')
                if pop:
                    for item in ['Nt', 'No', 'Ni', 'Ns', 'Nr', 'Np', 'Nm', 'y']:
                        feat.pop(item)
                    opdatas[plan['Node Type']].append({'feat':feat,'query':plan_json['query'],'delete':0})
                else:
                    opdatas[plan['Node Type']]['runtime_cost'].append({'feat': feat, 'query': plan_json['query'], 'delete': 0})
                    if 'startup_cost' in opdatas[plan['Node Type']].keys():
                        feat = self.feature_tool.get_model_raw_feature(res, plan['Node Type'], 'startup_cost')
                        opdatas[plan['Node Type']]['startup_cost'].append({'feat': feat, 'query': plan_json['query'], 'delete': 0})
            else:
                pass

        for plan_json in plan_trees:
            if pop and plan_json['planinfo']['Execution Time']>100000:
                continue
            get_op_info(plan_json['planinfo']['Plan'])

        return opdatas

    def sample_select(self,data,check_dim,num=20):
        feats = []
        ids = []
        costs = []
        queries = []
        for item in data:
            feats.append(item['feat'])
            queries.append(item['query'])
        if not len(feats):
            return [],[]
        df = pd.DataFrame(feats)
        df['query'] = np.array(queries)
        choosed_sample = []
        sqls = []
        df = df.sample(frac=1).reset_index(drop=True)
        cat_num = int(np.ceil(num/len(set(df[check_dim]))))
        for cat in set(df[check_dim]):
            count = 0
            for i in range(len(df[df[check_dim]==cat])):
                sqls.append(df[df[check_dim]==cat].iloc[i]['query'])
                count += 1
                if count >= cat_num:
                    break
        if len(sqls)<num:
            choosed_sample_num = len(sqls)
            sqls += list(df.iloc[-(num-choosed_sample_num):]['query'])
        if len(sqls)>num:
            sqls = sqls[:num]
        return sqls,choosed_sample

    def get_feat(self, sqls, operator, model, execute):
        feats = []
        for sql in tqdm(sqls):
            # sql = sql[:-1]+" limit 100;"
            try:
                if execute:
                    ans = \
                    self.db_connector.execute("EXPLAIN (ANALYZE, COSTS, BUFFERS, VERBOSE, FORMAT JSON) " + sql)[0][0][0]
                else:
                    ans = self.db_connector.execute("EXPLAIN (COSTS, FORMAT JSON) " + sql)[0][0][0]
            except:
                traceback.print_exc()
                continue
            plan_tree = Plan_class(self.scheme_info, ans, db=self.db_connector)
            results = plan_tree.get_plan_info(sql, execute)
            for result in results:
                if result['name']==operator:
                    feat = self.feature_tool.get_model_raw_feature(result, operator, model)
                    feats.append(feat)
        if not len(feats):
            return []
        # df = pd.DataFrame(feats)[self.feature_tool.features[operator][model][
        #                              'node_features'] + self.feature_tool.leaf_features + self.feature_tool.target_feature]
        return feats

    def add_exec_pool(self,plan_trees,op):
        result = {"runtime_cost":[],"startup_cost":[]}
        def get_op_info(plan,config):
            if 'Plans' in plan.keys():
                for item in plan['Plans']:
                    get_op_info(item,config)
            if plan['Node Type'] in self.exec_pool.keys():
                res = self.plan_tool.get_op_info(plan, execute=True)
                res.update(config)
                feat = self.feature_tool.get_model_raw_feature(res, plan['Node Type'], 'runtime_cost')
                # for item in ['Nt', 'No', 'Ni', 'Ns', 'Nr', 'Np', 'Nm', 'y']:
                #     feat.pop(item)
                if res['Rows'] > 0:
                    self.exec_pool[plan['Node Type']]['runtime_cost'].append({'feat':feat,'delete':0})
                    if plan['Node Type'] == op:
                        result['runtime_cost'].append(feat)
                    if 'startup_cost' in self.exec_pool[plan['Node Type']].keys():
                        feat = self.feature_tool.get_model_raw_feature(res, plan['Node Type'], 'startup_cost')
                        self.exec_pool[plan['Node Type']]['startup_cost'].append({'feat': feat, 'delete': 0})
                        if plan['Node Type'] == op:
                            result['startup_cost'].append(feat)
            else:
                pass

        for plan_json in plan_trees:
            get_op_info(plan_json['planinfo']['Plan'],plan_json['config'])

        # for op in self.feature_tool.features.keys():
        #     for model_name in self.feature_tool.features[op].keys():
        #         with open(f"{self.file_path}/{op}_{model_name}.txt", 'w') as f:
        #             for item in self.exec_pool[op][model_name]:
        #                 f.writelines(json.dumps(item) + "\n")
        return result

    def raw_feature_deal(self,feat,operator,model):
        features = self.feature_tool.features[operator][model]['node_features'] + self.feature_tool.leaf_features + self.feature_tool.target_feature
        res = []
        for feature in features:
            res.append(feat[feature])
        return res

    def select_samples(self,y,py,num):
        y = list(y)
        py = list(py)
        def compute_d(y, py):
            d = np.zeros((len(y), len(py)))
            for i in range(len(y)):
                for j in range(len(py)):
                    d[i][j] = abs(y[i] - py[j])
            return d
        py_id = list(np.arange(len(py)))
        res_id = []
        if not len(y):
            id = np.random.choice(py_id,1)[0]
            res_id.append(py_id[id])
            y.append(py[id])
            del py[id]
            del py_id[id]

        while len(res_id) < num:
            d = compute_d(y, py)
            t = np.min(d, axis=0)
            idx = np.argmax(t)
            res_id.append(py_id[idx])
            y.append(py[idx])
            del py[idx]
            del py_id[idx]
        return res_id

    def bucket_choose(self,buckets,hist,node=None):
        node = None
        sqls = []
        if node == None or not len(node.dataset):
            node_y = []
        else:
            node_y = node.dataset[:,-1]
        feats = {'runtime_cost':[],'startup_cost':[]}
        bucket_choice = []
        for i in range(len(buckets)):
            if len(buckets[i]['queries']):
                bucket_choice.append(i)
            elif 'feats' in buckets[i].keys() and len(buckets[i]['feats']['runtime_cost']):
                bucket_choice.append(i)
        if not len(bucket_choice):
            return sqls,feats
        num_per_bucket = int(np.ceil(self.MIN_SAMPLE_NUM/len(bucket_choice)))
        sum_prob = sum([hist[id]['prob'] for id in bucket_choice])
        probs  = {id:hist[id]['prob']/sum_prob for id in bucket_choice}
        for id in bucket_choice:
            bucket = buckets[id]
            num_bucket = int(np.ceil(self.MIN_SAMPLE_NUM*probs[id]))
            num_bucket =  num_per_bucket
            count = 0
            if 'feats' in bucket.keys() and len(bucket['feats']['runtime_cost']):
                if len(bucket['feats']['runtime_cost']):
                    py = [t['feat']['y'] for t in bucket['feats']['runtime_cost']]
                    if node == None:
                        ids = []
                        for id,item in enumerate(bucket['feats']['runtime_cost']):
                            if item['delete'] == 0:
                                ids.append(id)
                        # ids = np.arange(0,len(bucket['feats']['runtime_cost']),1)
                        choice = np.random.choice(ids,min(len(ids),num_bucket))
                    else:
                        choice = self.select_samples(node_y,py,min(len(py),num_bucket))
                    # choice = ids
                    for i in choice:
                        bucket['feats']['runtime_cost'][i]['delete'] = 1
                        if len(bucket['feats']['startup_cost']):
                            bucket['feats']['startup_cost'][i]['delete'] = 1
                    feats['runtime_cost'] += [bucket['feats']['runtime_cost'][i]['feat'] for i in choice]
                    if len(bucket['feats']['startup_cost']):
                        feats['startup_cost'] += [bucket['feats']['startup_cost'][i]['feat'] for i in choice]
                    count += len(choice)
            if count<num_bucket:
                if len(bucket['queries'])>num_bucket-count:
                    workload_queries = []
                    templates = {}
                    for item in bucket['queries']:
                        if '--template' in item['query']:
                            pattern = r'--template(.*)\n'
                            match = re.search(pattern, item['query'])
                            template =  match.group(1)
                            if template not in templates.keys():
                                templates[template] = [item]
                            else:
                                templates[template].append(item)
                            workload_queries.append(item)
                    if len(workload_queries)>num_bucket-count:
                        for key in templates:
                            templates[key] = sorted(templates[key], key=lambda x: x['cost'])
                        count = 0
                        while True:
                            for key in templates:
                                if len(templates[key])>=count+1:
                                    sqls.append(templates[key][count])
                                if len(sqls) > num_bucket-count:
                                    break
                            count += 1
                            if len(sqls) > num_bucket-count:
                                break
                        # sqls += list(np.random.choice(workload_queries,int(num_bucket-count),replace=False))
                    else:
                        sqls += workload_queries
                        choosed_sqls = [t['query'] for t in sqls]
                        cand = bucket['queries']
                        random.shuffle(cand)
                        for item in cand:
                            if item['query'] not in choosed_sqls:
                                sqls.append(item)
                            if len(sqls) >= num_bucket - count:
                                break
                        # sqls += list(np.random.choice(list(set(bucket['queries'])-set(workload_queries)), int(num_bucket - count - len(workload_queries)), replace=False))
                else:
                    sqls += bucket['queries']
        return sqls,feats

    def get_pool_data(self,operator,check_dim,buckets,filters):
        for model_name in self.exec_pool[operator].keys():
            for item in self.exec_pool[operator][model_name]:
                if self.check_data(filters,item['feat']):
                    value = item['feat'][check_dim]
                    if type(value) == str:
                        for idx, bucket in enumerate(buckets):
                            if 'feats' not in bucket.keys():
                                bucket['feats'] = {}
                                bucket['feats']['runtime_cost'] = []
                                bucket['feats']['startup_cost'] = []
                            if value in bucket['range']:
                                bucket['feats'][model_name].append(item)
                    else:
                        for idx,bucket in enumerate(buckets):
                            if 'feats' not in bucket.keys():
                                bucket['feats'] = {}
                                bucket['feats']['runtime_cost'] = []
                                bucket['feats']['startup_cost'] = []
                            if idx == 0 and value>=bucket['range'][0] and value<=bucket['range'][1]:
                                bucket['feats'][model_name].append(item)
                            elif value>bucket['range'][0] and value<=bucket['range'][1]:
                                if 'feats' not in bucket.keys():
                                    bucket['feats'] = {}
                                    bucket['feats']['runtime_cost'] = []
                                    bucket['feats']['startup_cost'] = []
                                bucket['feats'][model_name].append(item)

        return buckets

    def set_envs(self,env_cmd):
        for cmd in env_cmd:
            self.db_connector.execute(query=cmd,set_env=True)

    def get_filter_result(self,filter, value):
        if filter['type'] == "numeric":
            if  value <= filter['value'][0] or value > filter['value'][1]:
                return False
        else:
            if value not in filter['value']:
                return False
        return True

    def get_filters_result(self, filters, value):
        for filter in filters:
            if not self.get_filter_result(filter, value):
                return False
        return True

    def check_data(self, filters, data):
        for filter in filters:
            if filter['name']  in ['Rows','Loops','LeftRows','LeftLoops','RightRows','RightLoops','Selectivity']:
                continue
            if not self.get_filters_result([filter], data[filter['name']]):
                return False
        return True

    def check_query(self,query,operator,all_filters):
        env_set = query['env']
        if 'before' in env_set.keys():
            for command in env_set['before']:
                self.db_connector.execute(query=command, set_env=True)
        res_dict = self.db_connector.explain(query['query'], execute=False, timeout=6000000)
        deal_plan(res_dict)
        if 'after' in env_set.keys():
            for command in env_set['after']:
                self.db_connector.execute(query=command, set_env=True)
        nodes_info = list(self.plan_tool.get_plan_info(res_dict, query['query'], execute=False))
        flag = False
        op_item = None
        for node in nodes_info:
            if node['name'] == operator and self.check_data(all_filters, node):
                flag = True
                op_item = node
        return flag,op_item,res_dict['Plan']['Total Cost']

    def bucket_filter(self,operator,buckets,all_filters):
        new_buckets = [{'range':t['range'],'queries':[]} for t in buckets]

        # print("here")
        for idx,bucket in enumerate(buckets):
            for query in bucket['queries']:
                flag,op_item,cost = self.check_query(query,operator,all_filters)
                if flag:
                    query['info'] = op_item
                    query['cost'] = cost
                    new_buckets[idx]['queries'].append(query)
        return new_buckets

    def get_similar_sqls(self,num,queries,op,all_filters):
        res = set()
        if not len(queries):
            return []
        while len(queries)<2*num:
            queries += queries
        for query in queries:
            if len(res) > num:
                break
            new_query = self.sqlmodify_tool.modify_query_cardinality(query['sql'],op)
            if new_query == query['sql']  or new_query == '' or new_query in res or new_query == None:
                new_query = self.sqlmodify_tool.modify_query_comparison(query['sql'])
            if new_query != query['sql'] and new_query != '' and new_query not in res and new_query != None:
                flag, op_item,cost = self.check_query({'env':{},"query":new_query}, op, all_filters)
                if flag:
                    res.add(new_query)
        return [{"query":t,"env":{}} for t in res]

    def run_some_query(self,config,sql):
        res_dict = self.db_connector.explain(sql,execute=True,timeout=600000)
        if not res_dict:
            return {}
        deal_plan(res_dict)
        res = {}
        try:
            res['planinfo'] = res_dict
            res['query'] = sql
            res['config'] = config
        except:
            return {}
        return res

    def c0_param_data_gen(self,operator,check_dim,queries,hist,MIN_SAMPLE_NUM,setting_options):
        if len(queries)>20:
            queries = np.random.choice(queries,20,replace=False)
        env_setting_num_per_query = round(np.ceil(20/len(queries)))
        env_choice = np.linspace(0, 1, env_setting_num_per_query)
        results = []
        all_choice = [(x, y) for x in queries for y in env_choice]
        for query, i in tqdm(all_choice):
            self.db_connector.drop_cache()
            config = {}
            for c in setting_options.keys():
                if len(setting_options[c]):
                    value = np.random.choice(list(setting_options[c]))
                    self.db_connector.execute("set %s=%s;" % (c, value), set_env=True)
                else:
                    self.db_connector.set_knob_to_default(c)
                knob_value = self.db_connector.get_knob_value(c)
                config[c] = knob_value
            self.db_connector.set_knob_value(check_dim, i)
            knob_value = self.db_connector.get_knob_value(check_dim)
            config[check_dim] = knob_value
            res = self.run_some_query(config,query['sql'])
            if not len(res):
                continue
            results.append(res)
        with open(f"{self.file_path}/run_sqls.txt", 'a') as f:
            for item in results:
                f.writelines(json.dumps(item) + "\n")

        bucket_feats = {'runtime_cost': [], 'startup_cost': []}
        results = self.add_exec_pool(results, operator)
        bucket_feats['runtime_cost'] += results['runtime_cost']
        bucket_feats['startup_cost'] += results['startup_cost']

        feats = {'runtime_cost': [], 'startup_cost': []}
        for model_name in self.feature_tool.features[operator].keys():
            features = self.feature_tool.features[operator][model_name][
                           'node_features'] + self.feature_tool.leaf_features + self.feature_tool.target_feature
            for feat in bucket_feats[model_name]:
                feats[model_name].append([feat[feature] for feature in features])
        return feats, len(results)  # np.array(df)


    def getdata_with_filter(self,operator,check_dim,queries,hist,node,MIN_SAMPLE_NUM=20):
        self.MIN_SAMPLE_NUM = MIN_SAMPLE_NUM
        if self.knob_change:
            setting_options = {c: set() for c in dbms_cparams}
            for item in node.dataset:
                for idx,t in enumerate(node.node_features):
                    if t in setting_options.keys():
                        if self.db_connector.knobs[t].type == 'bool':
                            if item[idx] > 0.5:
                                setting_options[t].add('on')
                            else:
                                setting_options[t].add('off')
                        else:
                            setting_options[t].add(item[idx])

        filter = node.filters
        self.checked_dims[operator].append(check_dim)
        res = []
        if check_dim in dbms_cparams:
            return self.c0_param_data_gen(operator,check_dim,queries,hist,MIN_SAMPLE_NUM,setting_options)
        # buckets = copy.deepcopy(self.bucket_info.op_buckets[operator][check_dim])
        # buckets = self.sqlmodify_tool.modify_query_for_cparam(operator,check_dim,buckets,queries,filter)
        # buckets = self.bucket_filter(operator,buckets,filter)
        # buckets = self.get_pool_data(operator, check_dim, buckets,filter)
        # bucket_sqls,bucket_feats = self.bucket_choose(buckets,hist,node)
        # # if len(bucket_feats['runtime_cost'])<40:
        # bucket_sqls += self.get_similar_sqls(10,queries,operator,filter)
        #Ablation Study
        bucket_feats = {'runtime_cost':[],'startup_cost':[]}
        bucket_sqls = []
        if len(bucket_sqls)+len(bucket_feats['runtime_cost'])<self.MIN_SAMPLE_NUM:
            for i, data in enumerate(self.pool[operator]):
                if data['delete'] == 0:
                    res.append(data)
            sqls, ids = self.sample_select(res, check_dim,num=self.MIN_SAMPLE_NUM-(len(bucket_sqls)+len(bucket_feats['runtime_cost'])))
            count = 0
            for op in self.pool.keys():
                for item in self.pool[op]:
                    if item['query'] in sqls:
                        item['delete'] = 1
                        count += 1
            bucket_sqls+=[{"query":t,"env":{}} for t in sqls]
        plan_trees = []
        print("Runing Sqls")
        # bucket_sqls = []
        for sql in tqdm(bucket_sqls):
            self.db_connector.drop_cache()
            config = {}
            if self.knob_change:
                for c in setting_options.keys():
                    if self.knob_change and len(setting_options[c]):
                        value = np.random.choice(list(setting_options[c]))
                        self.db_connector.execute("set %s=%s;" % (c, value), set_env=True)
                    else:
                        self.db_connector.set_knob_to_default(c)
                    knob_value = self.db_connector.get_knob_value(c)
                    config[c] = knob_value
            if 'before' in sql['env'].keys():
                self.set_envs(sql['env']['before'])
            res_dict = self.db_connector.explain(sql['query'], execute=True, timeout=60000)
            try:
                if not res_dict:
                    continue
                deal_plan(res_dict)
                plan_trees.append({"query":sql['query'],"planinfo":res_dict,"config":config})
            except:
                print("here")
                continue
        if len(bucket_sqls):
            with open(f"{self.file_path}/run_sqls.txt", 'a') as f:
                for item in plan_trees:
                    f.writelines(json.dumps(item) + "\n")
            results = self.add_exec_pool(plan_trees,operator)
            bucket_feats['runtime_cost'] += results['runtime_cost']
            bucket_feats['startup_cost'] += results['startup_cost']
        feats = {'runtime_cost':[],'startup_cost':[]}
        for model_name in self.feature_tool.features[operator].keys():
            features = self.feature_tool.features[operator][model_name][
                           'node_features'] + self.feature_tool.leaf_features + self.feature_tool.target_feature
            for feat in bucket_feats[model_name]:
                feats[model_name].append([feat[feature] for feature in features])
        return feats,len(bucket_sqls)#np.array(df)


