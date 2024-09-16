import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from doepy import build
from tqdm import tqdm
from SALib.sample import saltelli

from database_util.database_info import *
from database_util.db_connector import *
from feature.plan import *
from feature.feature import *
from database_util.knob import *

from feature.infos import all_cparams

class LHS:
    def __init__(self,db,db_name,dir_path=None):
        self.db = db
        knobs = load_knobs_from_json_file()
        self.db.initial_tunning_knobs(knobs)
        self.scheme_info = Database_info(db_name)
        self.plan_tool = Plan_class(self.scheme_info, db=info_db)
        self.feature_tool = FeatureExtract()
        # self.pool = self.get_pool(dir_path,db_name)
        self.important_cparams = all_cparams

    def get_pool(self,dir_path,db_name):
        train_files = [f"{dir_path}/{db_name}_pool_scan.txt",
                       f"{dir_path}/{db_name}_pool_join.txt",
                       f"{dir_path}/{db_name}_pool_rand.txt",
                       f"{dir_path}/{db_name}_test.txt",]
        data = []
        for file_name in train_files:
            with open(file_name,'r') as f:
                data += f.readlines()

        opdatas = {"Seq Scan": [],
                   "Index Scan": [],
                   "Index Only Scan": [],
                   "Sort": [],
                   "Hash Join": [],
                   "Nested Loop": [],
                   "Merge Join": [],
                   "Aggregate": [],
                   }

        def get_op_info(plan):
            if 'Plans' in plan.keys():
                for item in plan['Plans']:
                    get_op_info(item)
            if plan['Node Type'] in opdatas.keys():
                res = self.plan_tool.get_op_info(plan, execute=True)
                feat = self.feature_tool.get_model_raw_feature(res, plan['Node Type'], 'runtime_cost')
                for item in ['Nt', 'No', 'Ni', 'Ns', 'Nr', 'Np', 'Nm', 'y']:
                    feat.pop(item)
                opdatas[plan['Node Type']].append({'feat':feat,'sql':plan_json['query']})
            else:
                pass

        for item in data:
            if 'Result' in item:
                continue
            plan_json = json.loads(item.strip())
            if plan_json['planinfo']['Execution Time']>100000:
                continue
            get_op_info(plan_json['planinfo']['Plan'])

        return opdatas

    def select_sqls(self,operators = ["Merge Join","Index Only Scan","Nested Loop","Hash Join","Index Scan","Seq Scan","Sort","Aggregate"]):
        opsets = { "Seq Scan": {},
                   "Index Scan": {},
                   "Index Only Scan": {},
                   "Sort": {},
                   "Hash Join": {},
                   "Nested Loop": {},
                   "Merge Join": {},
                   "Aggregate": {},
                }
        for op in operators:
            df = pd.DataFrame([t['feat'] for t in self.pool[op]])
            for key in df.columns:
                values = set(df[key])
                backet_num = 5
                if len(values)>backet_num:
                    opsets[op][key] = set(np.sort(list(set(df[key])), axis=None)[::int(len(values)/backet_num)])
                else:
                    opsets[op][key] = set(df[key])

        def item_new_info(op,item):
            for key in opsets[op].keys():
                if item[key] in opsets[op][key]:
                    return True
            return False

        def remove_info_in_set(op,item):
            for key in opsets[op].keys():
                if item[key] in opsets[op][key]:
                    opsets[op][key].remove(item[key])

        select_sqls = []
        count = 0
        for op in operators:
            while sum([len(opsets[op][t]) for t in opsets[op].keys()][:-1])!=0:
                for item in self.pool[op]:
                    # if count < 100 and op in ['Index Scan','Index Only Scan']:
                    #     select_sqls.append(item['sql'])
                    #     count += 1
                    #     continue
                    if item_new_info(op,item['feat']):
                        select_sqls.append(item['sql'])
                        for tem_op in self.pool.keys():
                            for op_item in self.pool[tem_op]:
                                if op_item['sql'] == item['sql']:
                                    feat = op_item['feat']
                                    remove_info_in_set(tem_op,feat)

        print(f"select {len(select_sqls)} sqls")
        return select_sqls

    def get_knob_sample(self,num_samples, knob_list, method = 'lhs'):

        knob_bound = {}
        for j in range(len(knob_list)):
            knob_bound[knob_list[j]] = [0,1]
        if method == 'lhs':
            samples = build.space_filling_lhs(
                knob_bound,
                num_samples=num_samples
            )
        elif method == 'sobol':
            bounds = np.zeros((len(knob_list),2))
            bounds[:,1] = 1
            problem = {
                'num_vars': len(knob_list),
                'names': knob_list,
                'bounds': bounds
            }
            samples_array = saltelli.sample(problem, int(num_samples/(len(knob_list)+2)),calc_second_order = False)
            samples = pd.DataFrame(samples_array,columns=knob_list)

        for j in range(len(samples)):
            for i in range(len(knob_list)):
                if self.db.knobs[knob_list[i]].type == 'bool':
                    if samples[knob_list[i]].iloc[j] >0.5:
                        samples[knob_list[i]].iloc[j] = 1
                    else:
                        samples[knob_list[i]].iloc[j] = 0

        return samples

    def run_some_query(self,f,config,sql):
        res_dict = self.db.explain(sql,execute=True,timeout=600000)
        res = {}
        try:
            res['planinfo'] = res_dict
            res['query'] = sql
            res['config'] = config
            f.writelines(json.dumps(res)+"\n")
        except:
            pass

    def _read_sqlgen_queries(self,file_name):
        sqls = []
        with open(file_name, 'r') as f:
            data = f.readlines()
            for i,line in enumerate(data):
                query = 'EXPLAIN (ANALYZE, COSTS, BUFFERS, VERBOSE, FORMAT JSON) ' + line.strip()
                sqls.append(query)
        return sqls

    def select_satisfy_sqls(self,filters,op,encoders):
        def item_satisfy(item):
            count = 0
            for key in filters.keys():
                if type(item['feat'][key]) == str:
                    if item['feat'][key] not in encoders[key].classes_:
                        count += 1
                        continue
                    else:
                        value = encoders[key].transform([item['feat'][key]])[0]
                else:
                    value = item['feat'][key]
                if value<filters[key][0]:
                    continue
                if value>filters[key][1]:
                    continue
                count += 1
            return count
        result = []
        for item in self.pool[op]:
            result.append({'sql':item['sql'],'score':item_satisfy(item)})
        sort_res = sorted(result, key=lambda i: i['score'], reverse=True)
        sqls = []
        max_score = sort_res[0]['score']
        for item in sort_res:
            if item['score'] == max_score:
                sqls.append(item['sql'])
            else:
                break
        return sqls

    def run_examples_al(self,samples,op,f,encoders):
#self.db.knobs[samples.columns[0]].normalize(samples['lock_timeout'].iloc[0])
        filters = {}
        for col in samples.columns:
            if col in important_cparams[op]['query']:
                filters[col] = [min(samples[col]),max(samples[col])]
        sqls = self.select_satisfy_sqls(filters,op,encoders)
        for sample_idx in tqdm(range(len(samples))):
            sql = np.random.choice(sqls)
            self.db.drop_cache()
            config = {}
            for j in range(len(important_cparams[op]['dbms'])):
                knob = important_cparams[op]['dbms'][j]
                value = self.db.knobs[knob].normalize(samples[knob].iloc[sample_idx])
                self.db.set_knob_value(knob, value)
                knob_value = self.db.get_knob_value(knob)
                config[knob] = knob_value
            # print("set %s = %s"%(knob_list[j],knob_value))
            self.run_some_query(f, config, sql)
            self.db.discard_session()


    def run_examples_test(self,savefile,sqlfile,num_samples,method = 'lhs'):
        select_sqls = self.select_sqls()
        # select_sqls = self._read_sqlgen_queries(sqlfile)
        # np.random.shuffle(samples)
        knob_list = self.db.ordered_knob_list
        samples = self.get_knob_sample(num_samples=num_samples, knob_list=knob_list, method=method)

        with open(savefile, 'w') as f:
            for i in tqdm(range(len(samples))):
                # sql = np.random.choice(select_sqls)
                # sql = select_sqls
                config = {}
                # modify setting
                for sql in tqdm(select_sqls):
                    self.db.drop_cache()
                    for j in range(len(knob_list)):

                        self.db.set_knob_value(knob_list[j], samples[knob_list[j]].iloc[i])
                        knob_value = self.db.get_knob_value(knob_list[j])
                        config[knob_list[j]] = knob_value
                    # print("set %s = %s"%(knob_list[j],knob_value))
                    self.run_some_query(f,config,sql)
                self.db.discard_session()

