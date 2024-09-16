import os

from feature.infos import all_cparams,dbms_cparams,features,query_cparams
from database_util.db_connector import Postgres_Connector
from database_util.database_info import Database_info
from feature.plan import Plan_class
from feature.feature import FeatureExtract
from feature.infos import schema_db_info  as db_info
import pickle
from database_util.knob import load_knobs_from_json_file
import re
import numpy as np
import json
from util.util import get_plantrees
BUCKET_NUM = 10
class Bucket:
    def __init__(self,db_name):
        if db_name in ['imdb','tpch','tpcds']:

            db_info['pg']['db_name'] = db_name
            self.db = Postgres_Connector(server=db_info['server'], pg=db_info['pg'], ssh=db_info['ssh'])

            self.op_buckets = self.get_bucket(db_name)

    def get_pool(self,db_name):
        dir_path = "./data/temporary/randomquery_pool"
        file_names = [
            f"{dir_path}/{db_name}_pool_scan.txt",
            f"{dir_path}/{db_name}_pool_join.txt",
            f"{dir_path}/{db_name}_test.txt"]
        data = []
        plan_trees = get_plantrees(file_names,subplan=True)

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
                opdatas[plan['Node Type']].append({'feat': feat, 'query': plan_json['query'], 'delete': 0})
            else:
                pass

        for plan_json in plan_trees:
            if plan_json['planinfo']['Execution Time'] > 100000:
                continue
            get_op_info(plan_json['planinfo']['Plan'])
        return opdatas

    def get_bucket(self,db_name):
        save_path = f"./data/temporary/buckets/{db_name}_bucket.pickle"
        if os.path.exists(save_path):
            file = open(save_path, "rb")
            return pickle.load(file)
        else:
            knobs = load_knobs_from_json_file()
            self.db.initial_tunning_knobs(knobs)
            self.scheme_info = Database_info(db_name)
            self.scheme_info.scheme_info_append()
            self.plan_tool = Plan_class(self.scheme_info)
            self.feature_tool = FeatureExtract()
            self.c0_cparams = dbms_cparams
            self.predefined_cparams = ['FilterNum', 'FilterOffset', 'FilterIntegerRatio', 'FilterFloatRatio',
                                       'FilterStrRatio', 'FilterColumnNum',
                                       'CondNum', 'CondOffset', 'CondIntegerRatio', 'CondFloatRatio', 'CondStrRatio',
                                       'CondColumnNum',
                                       'IndexCorrelation', 'IndexTreeHeight', 'IndexTreePages', 'IndexTreeUniqueValues',
                                       'ParentOp', 'LeftOp', 'RightOp',
                                       'InnerUnique', 'TablePages', 'TuplesPerBlock',
                                       'Strategy'
                                       ]
            self.afterdefined_cparams = ['SoutAvg', 'Rows', 'Loops', 'LeftSoutAvg', 'LeftRows', 'LeftLoops',
                                         'RightSoutAvg', 'RightRows', 'RightLoops', 'Selectivity', 'BucketsNum',
                                         'BatchesNum']
            self.pool = self.get_pool(db_name)
            res = {}
            for op in features.keys():
                res[op] = {}
                for cparam in dbms_cparams:
                    res[op][cparam] = self.get_predefined_bucket(cparam,op)
                for cparam in query_cparams:
                    if cparam in self.predefined_cparams:
                        res[op][cparam] = self.get_predefined_bucket(cparam,op)
                    elif cparam in self.afterdefined_cparams:
                        res[op][cparam] = self.get_afterdefined_bucket(cparam,op)
            file = open(save_path, "wb")
            pickle.dump(res, file)
            return res

    def get_predefined_bucket(self,cparam,operator):
        buckets = []
        if cparam in self.c0_cparams:
            if self.db.knobs[cparam].type == 'bool':
                buckets.append({"range": [0, 0], "queries": []})
                buckets.append({"range": [0, 1], "queries": []})
            else:
                for i in range(BUCKET_NUM):
                    l_bound = self.db.knobs[cparam].denormalize(i*1/BUCKET_NUM)
                    u_bound = self.db.knobs[cparam].denormalize((i+1)*1/BUCKET_NUM)
                    buckets.append({"range":[l_bound,u_bound],"queries":[]})
        else:
            infos = {}
            if cparam in ['IndexCorrelation', 'IndexTreeHeight', 'IndexTreePages', 'IndexTreeUniqueValues','TablePages', 'TuplesPerBlock','CondNum', 'CondOffset', 'CondIntegerRatio', 'CondFloatRatio', 'CondStrRatio', 'CondColumnNum']:
                if cparam in ['TablePages', 'TuplesPerBlock',]:
                    for table in self.scheme_info.table_features.keys():
                        if cparam == 'TablePages':
                            num = self.scheme_info.table_features[table]['table_pages']
                        elif cparam == 'TuplesPerBlock':
                            num = round(self.scheme_info.table_features[table]['tuple_num'] /
                                        self.scheme_info.table_features[table]['table_pages'], 2)
                        num = round(num, 2)
                        infos[num] = set()
                elif cparam in ['IndexCorrelation', 'IndexTreeHeight', 'IndexTreePages', 'IndexTreeUniqueValues']:
                    for index in self.scheme_info.index_features.keys():
                        if cparam == 'IndexCorrelation':
                            num = round(self.scheme_info.index_features[index]['indexCorrelation'], 2)
                        elif cparam == 'IndexTreeHeight':
                            num = round(self.scheme_info.index_features[index]['tree_height'])
                        elif cparam == 'IndexTreePages':
                            num = round(self.scheme_info.index_features[index]['pages'])
                        elif cparam == 'IndexTreeUniqueValues':
                            num = self.scheme_info.index_features[index]['distinctnum']
                        infos[num] = set()
                elif cparam in ['CondNum', 'CondOffset', 'CondIntegerRatio', 'CondFloatRatio', 'CondStrRatio',
                                'CondColumnNum', ]:
                    if cparam == 'CondNum':
                        for i in range(5):
                            infos[i] = set()
                    else:
                        for index in self.scheme_info.index_features.keys():
                            if cparam == 'CondOffset':
                                offset = 0
                                for col in self.scheme_info.index_features[index]['columns']:
                                    info = self.scheme_info.get_column_info(col)
                                    if info['offset'] > offset:
                                        offset = info['offset']
                                num = offset
                            elif cparam == 'CondColumnNum':
                                num = len(self.scheme_info.index_features[index]['columns'])
                            elif cparam in ['CondIntegerRatio', 'CondFloatRatio', 'CondStrRatio']:
                                cparam_type = re.findall("Cond(.*?)Ratio", cparam)[0]
                                n = 0
                                for col in self.scheme_info.index_features[index]['columns']:
                                    info = self.scheme_info.get_column_info(col)
                                    if info['mtype'] == cparam_type:
                                        n += 1
                                num = round(n / len(self.scheme_info.index_features[index]['columns']), 2)
                            infos[num] = set()
                keys = list(infos.keys())
                keys.sort()
                last_key = -np.Inf
                for key in keys:
                    buckets.append({
                        "range": [last_key, key],
                        "queries": []
                    })
                    last_key = key
                buckets.append({
                        "range": [last_key, np.Inf],
                        "queries": []
                    })
            elif cparam in ['FilterNum', 'FilterOffset', 'FilterIntegerRatio', 'FilterFloatRatio', 'FilterStrRatio', 'FilterColumnNum',"SoutAvg"]:
                range_info = {
                    "FilterIntegerRatio": [0, 1],
                    "FilterFloatRatio": [0, 1],
                    "FilterStrRatio": [0, 1],
                    "FilterNum": [0, 10],
                    "FilterColumnNum": [0, 10],
                    "FilterOffset": [0, 300],
                    "SoutAvg":[0,40]
                }
                crange = range_info[cparam]
                max_c = crange[1]
                min_c = crange[0]
                for i in range(BUCKET_NUM):
                    buckets.append({"range": [
                    ((max_c - min_c) / BUCKET_NUM) * i + min_c, ((max_c - min_c) / BUCKET_NUM) * (i + 1) + min_c],
                                    "queries": []})
                buckets[-1]['range'][1] = np.Inf
            elif cparam in ['ParentOp',  'LeftOp', 'RightOp']:
                for value in ['Aggregate', 'Sort', 'Seq Scan', 'Index Scan','Index Only Scan','Limit', 'Hash Join', 'Hash', 'Nested Loop', 'Materialize','Merge Join', 'Subquery Scan','Group','Incremental Sort','CTE Scan','Append','None','others']:
                    buckets.append({"range": [value],
                                    "queries": []})
            elif cparam in ['InnerUnique']:
                buckets.append({"range": [0, 0], "queries": []})
                buckets.append({"range": [0, 1], "queries": []})
            elif cparam in ['Strategy']:
                if operator in ['Hash Join','Merge Join','Nested Loop']:
                    types = ['Semi', 'Inner', 'Anti', 'Full', 'Right','Left','others']
                elif operator in ['Sort']:
                    types = ['quicksort', 'top-n heapsort','external merge','external sort','others']
                elif operator in ['Aggregate']:
                    types = ['Plain', 'Sorted', 'Hashed','others']
                elif operator in ['Seq Scan', 'Index Scan','Index Only Scan']:
                    types = ['Forward','Backward','none']
                else:
                    raise Exception("wrong")
                for type in types:
                    buckets.append({"range": [type],
                                    "queries": []})
            else:
                raise Exception("wrong")
        return buckets
    def get_afterdefined_bucket(self,cparam,operator):
        buckets = []
        values = [item['feat'][cparam] for item in self.pool[operator]]
        max_c = max(values)
        min_c = min(values)
        for i in range(BUCKET_NUM):
            buckets.append({"range": [
                ((max_c - min_c) / BUCKET_NUM) * i + min_c, ((max_c - min_c) / BUCKET_NUM) * (i + 1) + min_c],
                "queries": []})
        buckets[0]['range'][0] = 0
        buckets[-1]['range'][1] = np.inf
        return buckets

if __name__ == "__main__":
    tool = Bucket('tpch')
