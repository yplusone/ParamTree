import re
from tqdm import tqdm
import sys

sys.path.extend(["../", "./"])
from database_util.db_connector import *
from feature.plan import *
import random
import json
from database_util.benchmarker import benchmarker
from database_util.database_info import *
from knob import *
import os
import paramiko


def get_op(plan):
    if 'Plans' in plan.keys():
        for item in plan['Plans']:
            for x in get_op(item):
                yield x
    yield plan['Node Type']

class gendataset():

    def __init__(self, db, queries):
        self.db = db
        knobs = load_knobs_from_json_file()
        self.db.initial_tunning_knobs(knobs)
        self.queries = queries
        self.db_name = db_name
        self.scheme_info = Database_info(db_name)


    def run_some_query(self, f, config, sql):
        # self.db.drop_cache()
        res_dict = self.db.explain(sql, execute=True, timeout=600000)
        res = {}
        try:
            res['planinfo'] = res_dict
            res['query'] = sql
            res['config'] = config
            f.writelines(json.dumps(res) + "\n")
        except:
            print("wrong")
    def run_hash_join(self, save_file):
        sqls = []
        # self.run_all_queries_save_raw_data(f'{save_dir}{save_name}_0.txt',{})
        step = 3
        n_steps = np.arange(step) / step
        bool_step = [0, 1]
        knob_list = self.db.ordered_knob_list
        with open(save_file, 'w') as f:
            for sql in tqdm(self.queries):

                config = {}
                # modify setting
                self.db.drop_cache()
                for j in range(len(knob_list)):
                    # if knob_list[j] != 'cursor_tuple_fraction':
                    #     continue
                    if self.db.knobs[knob_list[j]].type == 'bool':
                        val = random.choice(bool_step)
                    else:
                        val = random.choice(n_steps)
                    self.db.set_knob_value(knob_list[j], val)
                    knob_value = self.db.get_knob_value(knob_list[j])
                    config[knob_list[j]] = knob_value
                    # print("set %s = %s"%(knob_list[j],knob_value))
                self.db.execute("set work_mem =  409600",set_env = True)
                config['work_mem'] = 40960
                self.run_some_query(f, config, sql)
                self.db.discard_session()

    def run_queries(self, save_file):
        sqls = []
        # self.run_all_queries_save_raw_data(f'{save_dir}{save_name}_0.txt',{})
        step = 3
        n_steps = np.arange(step) / step
        bool_step = [0, 1]
        knob_list = self.db.ordered_knob_list
        with open(save_file, 'w') as f:
            for query in tqdm(self.queries):
                sql = query['sql']
                config = {}
                # modify setting
                flag = False
                for _ in range(5):
                    o_res_dict = self.db.explain(sql,execute=False,timeout=1000)
                    knob_res = []
                    for j in range(len(knob_list)):
                        # if knob_list[j] != 'cursor_tuple_fraction':
                        #     continue
                        if self.db.knobs[knob_list[j]].type == 'bool':
                            val = random.choice(bool_step)
                        else:
                            val = random.choice(n_steps)
                        knob_res.append(val)
                        self.db.set_knob_value(knob_list[j], val)
                        knob_value = self.db.get_knob_value(knob_list[j])
                        config[knob_list[j]] = knob_value
                        # print("set %s = %s"%(knob_list[j],knob_value))
                    # run all query
                    n_res_dict = self.db.explain(sql, execute=False,timeout=1000)
                    if not o_res_dict or not n_res_dict:
                        continue
                    diff = max(n_res_dict['Plan']['Total Cost']/o_res_dict['Plan']['Total Cost'],o_res_dict['Plan']['Total Cost']/n_res_dict['Plan']['Total Cost'])
                    if diff<1.5:
                        self.db.discard_session()
                        continue
                    else:
                        n = ''.join(list(get_op(n_res_dict['Plan'])))
                        o = ''.join(list(get_op(o_res_dict['Plan'])))
                        if n != o:
                            print("here")
                            continue
                        flag  = True
                if not flag:
                    continue
                self.db.drop_cache()
                for j in range(len(knob_list)):
                    val = knob_res[j]
                    self.db.set_knob_value(knob_list[j], val)
                    knob_value = self.db.get_knob_value(knob_list[j])
                    config[knob_list[j]] = knob_value
                self.run_some_query(f, config, sql)
                self.db.discard_session()


if __name__ == "__main__":

    db_name = 'imdb'
    db = Postgres_Connector(server="47.96.181.98",
        pg = {'db_name':db_name,
            'username':"postgres",
            'password':"postgres",
            'port':58888,
            'command_ctrl':"docker exec -it  --user postgres database /home/yjn/pgsql13.1/bin/pg_ctl -D /home/yjn/pgsql13.1_data"},
        ssh = {
            'username':"root",
            'password':"daijian.83",
            'port':22
        })
    # runner = benchmarker(db_name=db_name, query_num=1, workload='scale')
    with open("./data/benchmark/synthetic.txt")  as f:
        data = f.readlines()
    runner = gendataset(db, data)
    runner.run_hash_join(save_file=f'./data/experiment/{db_name}_scale_knob.txt')
