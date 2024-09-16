from util.util import get_plantrees,get_test_results
from database_util.db_connector import *
from database_util.database_info import *
from feature.plan import Plan_class
from .sqlparse_util import *
from util.util import deal_plan
from tqdm import tqdm


class RandomSQL_Selector():

    def __init__(self, db, db_name,plan_tool):
        self.db = db
        self.db_name = db_name
        self.scheme_info = Database_info(db.db_name)
        self.plan_tool = plan_tool
        self.pool = self.get_pool()

    def parse(self,queries):

        res = []
        for query in tqdm(queries):
            res_dict = self.db.explain(query, execute=False, timeout=6000000)
            try:
                deal_plan(res_dict)
                plan_tree = res_dict['Plan']
                res.append({"query":query,
                        "plan":plan_tree,
                            "env":{},
                            "node":list(self.plan_tool.get_plan_info(res_dict,query,execute=False))})
            except:
                print(query)
        return res

    def get_queries(self,operator,cparams):
        # return []
        def get_key(info):
            res = ""
            for cparam in cparams:
                if type(info[cparam]) == str:
                    res += info[cparam] + "-"
                else:
                    res += str(round(info[cparam],2)) + "-"
            return res
        bucket = {}
        for item in self.pool:
            for node in item['node']:
                if node['name'] == operator:
                    res = get_key(node)
                    if res not in bucket.keys():
                        bucket[res] = []
                    bucket[res].append(item)
        res = []
        for key in bucket.keys():
            if len(bucket[key])>2:
                res += list(np.random.choice(list(bucket[key]),2,replace=False))
            else:
                res += list(bucket[key])
        for item in res:
            item['ast'] = sqlparse.parse(item['query'])[0],
        return res

    def get_pool(self):
        dir_path = "./data/benchmark/learnedsqlgen/"
        save_path = f"./data/temporary/randomquery_pool/{self.db_name}_pool.pickle"
        if os.path.exists(save_path):
            file = open(save_path, "rb")
            res = pickle.load(file)
        else:
            train_files = [f"{dir_path}/{self.db_name}_pool_scan.txt",
                           f"{dir_path}/{self.db_name}_pool_join.txt",
                           f"{dir_path}/{self.db_name}_pool_rand.txt",
                           f"{dir_path}/{self.db_name}_test.txt", ]
            plan_trees = get_plantrees(train_files,subplan=True)
            queries = [item['query'] for item in plan_trees]
            res = self.parse(queries)
            file = open(save_path, "wb")
            pickle.dump(res, file)
        return res

