import re
import os 
import traceback

class benchmarker():
    
    def __init__(self,db_name = 'tpch',query_num = 30,workload = "synthetic"):
        if 'tpch' in db_name:
            query_dir = './data/benchmark/tpch/'
            self.queries = self.read_tpch_queries(query_dir,query_num)
        elif 'tpcds' in db_name:
            query_dir = './data/benchmark/tpcds/sqls/'
            self.queries = self._read_tpcds_queries(query_dir,query_num)
        elif 'imdb' in db_name:
            if workload  == 'job':
                query_dir = './data/benchmark/imdb/job/'
            else:
                query_dir = f'./data/benchmark/imdb/workloads/{workload}.sql'
            self.queries = self._read_imdb_queries(query_dir,workload)

    def _read_imdb_queries(self,query_dir,workload):
        sqls = []
        if workload == 'job':
            for file in os.listdir(query_dir):
                file_name = query_dir + file
                with open(file_name, "r") as f:
                    sql = f.read().strip()
                    sqls.append({'sql':'EXPLAIN (ANALYZE, COSTS, BUFFERS, VERBOSE, FORMAT JSON) '+ sql ,'template':workload})
        else:
            with open(query_dir, "r") as f:
                data = f.readlines()
                for item in data:
                    sqls.append({'sql':'EXPLAIN (ANALYZE, COSTS, BUFFERS, VERBOSE, FORMAT JSON) '+ item.strip() ,'template':workload})
        return sqls

    def get_queries(self):
        return self.queries

    def deal_sql(self,sql):
        if 'interval' in sql:
            interval_str_origin = re.findall(r'interval.*?[\n,;]',sql)[0]
            num = re.findall(r'\d+',interval_str_origin)[0]
            if 'day' in interval_str_origin:
                interval_str = 'interval \''+str(num)+' days\''+interval_str_origin[-1]
            if 'year' in interval_str_origin:
                interval_str = 'interval \''+str(num)+' years\''+interval_str_origin[-1]
            if 'month' in interval_str_origin:
                interval_str = 'interval \''+str(num)+' months\''+interval_str_origin[-1]
            line = sql.replace(interval_str_origin,interval_str)
            return line
        else:
            return sql

    def deal_tpcds_sql(self,sql):
        interval_str_origin = ''
        if len(re.findall(r'\d+\sdays\)',sql)):
            interval_str_origin = re.findall(r'\d+\sdays\)',sql)[0]
            num = re.findall(r'\d+',interval_str_origin)[0]
        if interval_str_origin!='':
            if 'day' in interval_str_origin:
                interval_str = 'interval \''+str(num)+' days\''+interval_str_origin[-1]
            if 'year' in interval_str_origin:
                interval_str = 'interval \''+str(num)+' years\''+interval_str_origin[-1]
            if 'month' in interval_str_origin:
                interval_str = 'interval \''+str(num)+' months\''+interval_str_origin[-1]
            line = sql.replace(interval_str_origin,interval_str)
            return line
        else:
            return sql

    def read_tpch_queries(self, query_dir,repeat_num):
        sqls = []
        file_count = 0
        for file in os.listdir(query_dir)[:repeat_num]:
            count = 1
            if file[0]!='s' or file[-1]!='t':
                continue
            file_name = query_dir+file
            with open(file_name,mode = "r") as f:
                data = f.read()
                view_sql = re.findall(r"\n\ncreate.*?;",data,re.S)
                view_sql = self.deal_sql(view_sql[0][2:])
                res = re.findall(r"\n\n\nselect.*?;",data,re.S)
                for sql in res:
                    if count in [15]:
                        with_sql = view_sql.replace('create view','with').replace('as\n','as (\n')[:-1]+")\n"
                        sqls.append(
                            {'sql': 'EXPLAIN (ANALYZE, COSTS, BUFFERS, VERBOSE, FORMAT JSON) ' + with_sql +  self.deal_sql(sql[3:]),
                             'template': count})
                    else:
                        sqls.append({'sql':'EXPLAIN (ANALYZE, COSTS, BUFFERS, VERBOSE, FORMAT JSON) '+self.deal_sql(sql[3:]),'template':count})
                    count+=1
        return sqls
    
    def _read_tpcds_queries(self, query_dir,repeat_num):
        sqls = []
        for dirfile in os.listdir(query_dir)[:repeat_num]:
            for file in os.listdir(query_dir+dirfile):
                file_name = query_dir+dirfile+"/"+file
                template = int(re.findall(r"\d+",file)[0])
                # filter out the sqls cant compatible with postgres
                if template not in [3, 6, 7, 8, 9, 13, 15, 17, 18, 19,  22, 24, 25, 26, 27, 28, 29, 30, 31, 33,38, 39, 41, 42, 43, 44, 45, 46, 48,49, 50,51, 52,53,54, 55,56,57, 58, 59,60, 61, 62,63, 64, 65,66,67, 68, 69,71, 72, 73, 75, 76, 78, 79, 81, 83, 84, 85, 87, 88,89, 90, 91,93,96,97]:
                    continue
                with open(file_name, "r") as f:
                    try:
                        data = f.read()
                        data = data.split(";")[0]+";"
                        data = self.deal_tpcds_sql(data)
                        sqls.append({'sql':'EXPLAIN (ANALYZE, COSTS, BUFFERS, VERBOSE, FORMAT JSON) '+ data,'template':template})
                    except:
                        traceback.print_exc()
                        print(file_name)
        return sqls

    def get_query_num(self):
        return len(self.queries)

    def get_query_by_index(self,index):
        try:
            return self.queries[index]['sql'],self.queries[index]['template']
        except:
            return None  

            

if __name__ == "__main__":
    runner = benchmarker(db_name="imdb",query_num=1,workload='synthetic',)
    for i in range(50):
        print(runner.queries[i]['sql'][56:])
