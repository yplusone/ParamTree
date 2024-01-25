'''
Collect database information from pg_catalog and pg_statistic etc.
if you want to collect information from your own database, please modify the scheme_db_info in feature/info.py
'''

import re
import sys
import pickle
import json
import numpy as np
import decimal
import os

sys.path.extend(["../","./"])
from database_util.db_connector import Postgres_Connector
from feature.info import schema_db_info as db_info

class Database_info():
    def __init__(self,db_name):
        self.db_name  = db_name
        server = db_info['server'].replace('.','_')
        info_file = f'./schemeinfo/scheme_{db_name}_histogram_info.pickle'
        if os.path.exists(info_file):
            file = open(info_file,"rb")
            result = pickle.load(file)
            self.config_info = result['config']
            self.table_features = result['table']
            self.index_features = result['index']
            self.inner_bucket_size_info = result['inner_bucket_size_info']
        else:
            db_info['pg']['db_name'] = db_name
            self.db_connector = Postgres_Connector(server=db_info['server'], pg = db_info['pg'], ssh = db_info['ssh'])
            file = open(info_file, "wb")
            result = {}
            result['config']  =  self.get_config_infos()
            self.config_info = result['config']
            result['table'] = self.get_table_infos()
            self.table_features = result['table']
            result['inner_bucket_size_info'] = self.get_innerbucketsize_info()
            self.inner_bucket_size_info = result['inner_bucket_size_info']
            result['index'] = self.get_index_infos()
            self.index_features = result['index']
            pickle.dump(result, file)
            
        
    def get_config_infos(self):
        knob_file = './schemeinfo/conf.json'
        with open(knob_file) as f:
            confs = json.load(f)
        settings = confs.keys()
        res = {}
        for setting in settings:
            ans = self.db_connector.execute(f"select setting from pg_settings where name='{setting}'")[0][0]
            try:
                res[setting] = int(ans)
            except:
                if ans == "on" or ans == "off":
                    res[setting] = 1 if ans == "on" else 0
                else:
                    res[setting] = ans
        return res
        
    def get_index_infos(self):
        index_features = {}
        for table in self.table_features.keys():
            for index_name in self.table_features[table]['idx']:
                get_index_page_tuple_info = f"select relpages,reltuples from pg_class where relname='{index_name}';"
                index_page_tuple_info = self.db_connector.execute(get_index_page_tuple_info)
                index_features[index_name] = {}
                index_features[index_name]['pages'] = index_page_tuple_info[0][0]
                index_features[index_name]['tupls'] = index_page_tuple_info[0][1]
                inspect_info = self.db_connector.execute(f"SELECT * FROM bt_metap('\"{index_name}\"');")
                index_features[index_name]['tree_height'] = inspect_info[0][3]+1
                index_def = self.db_connector.execute(f"select indexdef from pg_indexes where indexname='{index_name}';")[0][0]    
                p1 = re.compile(r'[(](.*?)[)]', re.S)
                index_column = re.split(',', re.findall(p1, index_def)[0])
                index_column = [item.strip("\"") for item in index_column]
                index_features[index_name]['columns'] = [t.strip() for t in index_column]
                index_features[index_name]['key_column'] = len(index_column)
                index_features[index_name]['table'] = table
                if len(index_column) == 1:
                    index_features[index_name]['indexCorrelation'] = self.table_features[table]['columns'][index_column[0].strip("\"").strip()]['corr']
                    index_features[index_name]['type'] = self.table_features[table]['columns'][index_column[0].strip("\"").strip()]['type']
                    
                else:
                    index_features[index_name]['indexCorrelation'] = self.table_features[table]['columns'][index_column[0].strip()]['corr']*0.75
                    index_features[index_name]['type'] = 'multi'

                index_features[index_name]['distinctnum'] = self.inner_bucket_size_info[table][index_column[0].strip("\"").strip()][3]
        return index_features

    def get_table_infos(self):
        types_map = {"int4":"integer","numeric":"float","bpchar":"char","date":"date","varchar":"varchar","time":"date"}
        get_table_info_query = "SELECT \"table_name\" ,relpages,reltuples"\
	" FROM information_schema.tables t1, pg_class t2 WHERE table_schema = 'public' AND t1.\"table_name\" = t2.relname;"
        res = self.db_connector.execute(get_table_info_query)
        table_features = {}
        for i in range(len(res)):
            if res[i][1] == 0:
                continue
            table_name = res[i][0]
            table_features[table_name] = {}

            rtable_name = "\""+table_name+"\""

            table_features[table_name]['table_size'] = self.db_connector.execute(f"select pg_relation_size('{rtable_name}');")[0][0]
            table_features[table_name]['table_pages'] = res[i][1]
            table_features[table_name]['tuple_num'] = res[i][2]
            index_name = self.db_connector.execute(f"select indexname from pg_indexes where tablename='{table_name}';")
            table_features[table_name]['idx'] = [t[0] for t in index_name]
            table_features[table_name]['columns'] = {}
            get_attr_info = "SELECT "\
                            "base.\"column_name\","\
                            "t1.oid,"\
                            "col_description ( t1.oid, t2.attnum ),"\
                            "base.udt_name,"\
                            "COALESCE(character_maximum_length, numeric_precision, datetime_precision),"\
                            "correlation,"\
                            "(CASE "\
                                "WHEN ( SELECT t2.attnum = ANY ( conkey ) FROM pg_constraint WHERE conrelid = t1.oid AND contype = 'p' ) = 't' "\
                                "THEN 1 ELSE 0 "\
                            "END ) "\
                        "FROM "\
                            "information_schema.COLUMNS base,"\
                            "pg_class t1,"\
                            "pg_attribute t2,"\
                            "pg_stats t3 "\
                        "WHERE "\
                            f"base.\"table_name\" = '{table_name}' " \
                            "AND t1.relname = base.\"table_name\" "\
                            "AND t3.tablename = base.\"table_name\" "\
                            "AND t2.attname = base.\"column_name\" "\
                        "AND base.\"column_name\" = t3.attname "\
                            "AND t1.oid = t2.attrelid "\
                            "AND t2.attnum > 0 "\
                        "ORDER BY attnum;"    
            attr_info = self.db_connector.execute(get_attr_info)
            for item in attr_info:
                item[0] = item[0].strip("\"").strip()
            offset = 0
            for id,attr in enumerate(attr_info):
                table_features[table_name]['columns'][attr[0]] = {}
                table_features[table_name]['columns'][attr[0]]['type'] = attr[3] #types_map[attr[3]]
                if attr[4] == None:
                    table_features[table_name]['columns'][attr[0]]['width'] = 8
                else:
                    table_features[table_name]['columns'][attr[0]]['width'] = attr[4]
                table_features[table_name]['columns'][attr[0]]['index'] = id+1
                table_features[table_name]['columns'][attr[0]]['corr'] = attr[5]
                table_features[table_name]['columns'][attr[0]]['offset'] = offset
                
                if attr[4] == None:
                    attr[4] = 8
                offset += attr[4]
        return table_features
    
    def get_innerbucketsize_info(self):
        features = {}
        for table in self.table_features.keys():
            if table not in features.keys():
                features[table] = {}
            for column_name in self.table_features[table]['columns'].keys():
                info = self.get_pg_statistic_for_innerbucketsize(table,column_name)
                features[table][column_name] = info
        return features
                

    def get_pg_statistic_for_innerbucketsize(self,tbl_name,col_name):
        """
            Get statistis from pg_class/pg_attribute/pg_statistic.
        """
        reltuples=0
        mcv_freq=0
        stanullfrac=0
        stadistinct=0
        tbl_result=self.db_connector.execute('select relname,oid,reltuples from pg_class where relname=\''+tbl_name+'\';')[0]
        if not tbl_result: 
            print("No tbl name.")
            reltuples=-1
            return (reltuples,mcv_freq,stanullfrac,stadistinct)

        oid=tbl_result[1]
        reltuples=tbl_result[2]
        tbl_result=self.db_connector.execute('select attrelid,attname,attnum from pg_attribute where attname=\''+col_name+'\' and attrelid='+str(oid)+';')[0]
        if not tbl_result:
            print("No col name.")
            reltuples=-1
            return (reltuples,mcv_freq,stanullfrac,stadistinct)

        attnum=tbl_result[2]
        att_result=self.db_connector.execute('select * from pg_statistic where starelid='+str(oid)+' and staattnum='+str(attnum)+';') #【可能没有结果】
        if len(att_result)>0:
            results=att_result[0]
            stanullfrac=results[3]
            stadistinct=results[5]
            sta_array=[]
            for i in range(5):#0-4，
                sta_array.append([results[6+5*j+i] for j in range(5)]) #stakind1,staop1,stacoll,stanumbers1[1],stavalues1
                if(sta_array[i][0]==1):
                    mcv_array=sta_array[i][3] #{0.6,0.2}
                    mcv_freq=mcv_array[0]#first MCV freq
        return (reltuples,mcv_freq,stanullfrac,stadistinct)

    def scheme_info_append(self,fresh = False):
        scheme_file = f'./data/temporary/schemeinfo/scheme_{self.db_name}_histogram_info.txt'
        if os.path.exists(scheme_file) and not fresh:
            with open(scheme_file, 'r') as f:
                data = f.readlines()
                self.table_features = json.loads(data[0])
        else:
            with open(scheme_file, 'w') as f:
                for table in self.table_features.keys():
                    for column in self.table_features[table]['columns'].keys():
                        if self.table_features[table]['columns'][column]['type'] == "int4" or \
                                self.table_features[table]['columns'][column]['type'] == "numeric":
                            hb = self.db_connector.execute(
                                f"select histogram_bounds from pg_stats where tablename = '{table}' and attname = '{column}';")[
                                0][0]
                            if hb != None:
                                self.table_features[table]['columns'][column]['histogram_bounds'] = [
                                    float(t) for t in hb[1:-1].split(',')]
                            else:
                                ans = self.db_connector.execute(
                                    f"select distinct {column} from {table} order by {column};")
                                print(f"select distinct {column} from {table} order by {column};")
                                if ans[0][0] == None:
                                    self.table_features[table]['columns'][column]['histogram_bounds'] = []
                                else:
                                    if type(ans[0][0])==float or type(ans[0][0]) == decimal.Decimal:
                                        self.table_features[table]['columns'][column]['histogram_bounds'] = [
                                            float(t[0]) if t[0] and (type(t[0])==float or type(ans[0][0]) == decimal.Decimal) else 0 for t in ans]
                                    elif type(ans[0][0])==int:
                                        self.table_features[table]['columns'][column]['histogram_bounds'] = [
                                            int(t[0]) if type(t[0])==int else 0 for t in ans]
                                    else:
                                        self.table_features[table]['columns'][column]['histogram_bounds'] = [
                                             0 for t in ans]
                        if self.table_features[table]['columns'][column]['type'] == "date" or \
                                self.table_features[table]['columns'][column]['type'] == "bpchar":
                            ans = self.db_connector.execute(f"select distinct {column} from {table} order by {column};")
                            self.table_features[table]['columns'][column]['histogram_bounds'] = [str(t[0])
                                                                                                             for t in
                                                                                                             ans]
                f.writelines(json.dumps(self.table_features))
        for table in self.table_features.keys():
            column_info = self.table_features[table]['columns']
            for column in column_info.keys():
                if column_info[column]['type'] == 'int4':
                    column_info[column]['mtype'] = 'Integer'
                elif column_info[column]['type'] == 'numeric':
                    column_info[column]['mtype'] = 'Float'
                else:
                    column_info[column]['mtype'] = 'Str'

    
    def get_column_info(self,column):
        if '.' in column:
            table,col = column.split(".")
            if table in self.table_features.keys() and col in self.table_features[table]['columns'].keys():
                res = {}
                res['type'] = self.table_features[table]['columns'][col]['type']
                res['mtype'] = self.table_features[table]['columns'][col]['mtype']
                res['table'] = table
                res['offset'] = self.table_features[table]['columns'][col]['offset']
                res['column'] = col
                return res
        for table in self.table_features.keys():
            for col in self.table_features[table]['columns']:
                if col == column:
                    res = {}
                    res['type'] = self.table_features[table]['columns'][col]['type']
                    res['mtype'] = self.table_features[table]['columns'][col]['mtype']
                    res['table'] = table
                    res['offset'] = self.table_features[table]['columns'][col]['offset']
                    res['column'] = col
                    return  res
        longest_length = 0
        o_col,o_table = "",""
        for table in self.table_features.keys():
            for col in self.table_features[table]['columns']:
                if col in column:
                    if len(col)>longest_length:
                        o_col = col
                        o_table = table
                        longest_length = len(col)
        if o_col == "":
            return {}
        else:
            res = {}
            res['type'] = self.table_features[o_table]['columns'][o_col]['type']
            res['mtype'] = self.table_features[o_table]['columns'][o_col]['mtype']
            res['table'] = o_table
            res['offset'] = self.table_features[o_table]['columns'][o_col]['offset']
            res['column'] = o_col
            return  res

