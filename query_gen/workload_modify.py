import sqlparse
import random
import copy
from tqdm import tqdm
from database_util.benchmarker import benchmarker
from database_util.db_connector import *
from database_util.database_info import *
from feature.plan import Plan_class
from .sqlparse_util import *
from .randomsql_select import RandomSQL_Selector
from .bucket import Bucket
from util.util import deal_plan
BUCKET_NUM = 30
BUCKET_GENERATE_SAMPLES = 3

class SqlModify():
    def __init__(self, db,scheme_info):
        self.db = db
        self.scheme_info = scheme_info
        self.scheme_info.scheme_info_append()
        self.plan_tool = Plan_class(self.scheme_info)
        self.comparisons = self.collect_comparisons()
        self.randomsql_selector_tool = RandomSQL_Selector(self.db,self.db.db_name,self.plan_tool)
        self.key_words = self.get_key_word()
    def parse(self,queries):
        res = []
        for query in queries:
            res_dict = self.db.explain(query['sql'], execute=False, timeout=6000000)
            deal_plan(res_dict)
            plan_tree = res_dict['Plan']
            template = query['template'] if 'template' in query.keys() else 0
            res.append({"ast":sqlparse.parse(query['sql'])[0],
                        "query":f"--template{template}\n"+query['sql'],
                        "template":template,
                       "plan":plan_tree,
                        "env":{},
                        "node":list(self.plan_tool.get_plan_info(res_dict,query['sql'],execute=False))})
        return res

    def parse_with_envset(self,queries,operator):
        res = []
        for idx,query in enumerate(queries):
            res_dict = self.db.explain(query['sql'], execute=False, timeout=6000000)
            deal_plan(res_dict)
            plan_tree = res_dict['Plan']
            if query['template'] in ['synthetic','job-light','scale','job']:
                template = idx
            else:
                template = query['template'] if 'template' in query.keys() else 0
            res.append({"query": f"--template{template}\n" + query['sql'],
                        "template": template,
                        "plan": plan_tree,
                        "env": {},
                        "node": list(self.plan_tool.get_plan_info(res_dict, query['sql'], execute=False))})

            if operator in ['Hash Join','Nested Loop','Merge Join']: #'Seq Scan','Index Scan','Index Only Scan',
                env_set = {'before':[],'after':[]}
                if operator in ['Seq Scan','Index Scan','Index Only Scan']:
                    for op in list(set(['Seq Scan','Index Scan','Index Only Scan'])-set([operator])):
                        cm = op.lower().replace(" ", "")
                        env_set['before'].append(f"set enable_{cm}=off;")
                        env_set['after'].append(f"set enable_{cm}=on;")
                elif operator in ['Hash Join','Nested Loop','Merge Join']:
                    for op in list(set(['Hash Join','Nested Loop','Merge Join'])-set([operator])):
                        cm = op.lower().replace(" ","")
                        if op == 'Nested Loop':
                            cm = 'nestloop'
                        env_set['before'].append(f"set enable_{cm}=off;")
                        env_set['after'].append(f"set enable_{cm}=on;")
                for command in env_set['before']:
                    self.db.execute(query = command,set_env=True)
                res_dict = self.db.explain(query['sql'], execute=False, timeout=6000000)
                deal_plan(res_dict)
                for command in env_set['after']:
                    self.db.execute(query = command,set_env=True)
                res.append({"query": f"--template{template}\n" + query['sql'],
                            "env": env_set,
                            "template": template,
                            "plan": plan_tree,
                            "node": list(self.plan_tool.get_plan_info(res_dict, query['sql'], execute=False))})

        return res

    def get_key_word(self):
        sql_case =  "EXPLAIN (ANALYZE, COSTS, BUFFERS, VERBOSE, FORMAT JSON) select\n\ts_acctbal,\n\ts_name,\n\tn_name,\n\tp_partkey,\n\tp_mfgr,\n\ts_address,\n\ts_phone,\n\ts_comment\nfrom\n\tpart,\n\tsupplier,\n\tpartsupp,\n\tnation,\n\tregion\nwhere\n\tp_partkey = ps_partkey\n\tand s_suppkey = ps_suppkey\n\tand p_size = 22\n\tand p_type like '%TIN'\n\tand s_nationkey = n_nationkey\n\tand n_regionkey = r_regionkey\n\tand r_name = 'ASIA'\n\tand ps_supplycost = (\n\t\tselect\n\t\t\tmin(ps_supplycost)\n\t\tfrom\n\t\t\tpartsupp,\n\t\t\tsupplier,\n\t\t\tnation,\n\t\t\tregion\n\t\twhere\n\t\t\tp_partkey = ps_partkey\n\t\t\tand s_suppkey = ps_suppkey\n\t\t\tand s_nationkey = n_nationkey\n\t\t\tand n_regionkey = r_regionkey\n\t\t\tand r_name = 'ASIA'\n\t)\norder by\n\ts_acctbal desc,\n\tn_name,\n\ts_name,\n\tp_partkey;"
        result = {}
        ast = sqlparse.parse(sql_case)[0]
        keywords = list(get_keyword_obj(ast))
        for key in keywords:
            result[str(key).lower()] = key
        return result

    def collect_comparisons(self,fresh=False):
        comparison_file = f"./data/temporary/comparisons/{self.db.db_name}_comparisons.pickle"
        if os.path.exists(comparison_file) and not fresh:
            file = open(comparison_file, "rb")
            comparisons = pickle.load(file)
        else:
            comparisons = {}
            for table in self.scheme_info.table_features.keys():
                comparisons[table] = {}
                for col in self.scheme_info.table_features[table]['columns']:
                    comparisons[table][col] = set()
            data = self.parse(benchmarker(db_name=self.db.db_name, query_num=100).queries)
            for query in data:
                namelist = get_name_list(query['ast'])
                res = get_query_comparison(query['ast'],self.scheme_info,namelist)
                for item in res:
                    if 'sum' in item['comparison'] or 'avg' in item['comparison'] or 'min' in item['comparison'] or 'max' in item['comparison'] or 'count' in item['comparison']:
                       continue
                    comparisons[item['table']][item['column']].add(item['comparison'])
            for table in self.scheme_info.table_features.keys():
                for col in self.scheme_info.table_features[table]['columns']:
                    if len(comparisons[table][col])<20:
                        if self.scheme_info.table_features[table]['columns'][col]['mtype'] in ['Integer','Float']:
                            if len(self.scheme_info.table_features[table]['columns'][col]['histogram_bounds']):
                                sample_values = np.random.choice(self.scheme_info.table_features[table]['columns'][col]['histogram_bounds'],min(len(self.scheme_info.table_features[table]['columns'][col]['histogram_bounds']),20))
                                for idx in range(len(sample_values)):
                                    if idx%2 == 0:
                                        comparisons[table][col].add(f"{col} > {sample_values[idx]}")
                                    else:
                                        comparisons[table][col].add(f"{col} < {sample_values[idx]}")
            for table in self.scheme_info.table_features.keys():
                for col in self.scheme_info.table_features[table]['columns']:
                    comparisons[table][col] = list(comparisons[table][col])
            file = open(comparison_file, "wb")
            pickle.dump(comparisons, file)
            print(f"Model saved in {comparison_file}")
        return comparisons

    def find_bucket_item(self,buckets,value):
        if type(value) == str:
            for idx in range(len(buckets)):
                if value in buckets[idx]['range']:
                    return idx
        else:
            for idx in range(len(buckets)):
                if idx == 0 and value>= buckets[idx]['range'][0] and value<=buckets[idx]['range'][1]:
                    return idx
                elif value>buckets[idx]['range'][0] and value<=buckets[idx]['range'][1]:
                    return idx
        return None

    def get_bucket(self,sqls,buckets,cparam,operator):
        def get_cparam_value(node):
            if node['Node Type'] == operator:
                info = self.plan_tool.get_op_info(node,execute=False)
                yield info[cparam]
            if 'Plans' in node.keys():
                for item in node['Plans']:
                    item['parent'] = node['Node Type']
                    for x in get_cparam_value(item):
                        yield x


        for sql in sqls:
            res_dict = self.db.explain(sql, execute=False, timeout=6000000)
            try:
                deal_plan(res_dict)
                plan_tree = res_dict['Plan']
            except:
                pass
            for t in get_cparam_value(plan_tree):
                key = self.find_bucket_item(buckets,t)
                if key == None:
                    continue
                if sql not in buckets[key]['queries']:
                    buckets[key]['queries'].append({"query":sql,"env":{}})
        return



    def change_filter_ratio(self, operator, cparam, where_clause, query, tables,buckets,namelist):
        cparam_type = re.findall("Filter(.*?)Ratio",cparam)[0]
        namelist_verse = {namelist[t]: t for t in namelist}
        num = {"Integer": 0, "Float": 0, "Str": 0}
        comparisn_choices = {"Integer": {}, "Float": {}, "Str": {}}
        for table in tables:
            if table not in self.scheme_info.table_features.keys():
                if table in namelist.keys():
                    table = namelist[table]
                else:
                    continue
            if table not in self.scheme_info.table_features.keys():
                continue
            for col in self.comparisons[table].keys():
                if len(self.comparisons[table][col]):
                    type = self.scheme_info.get_column_info(col)['mtype']
                    num[type] += 1
                    comparisn_choices[type][(table,col)] = self.comparisons[table][col]
        num_choices = {}
        for i in range(num["Integer"]+1):
            for j in range(num['Float']+1):
                for k in range(num['Str']+1):
                    if i+j+k > 10:
                        continue
                    choice = {"Integer": i, "Float": j, "Str": k}
                    ratio = round(choice[cparam_type]/(i+j+k),2) if (i+j+k)>0 else 0
                    if ratio not in num_choices.keys():
                        num_choices[ratio] = [choice]
                    else:
                        num_choices[ratio].append(choice)
        sqls = []
        get_join_conditions = [item['comparison'] for item in get_join_comparison(where_clause['item'],self.scheme_info)]
        c_query, c_where_clause = copy_item_and_parsed_sql(query, where_clause)
        #generate BUCKET_GENERATE_SAMPLES samples for every bucket
        for idx,item in enumerate(buckets):

            filter_number_choice = []
            for key in num_choices.keys():
                if (idx>0 and key>item['range'][0] and key<=item['range'][1]) or (idx==0 and key>=item['range'][0] and key<=item['range'][1]):
                    filter_number_choice += num_choices[key]
            if not(len(filter_number_choice)):
                continue
            choices = np.random.choice(filter_number_choice,BUCKET_GENERATE_SAMPLES)
            for choice in choices:
                # c_query, c_where_clause = copy_item_and_parsed_sql(query, where_clause)
                comparisons = get_join_conditions.copy()
                for mtype in ['Integer','Float','Str']:
                    if len(comparisn_choices[mtype].keys())<choice[mtype]:
                        break
                    column_sample = np.random.choice(np.arange(len(comparisn_choices[mtype].keys())),choice[mtype],replace=False)
                    # comparisons += [np.random.choice(comparisn_choices[mtype][col],1)[0] for col in column_sample]

                    for i in column_sample:
                        t = list(comparisn_choices[mtype].keys())[i]
                        if t[0] in namelist_verse.keys():
                            comparisons.append(
                                namelist_verse[t[0]] + "." + np.random.choice(comparisn_choices[mtype][t], 1)[0])
                        else:
                            comparisons.append(t[0] + "." + np.random.choice(comparisn_choices[mtype][t], 1)[0])

                if not(len(comparisons)):
                    t_query, t_where_clause = copy_item_and_parsed_sql(query, where_clause)
                    delete_item_in_parsed_sql(t_query,t_where_clause)
                else:
                    gen_where = "where "+" and ".join(comparisons)+" "
                    gen_where = sqlparse.parse(gen_where)[0]
                    c_where_clause.tokens = gen_where.tokens
                if str(query) == str(c_query):
                    continue
                sqls.append(str(c_query))
                # print(query)
                # res_dict = db.explain(str(c_query), execute=False, timeout=6000000)
                # print(res_dict.result)
        self.get_bucket(sqls,buckets,cparam,operator)
        return buckets

    def change_filter_num(self, operator, cparam, where_clause, query, tables,buckets,namelist,column_repeat):
        namelist_verse = {namelist[t]:t for t in namelist}
        columns_candidate = []
        comparison_choices = {}
        tables = get_where_from_tables(where_clause['item'].parent,where_clause['item'])
        for table in tables:
            if table not in self.scheme_info.table_features.keys():
                if table in namelist.keys():
                    table = namelist[table]
                else:
                    continue
            if table not in self.scheme_info.table_features.keys():
                continue
            for col in self.comparisons[table].keys():
                if len(self.comparisons[table][col]):
                    if table not in comparison_choices.keys():
                        comparison_choices[table] = {}
                    columns_candidate.append([table,col])
                    comparison_choices[table][col] = self.comparisons[table][col]
        if not len(columns_candidate):
            return buckets
        column_num = len(columns_candidate) if not column_repeat else 2*len(columns_candidate)
        column_num = min(column_num,10)
        num_choices = {}
        for i in range(0,column_num+1):
            num_choices[int(i)] = [i]
        sqls = []
        get_join_conditions = [item['comparison'] for item in get_join_comparison(where_clause['item'],self.scheme_info)]
        c_query, c_where_clause = copy_item_and_parsed_sql(query, where_clause)
        #generate BUCKET_GENERATE_SAMPLES samples for every bucket
        for idx,item in enumerate(buckets):
            filter_number_choice = []
            for key in num_choices.keys():
                if (idx>0 and key>item['range'][0] and key<=item['range'][1]) or (idx==0 and key>=item['range'][0] and key<=item['range'][1]):
                    filter_number_choice += num_choices[key]
            if not(len(filter_number_choice)):
                continue
            choices = np.random.choice(filter_number_choice,BUCKET_GENERATE_SAMPLES)
            for choice in choices:
                # c_query, c_where_clause = copy_item_and_parsed_sql(query, where_clause)
                comparisons = get_join_conditions.copy()
                column_sample = np.random.choice(np.arange(len(columns_candidate)),choice,replace=column_repeat)
                for i in column_sample:
                    t = columns_candidate[i]
                    if t[0] in namelist_verse.keys():
                        comparisons.append(namelist_verse[t[0]]+"."+np.random.choice(comparison_choices[t[0]][t[1]],1)[0])
                    else:
                        comparisons.append(t[0]+"."+np.random.choice(comparison_choices[t[0]][t[1]], 1)[0])
                if not(len(comparisons)):
                    t_query, t_where_clause = copy_item_and_parsed_sql(query, where_clause)
                    delete_item_in_parsed_sql(t_query,t_where_clause)
                else:
                    gen_where = "where "+" and ".join(comparisons)+" "
                    gen_where = sqlparse.parse(gen_where)[0]
                    c_where_clause.tokens = gen_where.tokens
                if str(query) == str(c_query):
                    continue
                sqls.append(str(c_query))
                # print(query)
                try:
                    res_dict = self.db.explain(str(c_query), execute=False, timeout=6000000)
                    plan_tree = res_dict['Plan']
                except:
                    pass
        self.get_bucket(sqls,buckets,cparam,operator)
        return buckets
    
    def change_filter_offset(self, operator, cparam, where_clause, query, tables,buckets,namelist):
        def get_lessoffset_comparison_random(offset):
            keys_candidate = []
            for key in comparisn_choices.keys():
                if key<offset:
                    keys_candidate.append(key)
            if not len(keys_candidate):
                return []
            key = np.random.choice(keys_candidate,1)[0]
            col_id = np.random.choice(np.arange(len(comparisn_choices[key].keys())),1)[0]
            t = list(comparisn_choices[key].keys())[col_id]
            if t[0] in namelist_verse.keys():
                return [namelist_verse[t[0]]+"."+np.random.choice(comparisn_choices[key][t],1)[0]]
            else:
                return [t[0] + "." + np.random.choice(comparisn_choices[key][t], 1)[0]]

        namelist_verse = {namelist[t]: t for t in namelist}
        comparisn_choices = {}
        for table in tables:
            if table not in self.scheme_info.table_features.keys():
                if table in namelist:
                    table = namelist[table]
                else:
                    continue
            if table not in self.scheme_info.table_features.keys():
                continue
            for col in self.comparisons[table].keys():
                if len(self.comparisons[table][col]):
                    offset = self.scheme_info.get_column_info(col)['offset']
                    if offset not in comparisn_choices.keys():
                        comparisn_choices[offset] = {}
                    comparisn_choices[offset][(table,col)] = self.comparisons[table][col]
        sqls = []
        get_join_conditions = [item['comparison'] for item in get_join_comparison(where_clause['item'], self.scheme_info)]
        c_query, c_where_clause = copy_item_and_parsed_sql(query, where_clause)
        # generate BUCKET_GENERATE_SAMPLES samples for every bucket
        for idx, item in enumerate(buckets):
            offset_choice = []
            for key in comparisn_choices.keys():
                if (idx > 0 and key > item['range'][0] and key <= item['range'][1]) or (
                        idx == 0 and key >= item['range'][0] and key <= item['range'][1]):
                    offset_choice += [key]
            if not (len(offset_choice)):
                continue
            choices = np.random.choice(offset_choice, BUCKET_GENERATE_SAMPLES)
            for choice in choices:
                # c_query, c_where_clause = copy_item_and_parsed_sql(query, where_clause)
                comparisons = get_join_conditions.copy()
                column_sample = np.random.choice(np.arange(len(comparisn_choices[choice].keys())), 1)[0]
                t = list(comparisn_choices[choice].keys())[column_sample]
                if t[0] in namelist_verse.keys():
                    comparisons.append(namelist_verse[t[0]]+"."+np.random.choice(comparisn_choices[choice][t], 1)[0])
                else:
                    comparisons.append(t[0] + "." + np.random.choice(comparisn_choices[choice][t], 1)[0])
                for _ in range(np.random.randint(4)):
                    comparisons += get_lessoffset_comparison_random(choice)
                if not (len(comparisons)):
                    t_query, t_where_clause = copy_item_and_parsed_sql(query, where_clause)
                    delete_item_in_parsed_sql(t_query, t_where_clause)
                else:
                    gen_where = "where " + " and ".join(comparisons) + " "
                    gen_where = sqlparse.parse(gen_where)[0]
                    c_where_clause.tokens = gen_where.tokens
                if str(query) == str(c_query):
                    continue
                sqls.append(str(c_query))
                # print(query)
                # res_dict = db.explain(str(c_query), execute=False, timeout=6000000)
                # print(res_dict.result)
        self.get_bucket(sqls, buckets, cparam, operator)
        return buckets
    def condition_modify(self,operator,cparam,queries,buckets):
        if not len(queries):
            queries += self.randomsql_selector_tool.get_queries(operator, cparams=[cparam])
        sqls = [item['query'] for item in queries]

        def filter_attribute_modify(query):
            where_clauses = extract_where_part(query)
            # namelist = get_name_list(query)
            for where_clause in where_clauses:
                namelist = get_name_list(where_clause['item'].parent)
                # namelist = get_name_list(where_clause.parent)
                if not is_subselect(where_clause['item']):
                    tables = get_where_from_tables(where_clause['item'].parent,where_clause['item'])
                    if cparam in ['FilterIntegerRatio', 'FilterFloatRatio', 'FilterStrRatio']:
                        self.change_filter_ratio(operator,cparam,where_clause,query,tables,buckets,namelist)
                    elif cparam == "FilterNum":
                        self.change_filter_num(operator,cparam,where_clause,query,tables,buckets,namelist,column_repeat=False)
                    elif cparam == "FilterColumnNum":
                        self.change_filter_num(operator, cparam, where_clause, query, tables, buckets,namelist,column_repeat=True)
                    elif cparam == "FilterOffset":
                        self.change_filter_offset(operator, cparam, where_clause, query, tables, buckets,namelist)
        if 'Filter' in cparam:
            for query in sqls:
                query_t = sqlparse.parse(query)[0]
                filter_attribute_modify(query_t)

                # break
        return buckets
        
    def modify_query_cardinality(self,query,operator):
        def get_column_info(com):
            if 'select' in com.left.value or 'CAST' in com.left.value:
                return {}
            column_str = str(com.tokens[0])
            if '.' in com.left.value:
                try:
                    table, column = column_str.split('.')
                except:
                    pass
                if table in namelist.keys():
                    table = namelist[table]
                column_str = table + "." + column
            info = self.scheme_info.get_column_info(column_str)
            return info

        def get_random_com(parsed,where_clause=''):
            namelist = get_name_list(parsed)
            namelist_verse = {namelist[t]: t for t in namelist}
            if where_clause == '':
                tables = extract_tables(parsed)
            else:
                tables = get_where_from_tables(parsed,where_clause)
            if not len(tables):
                return ''
            t_tables = []
            for table in tables:
                if table in namelist.keys() and namelist[table] in self.comparisons.keys():
                    t_tables.append(namelist[table])
                elif table in self.comparisons.keys():
                    t_tables.append(table)
            if not len(t_tables):
                return ''
            table = np.random.choice(t_tables,1)[0]
            col_cand = []
            for col in self.comparisons[table].keys():
                if len(self.comparisons[table][col]):
                    col_cand.append(col)
            if not len(col_cand):
                return ''
            column = np.random.choice(col_cand,1)[0]
            com = np.random.choice(list(self.comparisons[table][column]))
            if table in namelist_verse.keys():
                com_token = sqlparse.parse(namelist_verse[table]+'.'+com)[0].tokens[0]
            else:
                com_token = sqlparse.parse(table+'.'+com)[0].tokens[0]
            return com_token


        ast = sqlparse.parse(query)[0]

        where_clauses = list(extract_where_part(ast))
        modify_flag = False
        if len(where_clauses):
            random.shuffle(where_clauses)
            for where_clause in where_clauses:
                comparisons = list(get_query_comparison_ast(where_clause['item']))
                namelist = get_name_list(where_clause['item'].parent)
                namelist_verse = {namelist[t]: t for t in namelist}
                if not len(comparisons):
                    com_token = get_random_com(where_clause['item'].parent,where_clause['item'])
                    if com_token != '':
                        where_clause['item'].tokens.insert(1, self.key_words[' '])
                        where_clause['item'].tokens.insert(2, com_token)
                        where_clause['item'].tokens.insert(3, self.key_words['\n'])
                        if '\t' in self.key_words.keys():
                            where_clause['item'].tokens.insert(4, self.key_words['\t'])
                        else:
                            where_clause['item'].tokens.insert(4, self.key_words[' '])
                        where_clause['item'].tokens.insert(5, self.key_words['and'])
                        where_clause['item'].tokens.insert(6, self.key_words[' '])
                        modify_flag = True
                        break
                    else:
                        continue
                choice_comparisons = []
                for com in comparisons:
                    try:
                        info = get_column_info(com)
                    except:
                        info = get_column_info(com)
                    tables = get_where_from_tables(where_clause['item'].parent,where_clause['item'])
                    if not len(info):
                        continue
                    if info['table'] in namelist_verse.keys():
                        if namelist_verse[info['table']] not in tables:
                            continue
                    if info['table'] not in namelist_verse.keys() and info['table'] not in tables:
                        continue
                    
                    if len(self.comparisons[info['table']][info['column']])>1:
                        choice_comparisons.append(com)
                if not len(choice_comparisons):
                    com_token = get_random_com(where_clause['item'].parent, where_clause['item'])
                    if com_token != '':
                        where_clause['item'].tokens.insert(1, self.key_words[' '])
                        where_clause['item'].tokens.insert(2, com_token)
                        where_clause['item'].tokens.insert(3, self.key_words['\n'])
                        if '\t' in self.key_words.keys():
                            where_clause['item'].tokens.insert(4, self.key_words['\t'])
                        else:
                            where_clause['item'].tokens.insert(4, self.key_words[' '])
                        where_clause['item'].tokens.insert(5, self.key_words['and'])
                        where_clause['item'].tokens.insert(6, self.key_words[' '])
                        modify_flag = True
                        break
                    else:
                        continue
                else:
                    com = choice_comparisons[np.random.choice(np.arange(len(choice_comparisons)), 1)[0]]
                    info = get_column_info(com)
                    if not len(info):
                        continue
                    table = ""

                    if len(com.left.value.split("."))>=2:
                        if info['table'] in namelist_verse.keys():
                            table = namelist_verse[info['table']]+"."
                        else:
                            table = com.left.value.split(".")[0]+"."
                    com_choice = set(self.comparisons[info['table']][info['column']]) - set([str(com)[len(table):]])
                    if not len(com_choice):
                        continue
                    com.tokens = sqlparse.parse(table+np.random.choice(list(com_choice), 1)[0])[0].tokens
                    modify_flag = True
        if not modify_flag:
            subqueries = list(extract_subselect(ast))
            for subquery in subqueries:
                tables = extract_tables(subquery['item'])
                if operator == "Index Only Scan":
                    info = self.parse([{'sql':query}])[0]
                    for item in info['node']:
                        if item['name'] == "Index Only Scan":
                            output = np.random.choice(item['original']['Output'], 1)[0]
                            if '.' in output:
                                choices = self.comparisons[item['original']['Relation Name']][re.split(r'\.', output)[1]]
                            else:
                                choices = self.comparisons[item['original']['Relation Name']][output]
                            if not choices:
                                continue
                            else:
                                choice = np.random.choice(choices, 1)[0]
                                where_clause = "where " + item['original']['Relation Name'] + "." + choice
                                break
                else:
                    com_token = get_random_com(subquery['item'])
                    if com_token == '':
                        continue
                    where_clause = "where " + str(com_token)

                id = -1
                for idx in np.arange(len(subquery['item'].tokens)-1,-1,-1):
                    if str(subquery['item'].tokens[idx]) == ';' or str(subquery['item'].tokens[idx]) in ['order by','group by','ORDER BY','GROUP BY','limit']:
                        id = idx
                if id != -1:
                    subquery['item'].tokens.insert(id,self.key_words[' '])
                    subquery['item'].tokens.insert(id+1,sqlparse.parse(where_clause)[0][0])
                    subquery['item'].tokens.insert(id+2, self.key_words[' '])
                    modify_flag = True
                    break
            # else:
            #     com_token = get_random_com(ast,where_clause)
            #     if com_token != '':
            #         for token in ast.tokens:
            #             if isinstance(token, sqlparse.sql.Where):
            #                 token.tokens.insert(1, self.key_words[' '])
            #                 token.tokens.insert(2, com_token)
            #                 token.tokens.insert(3, self.key_words['\n'])
            #                 if '\t' in self.key_words.keys():
            #                     token.tokens.insert(4, self.key_words['\t'])
            #                 else:
            #                     token.tokens.insert(4, self.key_words[' '])
            #                 token.tokens.insert(5, self.key_words['and'])
            #                 token.tokens.insert(6, self.key_words[' '])
            #                 modify_flag = True
        else:
            return str(ast)
        if modify_flag:
            return str(ast)

    def modify_query_comparison(self,query):
        # if "having" in query:
        #     print("here")
        ast = sqlparse.parse(query)[0]
        subqueries = list(extract_subselect(ast))
        subqueries.append({'item':ast,'location':[]})
        random.shuffle(subqueries)
        for subquery in subqueries:
            where_clauses = list(extract_where_part(subquery['item']))
            if not len(where_clauses):
                return ""
            where_clause = np.random.choice(where_clauses,1)[0]
            o_tables = get_where_from_tables(where_clause['item'].parent,where_clause['item'])
            namelist = get_name_list(where_clause['item'].parent)
            namelist_verse = {namelist[t]: t for t in namelist}
            tables = []
            for t in o_tables:
                if t not in self.comparisons.keys():
                    if t in namelist.keys() and namelist[t] in self.comparisons.keys():
                        tables.append(namelist[t])
                else:
                    tables.append(t)
            if not len(tables):
                return ""
            table = np.random.choice(tables,1)[0]
            cols = list(self.comparisons[table].keys())
            np.random.shuffle(cols)
            flag = False
            for col in cols:
                if len(self.comparisons[table][col]):
                    flag = True
                    if table in namelist_verse.keys():
                        comparison = sqlparse.parse(namelist_verse[table]+"."+np.random.choice(self.comparisons[table][col], 1)[0])[0]
                    else:
                        comparison = sqlparse.parse(table+"." + np.random.choice(self.comparisons[table][col], 1)[0])[0]
                    break
            if flag:
                where_clause['item'].tokens.insert(1, self.key_words[' '])
                where_clause['item'].tokens.insert(2, comparison)
                where_clause['item'].tokens.insert(3, self.key_words['\n'])
                if '\t' in self.key_words.keys():
                    where_clause['item'].tokens.insert(4, self.key_words['\t'])
                else:
                    where_clause['item'].tokens.insert(4, self.key_words[' '])
                where_clause['item'].tokens.insert(5, self.key_words['and'])
                where_clause['item'].tokens.insert(6, self.key_words[' '])
                return str(ast)
            else:
                return ''

    def parentop_modify(self,operator,cparam,queries,buckets):
        # buckets = {}
        # for op in ["Seq Scan", "Index Scan", "Index Only Scan",'Hash Join', 'Merge Join', 'Nested Loop','Sort', 'Aggregate','Hash','Increment Sort','CTE Scan','Materialize','Subquery Scan','Limit']:
        #     buckets[op] = set()
        temp_buckets = {t['range'][0]:set() for t in buckets}
        if not len(queries):
            queries += self.randomsql_selector_tool.get_queries(operator, cparams=[cparam])
        for query in queries:
            for item in query['node']:
                if item['name'] == operator:
                    temp_buckets[item[cparam]].add(str(query['ast']))
        results = []
        count = 0
        gen_buckets = copy.deepcopy(buckets)
        for item in gen_buckets:
            while len(item['queries']) < 20 and count < 100 and len(temp_buckets[item['range'][0]]):
                count += 1
                sql = np.random.choice(list(temp_buckets[item['range'][0]]), 1)[0]
                query = self.modify_query_cardinality(sql,operator)
                if query == '' or query in item['queries'] or query == None:
                    query = self.modify_query_comparison(sql)
                if query != '' and query not in item['queries'] and query != None:
                    results.append(query)
                    item['queries'].append(query)
                    try:
                        res_dict = self.db.explain(query,execute=False)
                        plan_tree = res_dict['Plan']
                    except:
                        query = self.modify_query_cardinality(sql,operator)

        self.get_bucket(results,buckets,cparam,operator)
        return buckets

    def relation_modify(self,operator,cparam,queries,buckets):
        queries = self.filter_queries(queries,operator)
        if not len(queries):
            queries += self.randomsql_selector_tool.get_queries(operator, cparams=[cparam])
        temp_buckets = copy.deepcopy(buckets)
        for idx,query in enumerate(queries):
            for item in query['node']:
                if item['name'] == operator:
                    # print(idx)
                    value = round(item[cparam],2)
                    for item in temp_buckets:
                        if value>item['range'][0] and value<=item['range'][1]:
                            item['queries'].append(query['query'])
        count = 0
        for idx,item in enumerate(buckets):
            while len(item['queries'])<20 and count<100:
                count += 1
                if not len(temp_buckets[idx]['queries']):
                    continue
                sql = np.random.choice(list(temp_buckets[idx]['queries']),1)[0]
                query = self.modify_query_cardinality(sql,operator)
                if query == '' or query in item['queries'] or query == None:
                    query = self.modify_query_comparison(sql)
                if query != '' or query in item['queries'] or query == None:
                    try:
                        res_dict = self.db.explain(query=query, execute=False,)
                        plan_tree = res_dict['Plan']
                    except:
                        print("here")
                        query = self.modify_query_cardinality(sql,operator)
                        continue
                    item['queries'].append({"query":query,"env":{}})
        return buckets

    def filter_queries(self,queries,operator):
        result = []
        for query in queries:
            for node in query['node']:
                if node['name'] == operator:
                    result.append(query)
                    break
        return result

    def set_envs(self,env_cmd):
        for cmd in env_cmd:
            self.db.execute(query=cmd,set_env=True)

    def join_modify(self, operator, cparam, queries, buckets):
        queries = self.filter_queries(queries,operator)
        if not len(queries):
            queries += self.randomsql_selector_tool.get_queries(operator, cparams=[cparam])
        temp_buckets = {}
        cparam_type = type(queries[0]['node'][0][cparam])
        for query in queries:
            for node in query['node']:
                if node['name'] == operator:
                    if cparam_type != str:
                        value = round(node[cparam],2)
                    else:
                        value = node[cparam]
                    if value not in temp_buckets.keys():
                        temp_buckets[value] = []
                    temp_buckets[value].append(query)
        # combine
        combine_buckets = copy.deepcopy(buckets)
        if cparam_type != str:
            for key in temp_buckets.keys():
                for idx,bucket in enumerate(combine_buckets):
                    if idx == 0 and key>=bucket['range'][0] and key<=bucket['range'][1]:
                        bucket['queries'] += temp_buckets[key]
                    if key>bucket['range'][0] and key<=bucket['range'][1]:
                        bucket['queries'] += temp_buckets[key]
        else:
            for bucket in combine_buckets:
                if bucket['range'][0] in temp_buckets.keys():
                    bucket['queries'] = temp_buckets[bucket['range'][0]]

        count = 0
        for idx,item in enumerate(buckets):
            if not len(combine_buckets[idx]['queries']):
                continue
            while len(item['queries']) < 20 and count < 100:
                count += 1
                sql = np.random.choice(list(combine_buckets[idx]['queries']), 1)[0]
                query = self.modify_query_cardinality(sql['query'],operator)
                if query == '' or query in item['queries'] or query == None:
                    query = self.modify_query_comparison(sql['query'])
                if query != '' or query in item['queries'] or query == None:
                    try:
                        res_dict = self.db.explain(query=query, execute=False,)
                        plan_tree = res_dict['Plan']
                    except:
                        print("here")
                        query = self.modify_query_cardinality(sql['query'], operator)
                    item['queries'].append({'query':query,'env':sql['env']})
        return buckets


    def modify_query_for_cparam(self,operator,cparam,buckets,queries,filter):
        # queries = self.workload_data
        # queries = self.parse(queries)
        queries = self.parse_with_envset(queries,operator)
        queries = self.filter_queries(queries,operator)
        if len(queries)>20:
            queries = np.random.choice(queries,20,replace=False)
        for query in queries:
            query["ast"] = sqlparse.parse(query['query'])[0]

        # print(f'{operator},{cparam}')
        if operator in ["Seq Scan","Index Scan","Index Only Scan"]:
            if cparam in ['FilterIntegerRatio', 'FilterFloatRatio', 'FilterStrRatio', 'FilterOffset','FilterColumnNum','FilterNum']:
                buckets = self.condition_modify(operator,cparam,queries,buckets)
            elif cparam in ['LeftSoutAvg','TablePages','TuplesPerBlock','LeftRows']:  #Change the table
                buckets = self.relation_modify(operator,cparam,queries,buckets)
            elif cparam in ['IndexCorrelation', 'IndexTreeHeight', 'IndexTreePages', 'IndexTreeUniqueValues']:
                buckets = self.relation_modify(operator, cparam, queries,buckets)
            elif cparam in ['CondNum', 'CondOffset', 'CondIntegerRatio', 'CondFloatRatio', 'CondStrRatio', 'CondColumnNum',]:
                buckets = self.relation_modify(operator, cparam,queries,buckets)
            elif cparam in ['ParentOp','Strategy']:
                buckets = self.parentop_modify(operator,cparam,queries,buckets)
            else:
                buckets = self.join_modify(operator, cparam, queries, buckets)
        else:  #if operator in ['Hash Join','Nested Loop','Merge Join']:
            # if cparam in ['BatchesNum','BucketsNum','']:
            buckets = self.join_modify(operator,cparam,queries,buckets)
        return [{'range':item['range'],'queries':list(item['queries'])} for item in buckets]

