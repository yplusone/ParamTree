import re
import queue
import traceback
import numpy as np

from database_util.inner_bucket_size import get_innerbucketsize
from feature.info import dbms_cparams

class Plan_class():

    def __init__(self, scheme_info):
        self.db_name = scheme_info.db_name
        self.bi_op = ['Hash Join', 'Merge Join', 'Nested Loop']
        self.nodetype = None
        self.config_info = scheme_info.config_info
        self.table_features = scheme_info.table_features
        self.index_features = scheme_info.index_features
        self.inner_bucket_size_info = scheme_info.inner_bucket_size_info

    def get_plan_info(self,plan_tree,sql, execute=True):
        result = []
        plans = queue.Queue()
        plans.put(plan_tree['Plan'])
        while not plans.empty():
            plan = plans.get()
            if plan['Node Type'] == 'Result':
                return []
            res = self.get_op_info(plan,execute=execute)
            res['query'] = sql
            res['original'] = plan
            result.append(res)
            if 'Plans' in plan.keys():
                for item in plan['Plans']:
                    item['parent'] = plan['Node Type']
                    plans.put(item)
        return result
    def get_feat(self, sql,sql_info = None):
        feats = []
        ans = {}
        # sql = 'EXPLAIN (ANALYZE, COSTS, BUFFERS, VERBOSE, FORMAT JSON) ' + sql
        if sql_info:
            for item in sql_info:
                if item['query'] == sql:
                    ans = item['planinfo']
        if not len(ans):
            try:
                ans = \
                    self.db.execute(sql)[0][
                        0][0]
            except:
                traceback.print_exc()
        self.plan_tree = ans
        results = self.get_plan_info(ans)

        return results,ans['Execution Time'] #feats

    def get_sort_info(self,text,res,type):
        outcolumnnum = 0
        calculatingnum = 0
        int_num, float_num, str_num = 0, 0, 0
        largest_offset = 0
        columnset = set()
        for outcolumn in text:
            if "(" in outcolumn:
                if "count(" not in outcolumn:
                    outcolumnnum += 1
                else:
                    calculatingnum += 1
                calculatingnum += len(re.split(r'[+*-/]', outcolumn))
            col = ''
            table = ''
            for tt_name in self.table_features.keys():
                for columns_name in self.table_features[tt_name]['columns'].keys():
                    if columns_name in outcolumn:
                        table = tt_name
                        col = columns_name
                        col_type = self.table_features[tt_name]['columns'][columns_name]['type']
            columnset.add(col)
            if table not in self.table_features.keys():
                str_num += 1
                continue
            if self.table_features[table]['columns'][col]['offset'] > largest_offset:
                largest_offset = self.table_features[table]['columns'][col]['offset']
            if col == '':
                col_type = 'bpchar'
            if col_type == 'int4' and 'numeric' not in outcolumn:
                int_num += 1
            elif 'numeric' in outcolumn or col_type == 'numeric':
                float_num += 1
            elif col_type == 'bpchar':
                if " ANY " in outcolumn:
                    str_num += len(re.findall(r'{(.*?)}', outcolumn)[0].split(','))
                else:
                    str_num += 1
            elif col_type == 'date':
                str_num += 1
            else:
                str_num += 1

        res[f'{type}Num'] = int_num + float_num + str_num + calculatingnum + outcolumnnum
        res[f'{type}Offset'] = largest_offset
        res[f'{type}IntegerRatio'] = int_num / (int_num + float_num + str_num)
        res[f'{type}FloatRatio'] = float_num / (int_num + float_num + str_num)
        res[f'{type}StrRatio'] = str_num / (int_num + float_num + str_num)
        res[f'{type}ColumnNum'] = len(columnset)
        return

    def get_filter_cond_info(self,plan,res):
        def get_key_info(filters,type):
            largest_offset = 0
            int_num, float_num, str_num = 0,0,0
            columnset = set()
            for filter in filters:
                t = re.split(r'\.', re.split(r'[<>!=)~]|(\sIS\s)', filter)[0].replace('(', '').strip())
                if len(t) < 2:
                    column = t[0]
                    table = ""
                else:
                    table, column = t

                column = column.replace("\"", "")
                if table not in self.table_features.keys():
                    for tt_name in self.table_features.keys():
                        for columns_name in self.table_features[tt_name]['columns'].keys():
                            if columns_name == column:
                                table = tt_name
                    if table not in self.table_features.keys():
                        for tt_name in self.table_features.keys():
                            if tt_name+"." in filter:
                                table = tt_name
                                break
                    # table = self.column_table_map[column.split('_')[0]]
                table = table.replace("\"", "")
                if table not in self.table_features.keys():
                    str_num += 1
                    continue
                if column not in self.table_features[table]['columns'].keys():
                    for column_name in self.table_features[table]['columns'].keys():
                        if column_name in filter:
                            column = column_name
                            break
                columnset.add(column)
                if table in self.table_features.keys():
                    if column in self.table_features[table]['columns'].keys():
                        if self.table_features[table]['columns'][column]['type'] == 'int4' and 'numeric' not in filter:
                            int_num += 1
                        elif 'numeric' in filter or self.table_features[table]['columns'][column]['type'] == 'numeric':
                            float_num += 1
                        elif self.table_features[table]['columns'][column]['type'] == 'bpchar':
                            if " ANY " in filter:
                                str_num += len(re.findall(r'{(.*?)}', filter)[0].split(','))
                            else:
                                str_num += 1
                        elif self.table_features[table]['columns'][column]['type'] == 'date':
                            str_num += 1
                        else:
                            str_num += 1
                        if self.table_features[table]['columns'][column]['offset'] > largest_offset:
                            largest_offset = self.table_features[table]['columns'][column]['offset']
                    else:
                        str_num += 1
                else:
                    str_num += 1

            res[f'{type}Num'] = int_num + float_num + str_num
            res[f'{type}Offset'] = largest_offset
            res[f'{type}IntegerRatio'] = int_num/(int_num + float_num + str_num)
            res[f'{type}FloatRatio'] = float_num/(int_num + float_num + str_num)
            res[f'{type}StrRatio'] = str_num/(int_num + float_num + str_num)
            res[f'{type}ColumnNum'] = len(columnset)

        def get_filter_info(info,type):
            if isinstance(info,str):
                filters = []
                ans = re.split('\sAND\s', info)
                for item in ans:
                    t = re.split('\sOR\s', item)
                    filters += t
            elif isinstance(info,list):
                filters = info
            get_key_info(filters,type)

        for type in ['Filter','Cond']:
            res[f'{type}Num'] = 0
            res[f'{type}Offset'] = 0
            res[f'{type}IntegerRatio'] = 0
            res[f'{type}FloatRatio'] = 0
            res[f'{type}StrRatio'] = 0
            res[f'{type}ColumnNum'] = 0
        if 'Filter' in plan.keys():
            if plan['Node Type'] == "Aggregate":
                self.get_sort_info(plan['Filter'],res,type = "Filter")
            else:
                get_filter_info(plan['Filter'], type="Filter")
        if 'Index Cond' in plan.keys():
            get_filter_info(plan['Index Cond'],type = "Cond")
        elif 'Index Cond' not in plan.keys() and plan['Node Type'] in ['Index Scan','Index Only Scan']:
            index_info = self.index_features[plan['Index Name']]
            fake_filters = [index_info['table']+"."+col+">1" for col in index_info['columns']]
            get_filter_info(fake_filters,type = "Cond")
        elif 'Hash Cond' in plan.keys():
            get_filter_info(plan['Hash Cond'],type = "Cond")
        elif 'Merge Cond' in plan.keys():
            get_filter_info(plan['Merge Cond'],type = "Cond")
        elif plan['Node Type'] == "Sort":
            self.get_sort_info(plan['Sort Key'],res,type = "Cond")
        elif plan['Node Type'] == "Aggregate":
            if 'Group Key' in plan.keys():
                self.get_sort_info(plan['Group Key'],res,type="Cond")
        return

    def get_index_info(self,plan,res):
        if plan['Node Type'] in ['Index Scan','Index Only Scan']:
            index_name = plan['Index Name']
            res['IndexCorrelation'] = self.index_features[index_name]['indexCorrelation']
            res['IndexTreeHeight'] = self.index_features[index_name]['tree_height']
            res['IndexTreePages'] = self.index_features[index_name]['pages']
            res['IndexTreeUniqueValues'] = self.index_features[index_name]['distinctnum']
        else:
            res['IndexCorrelation'] = 0
            res['IndexTreeHeight'] = 0
            res['IndexTreePages'] = 0
            res['IndexTreeUniqueValues'] = 0
    def get_strategy_info(self,plan,res):
        if 'Join Type' in plan.keys():
            res['Strategy'] = plan['Join Type']
        elif 'Sort Method' in plan.keys():
            res['Strategy'] = plan['Sort Method']
        elif 'Strategy' in plan.keys():
            res['Strategy'] = plan['Strategy']
        elif 'Scan Direction' in plan.keys():
            res['Strategy'] = plan['Scan Direction']
        else:
            res['Strategy'] = 'none'

    def get_child_info(self,plan,res,child,execute):
        if plan == None:
            res[f"{child}SoutAvg"] = 0
            res[f"{child}Rows"] = 0
            res[f"{child}Loops"] = 1
        else:
            res[f"{child}SoutAvg"] = plan['Plan Width']
            res[f"{child}Rows"] = plan['Actual Rows'] if execute else plan['Plan Rows']
            res[f"{child}Loops"] = plan['Actual Loops'] if execute else 0

    def get_general_info(self,plan,res,execute):
        res["Startup Cost"] = plan['Startup Cost']
        res["Total Cost"] = plan['Total Cost']
        res["Actual Startup Time"] = plan['Actual Startup Time'] if execute else 0
        res['Actual Total Time'] = plan['Actual Total Time'] if execute else 0
        if 'I/O Read Time' in plan.keys():
            res['I/O Read Time'] = plan['I/O Read Time']
        else:
            res['I/O Read Time'] = 0
        child_plans = []
        subplan_startup = 0
        subplan_total = 0
        if 'Plans' in plan.keys():
            for t in plan['Plans']:
                if 'Subplan Name' in t.keys():
                    if t['Parent Relationship'] in ['SubPlan']:
                        subplan_startup += t['Actual Total Time']*1 if execute else 0
                        subplan_total += t['Actual Total Time']*t['Actual Loops'] if execute else 0
                else:
                    child_plans.append(t)
        res['SubPlan Cost Time'] = subplan_total
        res['SubPlan Startup Time'] = subplan_startup
        if 'InitPlan' in plan.keys() and len(plan['InitPlan'])>0:
            cost = 0
            for t in plan['InitPlan']:
                cost += t['Actual Total Time'] if execute else 0
            res['InitPlan Cost Time'] = cost
        else:
            res['InitPlan Cost Time'] = 0
        if len(child_plans) >= 1:
            res['Left Startup Time'] = child_plans[0]['Actual Startup Time'] if execute else 0
            res['Left Total Time'] = child_plans[0]['Actual Total Time'] if execute else 0
            res['Left Startup Cost'] = child_plans[0]['Startup Cost']
            res['Left Total Cost'] = child_plans[0]['Total Cost']
            if len(child_plans) == 2:
                res['Right Startup Time'] = child_plans[1]['Actual Startup Time'] if execute else 0
                res['Right Total Time'] = child_plans[1]['Actual Total Time'] if execute else 0
                res['Right Startup Cost'] = child_plans[1]['Startup Cost']
                res['Right Total Cost'] = child_plans[1]['Total Cost']
            # elif len(child_plans)>2:
            #     print("here")

        if execute and plan['Actual Total Time'] == 0 :
            res['never_executed'] = True
        else:
            res['never_executed'] = False

    def get_hash_innerbucketsize(self,plan,res,execute):
        for cp in re.split('AND', plan['Hash Cond']):
            if len(re.split(r'[<>!=)]', cp)) > 2:
                tt = []
                for item in re.split(r'[<>!=)]', cp):
                    if item.isspace() or "::" in item:
                        continue
                    tt.append(item)
            try:
                table2, column2 = re.split(r'\.', tt[1].replace('(', '').strip())
            except:
                res['inner_bucket_size'] = 0.1
            table2 = table2.replace("\"", "")
            column2 = column2.replace("\"", "")
            if table2 not in self.table_features.keys():
                for tt_name in self.table_features.keys():
                    for columns_name in self.table_features[tt_name]['columns'].keys():
                        if columns_name == column2:
                            table2 = tt_name

            if table2 in self.table_features.keys():
                if execute:
                    res['inner_bucket_size'] = get_innerbucketsize(self.inner_bucket_size_info,table2, column2, plan['Inner Unique'],
                                                               res['BatchesNum'], res['BucketsNum'],
                                                               plan['Plans'][1]['Actual Rows'],
                                                               plan['Plans'][1]['Plans'][0]['Actual Rows'],self.db_name)
                else:
                    res['inner_bucket_size'] = get_innerbucketsize(self.inner_bucket_size_info,table2, column2, plan['Inner Unique'],
                                                                   res['BatchesNum'], res['BucketsNum'],
                                                                   plan['Plans'][1]['Plan Rows'],
                                                                   plan['Plans'][1]['Plans'][0]['Plan Rows'],self.db_name)
            else:
                res['inner_bucket_size'] = 0.1
    def get_split_filters(self, str):
        res = []
        ans = re.split('\sAND\s', str)
        for item in ans:
            t = re.split('\sOR\s', item)
            res += t
        return res
    def get_aggregate_info(self,plan,res):
        outcolumnnum = 0
        calculatingnum = 0
        for outcolumn in plan['Output']:
            if "(" in outcolumn:
                if "count(" not in outcolumn:
                    outcolumnnum += 1
                else:
                    calculatingnum += 1
                calculatingnum += len(re.split(r'[+*-/]', outcolumn))
                calculatingnum += len(re.split(r'[*/]', outcolumn))
        if 'Filter' in plan.keys():
            for outcolumn in self.get_split_filters(plan['Filter']):
                if "(" in outcolumn:
                    if "count(" not in outcolumn:
                        outcolumnnum += 1
                    else:
                        calculatingnum += 1
                    calculatingnum += len(re.split(r'[+*-/]', outcolumn))

        res['OutAggColumnNum'] = outcolumnnum
        res['CalculatingNum'] = calculatingnum

        if plan['Strategy'] == "Hashed":
            if "HashAgg Batches" in plan.keys():
                res['BatchesNum'] = plan["HashAgg Batches"]
            if "Planned Partitions" in plan.keys():
                res['partitions'] = plan["Planned Partitions"]
        if 'Rows Removed by Filter' in plan.keys():
            res['Rows'] += plan['Rows Removed by Filter']

    def get_op_info(self, plan,execute=True):
        # plan['Actual Rows'] = plan['Plan Rows']#plan['dd_est_card']
        res = {}
        res['name'] = plan['Node Type']
        self.get_general_info(plan, res,execute)
        self.get_filter_cond_info(plan,res)
        self.get_index_info(plan,res)
        self.get_strategy_info(plan,res)
        res['ParentOp'] = plan['parent']
        self.get_child_info(plan,res,child="",execute=execute)
        child_plans = []
        if 'Plans' in plan.keys():
            for t in plan['Plans']:
                if 'Subplan Name' in t.keys():
                    continue
                else:
                    child_plans.append(t)

        if 'Plans' in plan.keys() and len(child_plans)>=2:  # join
            res['LeftOp'] = child_plans[0]['Node Type']
            res['RightOp'] = child_plans[1]['Node Type']
            self.get_child_info(child_plans[0],res,child="Left",execute=execute)
            self.get_child_info(child_plans[1], res, child="Right",execute=execute)
        elif len(child_plans) == 1:
            res['LeftOp'] = child_plans[0]['Node Type']
            res['RightOp'] = "none"
            self.get_child_info(child_plans[0],res,child="Left",execute=execute)
            self.get_child_info(None, res, child="Right",execute=execute)
        else:
            res['LeftOp'] = "none"
            res['RightOp'] = "none"
            self.get_child_info(None, res, child="Left",execute=execute)
            self.get_child_info(None, res, child="Right",execute=execute)
        if 'Inner Unique' in plan.keys():
            res['InnerUnique'] = 1 if plan['Inner Unique']==True else 0
        else:
            res['InnerUnique'] = 0
        # special features
        if plan['Node Type'] in ['Seq Scan','Index Scan','Index Only Scan']:
            table_name = plan['Relation Name']
            res['LeftSoutAvg'] = self.table_features[table_name]['table_size'] / self.table_features[table_name]['tuple_num']
            res['LeftRows'] = self.table_features[table_name]['tuple_num']
            res['TablePages'] = self.table_features[table_name]['table_pages']
            res['TuplesPerBlock'] = self.table_features[table_name]['tuple_num'] / self.table_features[table_name]['table_pages']
            rows_rm = 0
            if plan['Node Type'] in ['Index Scan','Index Only Scan']:
                if 'Rows Removed by Filter' in plan.keys():
                    rows_rm += plan['Rows Removed by Filter']
                if 'Rows Removed by Index Recheck' in plan.keys():
                    rows_rm += plan['Rows Removed by Index Recheck']
            if execute:
                res['Selectivity'] = (plan['Actual Rows']+rows_rm)/self.table_features[table_name]['tuple_num']
            else:
                res['Selectivity'] = (plan['Plan Rows'] + rows_rm) / self.table_features[table_name]['tuple_num']
            # tpcds temlate.9不适用
            # if execute and plan['Node Type']=='Seq Scan' and abs(plan['Shared Read Blocks']+plan['Shared Hit Blocks'] - res['TablePages']) != 0 and plan['Relation Name'] in self.table_features.keys():
            #     res['never_executed'] = True
        else:
            res['TablePages'] = 0
            res['TuplesPerBlock'] = 0
            res['Selectivity'] = 0

        res['BucketsNum'] = 1
        res['BatchesNum'] = 1
        if plan['Node Type'] == "Hash Join":
            if 'Hash Buckets' not in plan['Plans'][1].keys() or 'Hash Batches' not in plan['Plans'][1].keys():
                res['BucketsNum'] = 1
                res['BatchesNum'] = 1
            else:
                res['BucketsNum'] = plan['Plans'][1]['Hash Buckets']
                res['BatchesNum'] = plan['Plans'][1]['Hash Batches']
            res['Actual Startup Time'] = plan['Plans'][1]['Actual Startup Time'] if execute else 0 #max(plan['Plans'][0]['Actual Startup Time'],plan['Plans'][1]['Actual Total Time'])
            res['Right Startup Time'] = plan['Plans'][1]['Plans'][0]['Actual Startup Time']  if execute else 0
            res['Right Total Time'] = plan['Plans'][1]['Plans'][0]['Actual Total Time'] if execute else 0
            # if res['Actual Startup Time']-res['Right Total Time']-res['Left Startup Time']<-10:
            #     print("here")
            # if res['Rows']<1:
            #     print("here")
            self.get_hash_innerbucketsize(plan,res,execute)
        if plan['Node Type'] == "Nested Loop" and res['RightOp'] == 'Materialize':
            res['Right Startup Time'] = plan['Plans'][1]['Plans'][0]['Actual Total Time'] if execute else 0
        res.update(self.config_info)
        if plan['Node Type'] == "Aggregate":
            self.get_aggregate_info(plan,res)
        return res