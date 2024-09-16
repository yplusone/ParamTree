import sqlparse
from sqlparse.sql import Where,Comparison,IdentifierList,Identifier
from sqlparse.tokens import Keyword,DML
import queue
import re

def is_subselect(parsed):
    """
    是否子查询
    :param parsed: T.Token
    """
    if not parsed.is_group:
        return False
    for item in parsed.tokens:
        if not item.is_keyword and ('select' in str(item) or 'SELECT' in str(item)) :
            return True
    return False

def extract_subselect(parsed,location = []):
    for idx,item in enumerate(parsed.tokens):
        if not item.is_keyword and ('select' in str(item) or 'SELECT' in str(item)):
            if isinstance(item,sqlparse.sql.Parenthesis):
                yield {'item':item,'location':location+[idx]}
            else:
                for x in extract_subselect(item,location+[idx]):
                    yield x

def extract_from_part(parsed):
    """
    提取from之后模块
    """
    from_seen = False
    for item in parsed.tokens:
        if from_seen:
            if is_subselect(item):
                for x in extract_from_part(item):
                    yield x
            elif item.ttype is Keyword:
                from_seen = False
                continue
            else:
                yield item
        elif item.ttype is Keyword and item.value.upper() == 'FROM':
            from_seen = True

def extract_from_table_part(parsed):
    """
    提取from之后table的模块
    """
    from_seen = False
    for item in parsed.tokens:
        if from_seen:
            if is_subselect(item):
                for x in extract_from_table_part(item):
                    yield x
            elif isinstance(item, sqlparse.sql.Parenthesis):
                for x in extract_from_table_part(item):
                    yield x
            elif item.ttype is Keyword and item.value.upper() not in ['JOIN','ON','LEFT OUTER JOIN']:
                from_seen = False
                continue
            if type(item) is IdentifierList:
                yield item
            elif type(item) is Identifier:
                yield item
        elif isinstance(item, sqlparse.sql.Parenthesis):
            for x in extract_from_table_part(item):
                yield x
        elif item.ttype is Keyword and item.value.upper() == 'FROM':
            from_seen = True

def get_name_list(parsed):
    res = extract_from_table_part(parsed)
    namelist = {}
    for item in res:
        if type(item) is IdentifierList:
            for token in item.tokens:
                if type(token) is Identifier:
                    if 'select' in str(token) or 'SELECT' in str(token):
                        continue
                    if " AS " in token.value:
                        table,alias = token.value.split("AS")
                        namelist[alias.strip()] = table.strip()
                    elif " as " in token.value:
                        table,alias = token.value.split("as")
                        namelist[alias.strip()] = table.strip()
                    elif len(str(token).split(" "))>=2 and ('select' not in str(token) and 'SELECT' not in str(token)):
                        # print(str(token))
                        table,alias = re.split(r"[ ]+", str(token))
                        namelist[alias.strip()] = table.strip()
        elif type(item) is Identifier and not item.is_keyword:
            if 'select' in str(item) or 'SELECT' in str(item):
                continue
            if " AS " in item.value:
                table,alias = item.value.split("AS")
                namelist[alias.strip()] = table.strip()
            elif " as " in item.value:
                table,alias = item.value.split("as")
                namelist[alias.strip()] = table.strip()
            elif len(str(item).split(" "))>=2 and ('select' not in str(item) and 'SELECT' not in str(item)):
                # print(str(token))
                table,alias = re.split(r"[ ]+", str(item))
                namelist[alias.strip()] = table.strip()
    return namelist

def extract_where_part(parsed,location = []):
    for idx,item in enumerate(parsed.tokens):
        if isinstance(item, sqlparse.sql.Where):
            yield {'item':item,'location':location+[idx]}
        
    for state in extract_subselect(parsed,location):
        for x in extract_where_part(state['item'],state['location']):
            yield x

def extract_table_identifiers(token_stream):
    for item in token_stream:
        if isinstance(item, IdentifierList):
            for identifier in item.get_identifiers():
                yield identifier.get_name()
        elif isinstance(item, Identifier) and not item.is_keyword:
            if 'select' in str(item) or 'SELECT' in str(item):
                continue
            yield item.get_name()
        # elif item.ttype is Keyword:
        #     yield item.value


def extract_tables(parsed):
    from_stream = extract_from_part(parsed)
    # join_stream = extract_join_part(sqlparse.parse(sql)[0])
    return list(extract_table_identifiers(from_stream))

def copy_item_and_parsed_sql(parsed,item):
    parsed_copy = sqlparse.parse(str(parsed))[0]
    token = parsed_copy
    for idx in item['location']:
        token = token.tokens[idx]
    return parsed_copy,token
    # parsed_copy = sqlparse.parse(str(parsed))[0]
    # s = queue.Queue()
    # copy_s = queue.Queue()
    # s.put(parsed)
    # copy_s.put(parsed_copy)
    # while not s.empty():
    #     n_token = s.get()
    #     copy_token = copy_s.get()
    #     if str(n_token) == str(item):
    #         return parsed_copy,copy_token
    #     if hasattr(n_token,'tokens'):
    #         for idx,token in enumerate(n_token.tokens):
    #             s.put(token)
    #             copy_s.put(copy_token.tokens[idx])
    # return parsed_copy,None

def delete_item_in_parsed_sql(parsed,item):
    s = queue.Queue()
    s.put(parsed)
    while not s.empty():
        n_token = s.get()
        if hasattr(n_token,'tokens'):
            for idx,token in enumerate(n_token.tokens):
                if str(token) == str(item):
                    del n_token.tokens[idx]
                    return
                s.put(token)
    raise Exception("The delete item is not found")

def get_ratios_filter(where_clause,scheme_info):
    num = {"int": 0, "float": 0, "str": 0}
    for info in get_query_comparison(where_clause,scheme_info):
        if len(info):
            num[info['mtype']] += 1
        else:
            num['str'] += 1


    all = num['int'] + num['float'] + num['str']
    if all == 0:
        return 0,0,0
    return num['int'] / all, num['float'] / all, num['str'] / all

def get_query_comparison(parsed,scheme_info,namelist):

    for token in parsed.tokens:
        if isinstance(token,sqlparse.sql.Comparison) and not isinstance(token.right,sqlparse.sql.Identifier) and not token.right.is_group:
            if isinstance(token.left,sqlparse.sql.Operation):
                continue
            flag = False
            for t in token.tokens:
                if isinstance(t,sqlparse.sql.Parenthesis):
                    flag = True
                    for x in get_query_comparison(t,scheme_info,namelist):
                        yield x
            if not flag:
                column_str = token.left.value
                comp = str(token)
                if '.' in token.left.value:
                    table,column = token.left.value.split('.')
                    comp = str(token).replace(table+'.','')
                    if table in namelist.keys():
                        table = namelist[table]
                    column_str = table+"."+column
                info = scheme_info.get_column_info(column_str)

                if len(info):
                    res = {"table":info['table'],
                            "column":info['column'],
                        "type":info['type'],
                        "mtype":info['mtype'],
                            "comparison":comp}
                    yield res
        elif isinstance(token,sqlparse.sql.Parenthesis):
            for x in get_query_comparison(token,scheme_info,namelist):
                yield x
        elif isinstance(token,sqlparse.sql.Where):
            for x in get_query_comparison(token,scheme_info,namelist):
                yield x

def get_query_comparison_ast(parsed):

    for token in parsed.tokens:
        if isinstance(token,sqlparse.sql.Comparison) and not isinstance(token.right,sqlparse.sql.Identifier):
            if 'sum(' in str(token) or 'avg(' in str(token) or 'min(' in str(token) or 'max(' in str(token):
                continue
            yield token
            for t in token.tokens:
                if isinstance(t,sqlparse.sql.Parenthesis):
                    for x in get_query_comparison_ast(t):
                        yield x
        elif isinstance(token,sqlparse.sql.Parenthesis):
            for x in get_query_comparison_ast(token):
                yield x
        elif isinstance(token,sqlparse.sql.Where):
            for x in get_query_comparison_ast(token):
                yield x
        elif isinstance(token,sqlparse.sql.Identifier):
            for x in get_query_comparison_ast(token):
                yield x

def get_query_all_comparison_ast(parsed):

    for token in parsed.tokens:
        if isinstance(token,sqlparse.sql.Comparison):
            if 'sum(' in str(token) or 'avg(' in str(token) or 'min(' in str(token) or 'max(' in str(token):
                continue
            yield token
        elif isinstance(token,sqlparse.sql.Parenthesis):
            for x in get_query_all_comparison_ast(token):
                yield x
        elif isinstance(token,sqlparse.sql.Where):
            for x in get_query_all_comparison_ast(token):
                yield x
        elif isinstance(token,sqlparse.sql.Identifier):
            for x in get_query_all_comparison_ast(token):
                yield x

def get_join_comparison(parsed,scheme_info):

    for token in parsed.tokens:
        if isinstance(token,sqlparse.sql.Comparison) and isinstance(token.right,sqlparse.sql.Identifier):
            info = scheme_info.get_column_info(token.left.value)
            if len(info):
                res = {"table":info['table'],
                        "column":info['column'],
                       "type":info['type'],
                       "mtype":info['mtype'],
                        "comparison":str(token)}
                yield res
        elif token.is_keyword and token.normalized == 'INTERSECT':
            break
        elif isinstance(token,sqlparse.sql.Parenthesis):
            for x in get_join_comparison(token,scheme_info):
                yield x
        elif isinstance(token,sqlparse.sql.Where):
            for x in get_join_comparison(token,scheme_info):
                yield x

def get_keyword_obj(parsed):
    for token in parsed.tokens:
        if token.is_keyword or token.is_whitespace:
            yield token
        elif isinstance(token,sqlparse.sql.Parenthesis):
            for x in get_keyword_obj(token):
                yield x
        elif isinstance(token,sqlparse.sql.Where):
            for x in get_keyword_obj(token):
                yield x
        elif isinstance(token,sqlparse.sql.Identifier):
            for x in get_keyword_obj(token):
                yield x

def get_where_from_tables(ast,where_clause):
    s = []
    find_where = False
    for token in ast.tokens:
        if token != where_clause:
            s.append(token)
        else:
            find_where = True
            break
    if not find_where:
        return []
    flag = False
    a = []
    tables = []
    while len(s):
        t = s.pop()
        if (str(t)).upper() == 'FROM':
            tables = list(extract_table_identifiers(a))
            break
        else:
            a.append(t)
    return tables
        

if __name__=="__main__":
    sql = "EXPLAIN (ANALYZE, COSTS, BUFFERS, VERBOSE, FORMAT JSON)  WITH all_sales AS (  SELECT d_year        ,i_brand_id        ,i_class_id        ,i_category_id        ,i_manufact_id        ,SUM(sales_cnt) AS sales_cnt        ,SUM(sales_amt) AS sales_amt  FROM (SELECT d_year              ,i_brand_id              ,i_class_id              ,i_category_id              ,i_manufact_id              ,cs_quantity - COALESCE(cr_return_quantity,0) AS sales_cnt              ,cs_ext_sales_price - COALESCE(cr_return_amount,0.0) AS sales_amt        FROM catalog_sales JOIN item ON i_item_sk=cs_item_sk                           JOIN date_dim ON d_date_sk=cs_sold_date_sk                           LEFT JOIN catalog_returns ON (cs_order_number=cr_order_number                                                      AND cs_item_sk=cr_item_sk)        WHERE i_category='Electronics'        UNION        SELECT d_year              ,i_brand_id              ,i_class_id              ,i_category_id              ,i_manufact_id              ,ss_quantity - COALESCE(sr_return_quantity,0) AS sales_cnt              ,ss_ext_sales_price - COALESCE(sr_return_amt,0.0) AS sales_amt        FROM store_sales JOIN item ON i_item_sk=ss_item_sk                         JOIN date_dim ON d_date_sk=ss_sold_date_sk                         LEFT JOIN store_returns ON (ss_ticket_number=sr_ticket_number                                                  AND ss_item_sk=sr_item_sk)        WHERE i_category='Electronics'        UNION        SELECT d_year              ,i_brand_id              ,i_class_id              ,i_category_id              ,i_manufact_id              ,ws_quantity - COALESCE(wr_return_quantity,0) AS sales_cnt              ,ws_ext_sales_price - COALESCE(wr_return_amt,0.0) AS sales_amt        FROM web_sales JOIN item ON i_item_sk=ws_item_sk                       JOIN date_dim ON d_date_sk=ws_sold_date_sk                       LEFT JOIN web_returns ON (ws_order_number=wr_order_number                                              AND ws_item_sk=wr_item_sk)        WHERE i_category='Electronics') sales_detail  GROUP BY d_year, i_brand_id, i_class_id, i_category_id, i_manufact_id)  SELECT  prev_yr.d_year AS prev_year                           ,curr_yr.d_year AS year                           ,curr_yr.i_brand_id                           ,curr_yr.i_class_id                           ,curr_yr.i_category_id                           ,curr_yr.i_manufact_id                           ,prev_yr.sales_cnt AS prev_yr_cnt                           ,curr_yr.sales_cnt AS curr_yr_cnt                           ,curr_yr.sales_cnt-prev_yr.sales_cnt AS sales_cnt_diff                           ,curr_yr.sales_amt-prev_yr.sales_amt AS sales_amt_diff  FROM all_sales curr_yr, all_sales prev_yr  WHERE curr_yr.i_brand_id=prev_yr.i_brand_id    AND curr_yr.i_class_id=prev_yr.i_class_id    AND curr_yr.i_category_id=prev_yr.i_category_id    AND curr_yr.i_manufact_id=prev_yr.i_manufact_id    AND curr_yr.d_year=2001    AND prev_yr.d_year=2001-1    AND CAST(curr_yr.sales_cnt AS DECIMAL(17,2))/CAST(prev_yr.sales_cnt AS DECIMAL(17,2))<0.9  ORDER BY sales_cnt_diff,sales_amt_diff  limit 100; "
    ast = sqlparse.parse(sql)[0]
    # a = list(get_query_comparison_ast(ast))
    # print(a)
    # namelist = get_name_list(ast)
    # a = list(get_query_comparison_ast(ast,namelist))
    where_clauses = list(extract_where_part(ast))
    # print(where_clauses)
    # for where_clause in where_clauses:
    #     tablse = get_where_from_tables(where_clause.parent,where_clause)
    select_clause = list(extract_subselect(ast))