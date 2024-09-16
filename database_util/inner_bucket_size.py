# coding=UTF-8<code>

import psycopg2
from pyparsing import col
from .db_connector import *
from feature.infos import schema_db_info as db_info
#求inner_bucket_size
inner_bucket_size=0
DEFAULT_NUM_DISTINCT = 200 #default number of distinct values in a table

#辅助函数
def clamp_row_est(nrows):
	if (nrows <= 1.0):
		nrows = 1.0
	else:
		nrows = int(nrows)
	return nrows


def get_pg_statistic(tbl_name,col_name,db):
    """
        Get statistis from pg_class/pg_attribute/pg_statistic.
    """
    reltuples=0
    mcv_freq=0
    stanullfrac=0
    stadistinct=0
    #pg_class:select relname,oid,reltuples from pg_class where relname='link_type';
    tbl_result=db.execute('select relname,oid,reltuples from pg_class where relname=\''+tbl_name+'\';')[0]
    if not tbl_result: # 如果无结果返回None
        print("No tbl name.")
        reltuples=-1
        return (reltuples,mcv_freq,stanullfrac,stadistinct)

    oid=tbl_result[1]
    reltuples=tbl_result[2]
    #pg_attribute:select attrelid,attname,attnum from pg_attribute where attname='id' and attrelid=16613;
    tbl_result=db.execute('select attrelid,attname,attnum from pg_attribute where attname=\''+col_name+'\' and attrelid='+str(oid)+';')[0]
    if not tbl_result:
        print("No col name.")
        reltuples=-1
        return (reltuples,mcv_freq,stanullfrac,stadistinct)

    attnum=tbl_result[2]
    #pg_statistics-select * from pg_statistic where starelid=16613 and staattnum=1;
    att_result=db.execute('select * from pg_statistic where starelid='+str(oid)+' and staattnum='+str(attnum)+';') #【可能没有结果】
    if len(att_result)>0:
        results=att_result[0]
        stanullfrac=results[3]
        stadistinct=results[5]
        #sslots，目前只需要mcv[0]
        sta_array=[]
        for i in range(5):#0-4，
            sta_array.append([results[6+5*j+i] for j in range(5)]) #stakind1,staop1,stacoll,stanumbers1[1],stavalues1
            #stakind=1,STATISTIC_KIND_MCV-在列中出现最频繁的值按频率值进行排序
            if(sta_array[i][0]==1):
                mcv_array=sta_array[i][3] #{0.6,0.2}
                mcv_freq=mcv_array[0]#first MCV freq
    return (reltuples,mcv_freq,stanullfrac,stadistinct)


def get_variable_numdistinct(tuples,reltuples,mcv_freq,stanullfrac,stadistinct,isUnique):
    """
    1.检查pg_statistic中是否相应值 2.如果没有（即=0，Unkown),判断是否为unique
    """
    isdefault=False
    if(stadistinct == 0):
        if (isUnique):
            stadistinct = -1.0 * (1.0 - stanullfrac)

    if stadistinct>0.0:
        return isdefault,clamp_row_est(stadistinct)

    #No tbl_name
    if(reltuples==-1):
        isdefault=True
        ndistinct= DEFAULT_NUM_DISTINCT
        return isdefault,ndistinct

    if(tuples <= 0.0):
        isdefault=True
        ndistinct= DEFAULT_NUM_DISTINCT
        return isdefault,ndistinct

    if stadistinct <0.0:
        #ntuples = vardata->rel->tuples;
        ndistinct=clamp_row_est(-stadistinct*tuples)
        return isdefault,ndistinct

    if(tuples<DEFAULT_NUM_DISTINCT):
        ndistinct=clamp_row_est(tuples)
        return isdefault,ndistinct

    isdefault=True
    ndistinct= DEFAULT_NUM_DISTINCT
    return isdefault,ndistinct

def get_innerbucketsize(tbl_name,col_name,isUnique,num_batches,num_buckets,rows,tuples,db_name):
    db_info['pg']['db_name'] = db_name
    db = Postgres_Connector(server=db_info['server'], pg = db_info['pg'], ssh = db_info['ssh'])
    nbuckets=num_batches*num_buckets
    if isUnique:
        return 1/nbuckets
    #[start]模拟estimate_hash_bucket_stats逻辑
    #[1]得到mcv_freq
    (reltuples,mcv_freq,stanullfrac,stadistinct)=get_pg_statistic(tbl_name,col_name,db)
    #TODO  其实应该是reltuples 基表的元组数量，但是=0的那些又咋办？但是还是有对不上的 我麻了
    #实际测出来应该是tuples=rows而不是reltuples。
    #tuples=reltuples

    #[2]模拟get_variable_numdistinct得到ndistinct
    isdefault,ndistinct=get_variable_numdistinct(tuples,reltuples,mcv_freq,stanullfrac,stadistinct,isUnique)
    # print(isdefault,ndistinct)

    #[3]如果ndistinct不是真实值，直接使用0.1/mcv_freq
    if(isdefault):
        inner_bucket_size=min(0.1,mcv_freq)
        #return
    else:
        #[4]Compute avg freq
        avgfreq=(1.0 - stanullfrac) / ndistinct
        #[5]Adjust ndistinct to account for restriction clauses. 针对选择性谓词修正ndistinct，默认平均分布，谓词选择率*ndistinct
        if(tuples>0):
            ndistinct=clamp_row_est(ndistinct*(rows*1.0/tuples)) #ndistinct *= vardata.rel->rows / vardata.rel->tuples;
        #[6]Initial estimate of bucketsize fraction is 1/nbuckets
        estfract=1.0/(min(nbuckets,ndistinct))
        #[7]Adjust estimated bucketsize upward to account for skewed distribution.
        # if (avgfreq > 0.0 and mcv_freq > avgfreq):
        #     estfract *= mcv_freq / avgfreq
        #[8]Clamp.
        estfract=max(estfract,1.0e-6)
        estfract = min(estfract,1.0)# clamp to [1.0e-6,1.0]
        inner_bucket_size=estfract
    #几个比较重要的值
    # print(tuples,nbuckets,stanullfrac,stadistinct,ndistinct,isdefault,mcv_freq)
    # print(inner_bucket_size)
    # print(inner_bucket_size)
    # if inner_bucket_size > 0.01:
    #     print("here")
    return inner_bucket_size

