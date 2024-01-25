from .db_connector import *
from feature.info import schema_db_info as db_info
inner_bucket_size=0
DEFAULT_NUM_DISTINCT = 200 #default number of distinct values in a table

def clamp_row_est(nrows):
	if (nrows <= 1.0):
		nrows = 1.0
	else:
		nrows = int(nrows)
	return nrows

def get_variable_numdistinct(tuples,reltuples,mcv_freq,stanullfrac,stadistinct,isUnique):
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

def get_innerbucketsize(inner_bucket_size_info,tbl_name,col_name,isUnique,num_batches,num_buckets,rows,tuples,db_name):
    db_info['pg']['db_name'] = db_name
    # db = Postgres_Connector(server=db_info['server'], pg = db_info['pg'], ssh = db_info['ssh'])
    nbuckets=num_batches*num_buckets
    if isUnique:
        return 1/nbuckets
    #[1]get mcv_freq
    (reltuples,mcv_freq,stanullfrac,stadistinct)=inner_bucket_size_info[tbl_name][col_name]

    #[2]simulate get_variable_numdistinct to get ndistinct
    isdefault,ndistinct=get_variable_numdistinct(tuples,reltuples,mcv_freq,stanullfrac,stadistinct,isUnique)
    # print(isdefault,ndistinct)

    #[3]if ndistinct is not accurate，use 0.1/mcv_freq
    if(isdefault):
        inner_bucket_size=min(0.1,mcv_freq)
        #return
    else:
        #[4]Compute avg freq
        avgfreq=(1.0 - stanullfrac) / ndistinct
        #[5]Adjust ndistinct to account for restriction clauses. 
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
    # print(tuples,nbuckets,stanullfrac,stadistinct,ndistinct,isdefault,mcv_freq)
    # print(inner_bucket_size)
    # print(inner_bucket_size)
    # if inner_bucket_size > 0.01:
    #     print("here")
    return inner_bucket_size

