
import json
import numpy as np


from .info import features,dbms_cparams,query_cparams
BLCKSZ = 8 * 1024 #bytes


class FeatureExtract():
    def __init__(self):
        knob_file = './schemeinfo/conf.json'
        with open(knob_file) as f:
            confs = json.load(f)
        self.general_features = confs.keys()
        self.features = features
        self.leaf_features = ['Nt', 'No','Ni','Ns','Nr','Nm','Np']
        self.target_feature = ['y']

    def get_general_features(self,plan_json):
        res = {}
        op = plan_json['name']
        for key in self.features[op]['runtime_cost']['node_features']:
            if key in dbms_cparams:
                if plan_json[key] == "on":
                    res[key] = 1
                elif plan_json[key] == "off":
                    res[key] = 0
                else:
                    try:
                        res[key] = float(plan_json[key])
                    except:
                        res[key] = plan_json[key]
            else:
                if key in ['NT','NO','NI','NS','NR']:
                    res[key] = 0
                    continue
                try:
                    res[key] = float(plan_json[key])
                except:
                    res[key] = plan_json[key]
        return res



    def get_model_feature(self,plan_json,operator,model):
        feat = self.get_model_raw_feature(plan_json,operator,model)
        features = self.features[operator][model]['node_features'] + self.leaf_features + self.target_feature
        res = []
        for feature in features:
            res.append(feat[feature])
        return res

    def get_model_raw_feature(self,plan_json,operator,model):
        
        sort_mem_bytes = int(plan_json['work_mem'])*1024
        effective_cache_size = int(plan_json['effective_cache_size']) #pages
        work_mem = int(plan_json['work_mem'])*1024 #bytes
        X = self.get_general_features(plan_json)
        if operator == 'Sort' and model == 'startup_cost':
            X['Nt'] = 0
            X['No'] = 2*plan_json['LeftRows']*np.log2(plan_json['LeftRows']+1)*plan_json['CondNum']
            X['Ni'] = 0
            X['Ns'] = 0
            X['Nr'] = 0
            X['Np'] = 0
            X['Nm'] = 0
            tuples = max(1,plan_json['Rows'])
            output_bytes = tuples * plan_json['SoutAvg']
            if output_bytes>sort_mem_bytes:
                X['No'] = 2*plan_json['LeftRows']*np.log2(plan_json['LeftRows']+1)*plan_json['CondNum']
                def compute_merge_order(allowedMem):
                    TAPE_BUFFER_OVERHEAD = BLCKSZ
                    MERGE_BUFFER_SIZE = BLCKSZ * 32
                    MINORDER = 6
                    MAXORDER = 500
                    mOrder = (allowedMem - TAPE_BUFFER_OVERHEAD)/(MERGE_BUFFER_SIZE + TAPE_BUFFER_OVERHEAD)
                    mOrder = max(mOrder, MINORDER)
                    mOrder = min(mOrder, MAXORDER)
                    return mOrder
                mergeorder = compute_merge_order(sort_mem_bytes)
                def getrowinfo(row):
                    if 'Sort Space Type' in row.keys() and row['Sort Space Type'] == 'Memory':
                        return 0
                    Left_SinTot = row['LeftRows'] * row['LeftSoutAvg']
                    return 2*np.ceil(Left_SinTot/BLCKSZ)*max(1,np.ceil(np.log(Left_SinTot/sort_mem_bytes+1)/np.log(mergeorder)))
                X['Ns'] = getrowinfo(plan_json)*0.75
                X['Nr'] = getrowinfo(plan_json)*0.25
            elif plan_json['LeftRows']>2*tuples or ('Sort Space Type' in plan_json.keys() and plan_json['Sort Space Type'] == 'Memory'):
                X['Nt'] = 0
                X['No'] = 2*plan_json['LeftRows'] * np.log2(2*tuples)*plan_json['CondNum']

            X['y'] = plan_json['Actual Startup Time']-plan_json['Left Total Time']

        elif operator == 'Sort' and model == 'runtime_cost':
            X['Nt'] = 0
            X['No'] = plan_json['LeftRows']*plan_json['CondNum']
            X['Ni'] = 0
            X['Ns'] = 0
            X['Nr'] = 0
            X['Np'] = plan_json['Rows']
            X['Nm'] = 0
            X['y'] = plan_json['Actual Total Time']-plan_json['Actual Startup Time']

        elif operator == 'Hash Join' and model == 'startup_cost':
            X['Nt'] = plan_json['RightRows']
            X['No'] = plan_json['RightRows']*plan_json['CondNum']
            X['Ni'] = 0

            Right_SinTot = plan_json['RightRows'] * plan_json['RightSoutAvg']
            if Right_SinTot / BLCKSZ < float(plan_json['work_mem']):
                X['Ns'] = np.sign(Right_SinTot -1)*Right_SinTot /BLCKSZ
            else:
                X['Ns'] = np.sign(Right_SinTot - 1) * Right_SinTot / BLCKSZ * 2
            X['Nr'] = 0
            X['Np'] = 0
            X['Nm'] = 0
            X['y']  = plan_json['Actual Startup Time']-plan_json['Right Total Time']

        elif operator == 'Hash Join' and model == 'runtime_cost':
            innerbucketsize = plan_json['inner_bucket_size']
            Right_SinTot = plan_json['RightRows'] * plan_json['RightSoutAvg']
            Left_SinTot = plan_json['LeftRows'] * plan_json['LeftSoutAvg']
            X['Nt'] = plan_json['Rows']
            X['No'] = plan_json['LeftRows']*plan_json['CondNum']+plan_json['FilterNum']*plan_json['Rows']
            X['Ni'] = 0
            if Right_SinTot/BLCKSZ<int(plan_json['work_mem']) or Left_SinTot/BLCKSZ < int(plan_json['work_mem']):
                X['Ns'] = np.sign(plan_json['BatchesNum']-1)*(Right_SinTot/BLCKSZ + 2*Left_SinTot/BLCKSZ)
            else:
                X['Ns'] = np.sign(plan_json['BatchesNum'] - 1) * (2*Right_SinTot / BLCKSZ + 4 * Left_SinTot / BLCKSZ)
            X['Nr'] = 0
            X['Np'] = plan_json['Rows']
            rows = plan_json['RightRows']*innerbucketsize
            if rows < 1:
                rows = 1
            else:
                rows = int(rows)
            X['Nm'] = plan_json['CondNum']*plan_json['LeftRows']*rows*0.5
            X['y'] = plan_json['Actual Total Time']-plan_json['Actual Startup Time']-plan_json['Left Total Time']#+plan_json['Left Startup Time']

        elif operator == 'Seq Scan' and model == 'runtime_cost':
            X['Nt'] = plan_json['LeftRows']
            X['No'] = plan_json['LeftRows']*plan_json['FilterNum']
            X['Ni'] = 0
            X['Ns'] = plan_json['TablePages']
            X['Nr'] = 0
            X['Np'] = -plan_json['Rows']
            X['Nm'] = 0
            X['y'] = plan_json['Actual Total Time'] - plan_json['InitPlan Cost Time']

        elif operator == 'Index Scan' and model == 'runtime_cost':
            plan_json['fetch_rows'] = np.ceil(plan_json['LeftRows'] * plan_json['Selectivity'])
            X['Nt'] = plan_json['fetch_rows'] 
            X['No'] = plan_json['fetch_rows'] * (
                        plan_json['CondNum'] + plan_json['FilterNum']) + np.ceil(np.log2(plan_json['LeftRows']))
            X['Ni'] = plan_json['fetch_rows']
            indexCorrelation = plan_json['IndexCorrelation']


            def mackert_lohman(tuples_fetched,pages,index_pages):
                total_pages = pages+index_pages
                T = max(1,pages)
                b = effective_cache_size * T / total_pages
                b = max(1, np.ceil(b))
                if T <= b:
                    pages_fetched = (2 * T * tuples_fetched) / (2 * T + tuples_fetched)
                    if pages_fetched >= T:
                        pages_fetched = T
                    else:
                        pages_fetched = np.ceil(pages_fetched)
                else:
                    lim = (2 * T * b) / (2 * T - b)
                    if tuples_fetched <= lim:
                        pages_fetched = (2 * T * tuples_fetched) / (2 * T + tuples_fetched)
                    else:
                        pages_fetched = b + (tuples_fetched - lim) * (T - b) / T
                    pages_fetched = np.ceil(pages_fetched)
                return pages_fetched
            if plan_json['Loops'] == 1:
                ml_value = mackert_lohman(plan_json['fetch_rows']*plan_json['Loops'],plan_json['TablePages'],plan_json['IndexTreePages'])
                X['Ns'] = indexCorrelation * indexCorrelation * np.ceil(
                    plan_json['Selectivity'] * plan_json['TablePages'] - 1)
                X['Nr'] = np.ceil(plan_json['Selectivity'] * plan_json['IndexTreePages']) + ml_value * (
                            1 - indexCorrelation * indexCorrelation)
            else:
                pages_fetched_1 = mackert_lohman(plan_json['fetch_rows']*plan_json['Loops'],plan_json['TablePages'],plan_json['IndexTreePages'])

                pages_fetched_2 = mackert_lohman(np.ceil(plan_json['TablePages']*plan_json['Selectivity'])*plan_json['Loops'],plan_json['TablePages'],plan_json['IndexTreePages'])
                X['Nr'] = (pages_fetched_1+pages_fetched_2)/plan_json['Loops']
                X['Ns'] = 0

            X['Ns'] = X['Ns'] if X['Ns'] > 0 else 0
            X['Nr'] = X['Nr'] if X['Nr'] > 0 else 0
            X['Np'] = plan_json['Rows']
            X['Nm'] = (plan_json['IndexTreeHeight'] + 1) * 50
            X['y'] = plan_json['Actual Total Time'] - plan_json['InitPlan Cost Time']


        elif operator == 'Index Only Scan' and model == 'runtime_cost':
            X['Nt'] = plan_json['LeftRows'] * plan_json['Selectivity']
            X['No'] = plan_json['LeftRows'] * plan_json['Selectivity'] * plan_json['CondNum']+np.ceil(np.log2(plan_json['LeftRows']))
            X['Ni'] = plan_json['LeftRows'] * plan_json['Selectivity']
            X['Ns'] = 0
            X['Nr'] = np.ceil(plan_json['Selectivity'] * plan_json['IndexTreePages'])
            X['Np'] = 1
            X['Nm'] = (plan_json['IndexTreeHeight']+1)*50
            X['y'] = plan_json['Actual Total Time'] - plan_json['InitPlan Cost Time']


        elif operator == 'Nested Loop' and model == 'runtime_cost':
            X['Nt'] = plan_json['LeftRows'] * plan_json['RightRows']
            X['Ni'] = 0
            X['Nr'] = 0
            b = plan_json['Left Total Time']+plan_json['Right Startup Time']
            if plan_json['InnerUnique'] == True:
                if (plan_json['LeftRows'] * plan_json['RightRows']) == 0:
                    inner_scan_frac = 0
                else:
                    inner_scan_frac = plan_json['Rows'] / (plan_json['LeftRows'] * plan_json['RightRows'])
            else:
                inner_scan_frac = 1
            Right_SinTot = plan_json['RightRows'] * plan_json['RightSoutAvg']
            if plan_json['RightOp']=='Materialize' or plan_json['RightOp']=='Sort':
                if  Right_SinTot>work_mem:
                    X['No'] = plan_json['LeftRows'] * plan_json['RightRows']*inner_scan_frac*plan_json['CondNum'] + plan_json['LeftRows'] * plan_json['RightRows']*inner_scan_frac
                    X['Ns'] = plan_json['LeftRows'] * Right_SinTot / BLCKSZ
                else:
                    X['No'] = plan_json['LeftRows'] * plan_json['RightRows']*inner_scan_frac*plan_json['CondNum'] + plan_json['LeftRows'] * plan_json['RightRows']*inner_scan_frac
                    X['Ns'] = 0
            else:

                b += plan_json['Right Startup Time']*(plan_json['LeftRows']-1)+plan_json['LeftRows']*(plan_json['Right Total Time']-plan_json['Right Startup Time'])*inner_scan_frac
                X['No'] = plan_json['LeftRows'] * plan_json['RightRows']*inner_scan_frac*plan_json['CondNum']
                X['Ns'] = 0
            X['Np'] = plan_json['Rows']
            X['Nm'] = 0 
            X['y'] = plan_json['Actual Total Time']-b

        
        elif operator == 'Merge Join' and model == 'runtime_cost':
            X['Nt'] = plan_json['Rows']
            X['Ni'] = 0
            X['Nr'] = 0
            rescantuples = plan_json['Rows']-plan_json['RightRows']
            rescantuples = rescantuples if rescantuples>0 else 0
            if plan_json['RightRows'] == 0:
                rescanratio = 0
            else:
                rescanratio = 1+rescantuples/plan_json['RightRows']
            b = plan_json['Left Total Time']+plan_json['Right Total Time']
            X['No'] = plan_json['CondNum']*(plan_json['LeftRows'] + plan_json['RightRows']*rescanratio)
            X['Ns'] = 0
            X['Np'] = 0
            X['Nm'] = (plan_json['Right Total Time']-plan_json['Right Startup Time'])*(rescanratio-1)
            X['y'] = plan_json['Actual Total Time']-b
        
        elif operator == 'Aggregate' and model == 'startup_cost':
            X['Nt'] = 0
            X['No'] = 0
            X['Ni'] = 0
            X['Ns'] = 0
            X['Nr'] = 0
            X['Np'] = 0
            X['Nm'] = 0
            nbatches = 1
            Left_SinTot = plan_json['LeftRows'] * plan_json['LeftSoutAvg']
            if plan_json['Strategy'] == "Plain":
                X['No'] = plan_json['CalculatingNum']*plan_json['LeftRows']+plan_json['OutAggColumnNum']
                X['y'] = plan_json['Actual Startup Time']-plan_json['Left Total Time']
            elif plan_json['Strategy'] == "Sorted":
                X['y'] = plan_json['Actual Startup Time']-plan_json['Left Startup Time']

            elif plan_json['Strategy'] == "Hashed":
                nbatches = plan_json['BatchesNum']
                num_partitions = plan_json['partitions']
                nbatches = max(nbatches,1)
                num_partitions = max(num_partitions,2)
                depth = np.ceil(np.log(nbatches)/np.log(num_partitions))
                input_pages = np.ceil(Left_SinTot/BLCKSZ)
                X['Nr'] = depth*input_pages*2
                X['Nt'] = depth*2*plan_json['LeftRows']
                X['No'] = (plan_json['CalculatingNum']+plan_json['CondNum'])*plan_json['LeftRows']
                X['y'] = plan_json['Actual Startup Time']-plan_json['Left Total Time']
            elif plan_json['Strategy'] == "Mixed":
                X['No'] = plan_json['LeftRows']
                X['y'] = plan_json['Actual Startup Time'] - plan_json['Left Total Time']
           


        elif operator == 'Aggregate' and model == 'runtime_cost':
            X['Nt'] = 0
            X['No'] = 0
            X['Ni'] = 0
            X['Ns'] = 0
            X['Nr'] = 0
            X['Np'] = 0
            X['Nm'] = 0
            nbatches = 1
            X['y'] = plan_json['Actual Total Time']-plan_json['Actual Startup Time']
            Left_SinTot = plan_json['LeftRows'] * plan_json['LeftSoutAvg']
            if plan_json['Strategy'] == "Plain":
                X['Nt'] = 1
            elif plan_json['Strategy'] == "Sorted":
                X['No'] = (plan_json['CalculatingNum']+plan_json['CondNum'])*plan_json['LeftRows']+plan_json['Rows']*plan_json['OutAggColumnNum']
                X['Nt'] = plan_json['Rows']
                X['y'] = plan_json['Actual Total Time']-plan_json['Left Total Time']
            elif plan_json['Strategy'] == "Hashed":
                nbatches = plan_json['BatchesNum']
                num_partitions = plan_json['partitions']
                nbatches = max(nbatches, 1)
                num_partitions = max(num_partitions, 2)
                depth = np.ceil(np.log(nbatches)/np.log(num_partitions))
                # depth = max(depth,1)
                input_pages = np.ceil(Left_SinTot/BLCKSZ)
                X['Ns'] = depth*input_pages*2
                X['Nt'] = plan_json['Rows']
                X['No'] = plan_json['Rows']*plan_json['OutAggColumnNum']
        if X['y']<0:
            X['y'] = 0
        X['Np'] = 0
        for key in ['NT','NO','NI','NS','NR']:
            X[key] = X[key[0]+key[1].lower()]


        X['No'] = X['No']+X['Nm']
        X['Nm'] = 0
        X['Np'] = 0
        X['Nt'] = X['Nt'] * plan_json['Loops']
        X['No'] = X['No'] * plan_json['Loops']
        X['Ni'] = X['Ni'] * plan_json['Loops']
        X['Ns'] = X['Ns'] * plan_json['Loops']
        X['Nr'] = X['Nr'] * plan_json['Loops']
        X['Np'] = X['Np']
        X['Nm'] = X['Nm']
        if model == 'runtime_cost':
            X['y'] = X['y'] * plan_json['Loops'] - plan_json['SubPlan Cost Time']
        else:
            X['y'] = X['y'] * plan_json['Loops'] - plan_json['SubPlan Startup Time'] - plan_json['InitPlan Cost Time']
        res = {}
        for key in self.features[operator][model]['node_features']+self.leaf_features+self.target_feature:
            res[key] = X[key]
        if X['y']>-3 and X['y']<0:
            X['y'] = 0
        for key in self.leaf_features+self.target_feature:
            if X[key] < 0:
                print("here")
        return res

