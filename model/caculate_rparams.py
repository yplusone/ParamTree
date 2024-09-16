import re
import pandas as pd
import numpy as np
from util.MyLinearModel import MyLinearRegression
class Calculate_5params():
    def __init__(self,plan_trees,scheme_info):
        self.scheme_info = scheme_info
        self.coef = self.get_5_params_by_linearModel(plan_trees)
        self.scale = self.get_scale(plan_trees)
    def get_split_filters(self, str):
        res = []
        ans = re.split('\sAND\s', str)
        for item in ans:
            t = re.split('\sOR\s', item)
            res += t
        return res

    def get_feature(self,plan):
        res = {}
        if plan['Node Type'] == "Seq Scan":
            res['Ns'] = plan['Shared Read Blocks']
            res['Nt'] = self.scheme_info.table_features[plan['Relation Name']]['tuple_num']
            if 'Filter' in plan.keys():
                res['No'] = len(self.get_split_filters(plan['Filter']))*res['Nt']
            else:
                res['No'] = 0
            res['Ni'] = 0
            res['Nr'] = 0
            res['y_disk'] = plan['I/O Read Time']
            res['y_cpu'] = plan['Actual Total Time'] - plan['I/O Read Time']
            res['correlation'] = 0
        elif plan['Node Type'] == "Index Scan":
            res['Nr'] = plan['Shared Read Blocks']
            res['Nt'] = plan['Actual Rows']
            if 'Filter' in plan.keys():
                if 'Index Cond' in plan.keys():
                    res['No'] = (len(self.get_split_filters(plan['Index Cond']))+len(self.get_split_filters(plan['Filter'])))*res['Nt']
                else:
                    res['No'] = res['Nt']
            else:
                if 'Index Cond' in plan.keys():
                    res['No'] = len(self.get_split_filters(plan['Index Cond']))*res['Nt']
                else:
                    res['No'] = res['Nt']
            res['Ni'] = plan['Actual Rows']
            res['Ns'] = 0
            res['y_disk'] = plan['I/O Read Time']
            res['y_cpu'] = plan['Actual Total Time'] - plan['I/O Read Time']

            res['correlation'] = self.scheme_info.index_features[plan['Index Name']]['indexCorrelation']
        res['y'] = plan['Actual Total Time']
        return res
    def get_scale(self,plan_trees):
        pred = []
        true = []
        def get_op_info(plan):
            if 'Plans' in plan.keys():
                for item in plan['Plans']:
                    get_op_info(item)
            if plan['Node Type'] in ['Seq Scan']:
                pred.append(plan['Total Cost'])
                true.append(plan['Actual Total Time'])
            else:
                pass

        for plan in plan_trees:
            get_op_info(plan['Plan'])

        return np.median(np.array(pred)/np.array(true))


    def get_5_params_by_linearModel(self,plan_trees):
        result = []
        def get_op_info(plan):
            if 'Plans' in plan.keys():
                for item in plan['Plans']:
                    get_op_info(item)
            if plan['Node Type'] in ['Seq Scan','Index Scan']:
                result.append(self.get_feature(plan))
            else:
                pass
        for plan in plan_trees:
            get_op_info(plan['Plan'])
        df = pd.DataFrame(result)
        df_a = df[(df['Ns'] > 100) & (df['Nr'] == 0)]

        # print("disk fit score: %.2f"%lr.score(df[['Ns','Nr']],df['y_disk']))
        cs = np.median(df_a['y_disk']/df_a['Ns'])

        df_a = df[df['Nr']>10]
        cr = np.median((df_a['y_disk'] - df_a['Ns']*cs)/df_a['Nr'])
        if len(df[df['No']>0]) < len(df)*2/3:
            df_a = pd.concat([df[df['No']>0],df[df['No']==0].iloc[:round(len(df[df['No']>0])*1/2)]])
        lr = MyLinearRegression()
        df_a = df[(df['Ni'] == 0) &(df['No']!=df['Nt']) & (df['No']!=0)]
        lr.fit(np.array(df_a[['Nt','No']]),np.array(df_a['y_cpu']))
        # pred = lr.predict(df[['Nt','No','Ni']])
        # plt.scatter(df['y_cpu'],pred,s=0.3)
        # plt.show()
        # print("cpu fit score: %.2f"%lr.score(df[['Nt','No','Ni']],df['y_cpu']))
        ct = lr.coef_[0]
        co = lr.coef_[1]
        df_t = df[df['Ni'] != 0]
        lr.fit(np.array(df_t[['Nt', 'No', 'Ni']]), np.array(df_t['y_cpu']))
        # ci = np.median((df_t['y_cpu']-df_t['Nt']*lr.coef_[0]-df_t['No']*lr.coef_[1])/df_t['Ni'])#lr.coef_[2]
        ci = lr.coef_[2]
        return [ct,co,ci,cs,cr,co,co]