import json
import sys
from aim import Run
import shutil

sys.path.extend(["../", "./"])
from database_util.db_connector import *
from model.query_inference import QueryInference
from util.util import get_plantrees, get_test_results, print_args
from feature.infos import coefs, scale,query_cparams
import argparse

import numpy as np
from database_util.database_info import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # experiment args
    parser.add_argument('--mode', default="TEST", help='select running mode in TEST, NORMAL_TRAIN, AL_TRAIN')
    parser.add_argument('--workload', default="./data/experiment/deepdb/imdb_job-light.txt", help='Workload file', nargs='+')
    
    parser.add_argument('--train_data', default="./data/imdb_synthetic.txt", help='Workload file',
                        nargs='+')
    parser.add_argument('--test_data', default="./data/experiment/deepdb/imdb_job-light.txt", help='Workload file',
                        nargs='+')
    parser.add_argument('--db', default='imdb', help='Which database to be tested')
    parser.add_argument('--load_model_name', default='imdb_synthetic', help='The model name to load')
    parser.add_argument('--save_model_name', default='imdb_synthetic', help='The model name to save')
    parser.add_argument('--load_model', action='store_true', help='Whether load exist model')
    parser.add_argument('--subplan', action='store_true', help='Whether compatible with subplan query')

    # ParamTree args
    parser.add_argument('--leaf_num', type=int, default=10, help='Set the threshold of sample number in leaf node')
    parser.add_argument('--train_samples', type=int, default=2000, help='Set the number of train samples')
    parser.add_argument('--trim', type=float, default=0.1, help='Set for parameter instability test')
    parser.add_argument('--alpha', type=float, default=0.05, help='Set the threshold of confidence')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Set the Batch size for paramtree to clear buffer automaticly')

    # AL train args
    parser.add_argument('--qerror_threshold',type=float, default=1.5,help='Set the threshold of qerror for buffer')
    parser.add_argument('--sample_num_per_expansion',type=int, default=50,help='Set the number of samples to collect for node expansion')
    parser.add_argument('--buffer_size',type=int, default=100,help='The buffer size for bad predict queries')
    parser.add_argument('--knob_change',action='store_true',help='Whether the database knob changes')
    parser.add_argument('--aim', action='store_true', help='Whether use aim')
    parser.add_argument('--random_ratio', type=float, default=0.2, help='Set the ratio of selecting operator randomly')

    args = parser.parse_args()
    print_args(parser, args)


    tool = QueryInference(args.db, coefs=coefs, scale=scale, args=args, load=args.load_model,
                          load_model_name=args.load_model_name,node_features=query_cparams) #,node_features=query_cparams
    for op in tool.mobtrees.feature_tool.features.keys():
        for key in tool.mobtrees.feature_tool.features[op].keys():
            tool.mobtrees.feature_tool.features[op][key]['node_features'] = query_cparams
    time_start = time.time()
    if args.mode == "TEST":
        tool = QueryInference(args.db,coefs=coefs,scale=scale,args=args,load=args.load_model,load_model_name=args.load_model_name)
        for test_file in args.workload:
            test_plans = get_plantrees([test_file],subplan=False)
            start_time = time.time()
            y_pred = tool.predict(test_plans)
            end_time = time.time()
            print((end_time-start_time)/len(test_plans))
            y_true = [t['planinfo']['Plan']['Actual Total Time'] for t in test_plans]
            print(get_test_results(y_pred,y_true))
    elif args.mode == "NORMAL_TRAIN":
        tool.fit(args.train_data)
        tool.save_model(args.save_model_name)
    elif args.mode == "AL_TRAIN":
        tool = QueryInference(args.db,coefs=coefs,scale=scale,args=args,load=args.load_model,load_model_name=f"{args.load_model_name}")
        if args.aim:
            run = Run()
            run["hparams"] = vars(args)
        else:
            run = None
        pool_file_path = f"./data/temporary/query_gen_pool"
        file_path = f"{pool_file_path}/db_{args.db}"
        if os.path.exists(file_path):
            shutil.rmtree(file_path)
        tool.train_model_actively(args,run)


    time_end = time.time()
    print('time cost', time_end - time_start, 's')