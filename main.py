import sys
import argparse

sys.path.extend(["../", "./"])
from database_util.database_info import *
from database_util.db_connector import *
from model.query_inference import QueryInference
from util.util import get_plantrees, get_test_results, print_args
from feature.info import coefs, scale,query_cparams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # experiment args
    parser.add_argument('--train_data', default="./data/imdb_synthetic.txt", help='Workload file',
                        nargs='+')
    parser.add_argument('--test_data', default="./data/imdb_job-light.txt", help='Workload file',
                        nargs='+')
    parser.add_argument('--db', default='imdb', help='Which database to be tested')
    parser.add_argument('--load_model_name', default='imdb_random', help='The model name to load')
    parser.add_argument('--save_model_name', default='imdb_random', help='The model name to save')
    parser.add_argument('--load_model', action='store_true', help='Whether load exist model')
    parser.add_argument('--test', action='store_true', help='Whether only test')

    # ParamTree args
    parser.add_argument('--leaf_num', type=int, default=10, help='Set the threshold of qerror for buffer')
    parser.add_argument('--trim', type=float, default=0.1, help='Set for parameter instability test')
    parser.add_argument('--alpha', type=float, default=0.05, help='Set the threshold of confidence')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Set the Batch size for paramtree to clear buffer automaticly')

    args = parser.parse_args()
    print_args(parser, args)


    tool = QueryInference(args.db, coefs=coefs, scale=scale, args=args, load=args.load_model,
                          load_model_name=args.load_model_name,node_features=query_cparams)
    
    
    for op in tool.mobtrees.feature_tool.features.keys():
        for key in tool.mobtrees.feature_tool.features[op].keys():
            tool.mobtrees.feature_tool.features[op][key]['node_features'] = query_cparams
    time_start = time.time()
    if not args.test:
        tool.fit(args.train_data)
        tool.save_model(args.save_model_name)
    else:
        for test_file in [args.test_data]:
            test_plans = get_plantrees(test_file, subplan=True)
            y_pred = tool.predict(test_plans)
            y_true = [t['planinfo']['Plan']['Actual Total Time'] for t in test_plans]
            print(get_test_results(y_pred, y_true))

    time_end = time.time()
    print('time cost', time_end - time_start, 's')