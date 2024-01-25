# Rethinking Learned Cost Models: Why Start from Scratch?

Core code of our SIGMOD'24 paper "Rethinking Learned Cost Models: Why Start from Scratch?"


In this repository, we have open-sourced the core code for ParamTree. This code can build decision trees for query/data-related C-params and predict the execution cost of queries.

## Requirements

- Python 3.7
- pip3 install -r requirements.txt
- Install PostgreSQL 13.3
- Install Extension in PostgresL: pageinspect and pg_hint_plan
- alter system set max_parallel_workers_per_gather =0; 
- alter system set enable_bitmapscan = off; 
- Analyze;


## Data Collection
1. Add "EXPLAIN (ANALYZE, COSTS, BUFFERS, VERBOSE, FORMAT JSON)" to get the running statistics of queries.
2. Because this model does not support Bitmap Index Scan, and due to machine performance issues, some query execution times are too long. We ignored these queries during execution.
3. To eliminate interference from other programs and the impact of cached data in the shared buffer, please exclude interference from other programs and clear the database cache when collecting data.
4. Data Format
```
{
     "planinfo": result return by PostgreSQL,
     "template": 'synthetic'/'scale'/'job-light' for IMDB or 'Q1..Q22' for TPCH,
     "query': Executed query
}
```


## Run
This implementation follows the traditional approach of training the model using a training dataset and testing it using a testing dataset. This is the same approach adopted by other deep learning-based methods, allowing for a more equitable comparison.

Our method can be trained:
```
python main.py --train_data ./data/imdb_synthetic.txt --test_data ./data/imdb_job-light.txt --db imdb --save_model_name imdb_synthetic --load_model_name imdb_synthetic --leaf_num 10
```

Test:
```
python main.py --test --load_model --test_data ./data/imdb_scale.txt --db imdb --load_model_name imdb_synthetic
```

### Information
If you want to collect information from your own database, please fill in the relevant information of the database for connection in \feature\info.py. As clearing the cache of the operating system is required during runtime, a Linux account with root privileges also needs to be provided.

Example:
```
db_info = {'server':"127.0.0.1",
            'pg':{
                  'username':"postgres",
                 'password':"postgres",
                 'port':5434,
                 'command_ctrl':"docker exec -it  --user postgres database /home/usr/pgsql13.1/bin/pg_ctl -D /home/usr/pgsql13.1_data"},
            'ssh' : {
                   'username':"root",
                 'password':"root",
                 'port':22
            }
        }
```
