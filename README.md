# Rethink of a Learned Cost Model: Why Start from Scratch?

python implementation of "Rethinking Learned Cost Models: Why Start from Scratch?"

## Requirements

- Python 3.7
- pip3 install -r requirements.txt
- Install PostgreSQL 13.3
- Install Extension in PostgresL: pageinspect and pg_hint_plan
- alter system set max_parallel_workers_per_gather =0; 
- alter system set enable_bitmapscan = off; 
- Analyze;

## Main Modules
|  Module   | Description  |
|-------|------|
|  model  | Includes how ParamTree to split nodes and how to use ParamTree to predict the execution time of physical plans.|
| feature  | Includes how to extract c-params from queries and databases. Also includes some data used in the rule for calculating cost. |
| experiments  | The code includes experiments for conducting passive and active learning. |
| recommendation  | Code for Section4.2, which recommand c-param for the next split candidate |
| query_gen  | Code for Section4.3, which generate queries from workload queries |



## Run
Our method can be trained in two ways:
- **Passive**: similar to traditional deep learning models, which training on the existing samples
- **Active**: actively generate samples based on actual workflow for training

### Information
Before running, please fill in the relevant information of the database for connection in \feature\info.py. As clearing the cache of the operating system is required during runtime, a Linux account with root privileges also needs to be provided. 

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

It is necessary to create the corresponding database on PostgreSQL and import the data. Some information still needs to be retrieved from the database, such as innerbucketsize in Hash Join.

### Passive
#### Train model

```
python3 main.py --mode NORMAL_TRAIN --train_data ./data/experiment/imdb_synthetic.txt --test_data ./data/experiment/imdb_job-light.txt --db imdb --save_model_name imdb_synthetic_temp --load_model_name imdb_synthetic_temp --leaf_num 20
```
#### Test model
```
python3 main.py --mode TEST --load_model --workload ./data/experiment/imdb_job-light.txt --db imdb --load_model_name imdb_synthetic_temp

python3 main.py --mode TEST --load_model --workload ./data/experiment/imdb_scale.txt --db imdb --load_model_name imdb_synthetic_temp
```
### Active
#### Train model
```
python3 main.py --mode AL_TRAIN --workload ./data/experiment/tpcds_test.txt --db tpcds --save_model_name tpcds_actively --qerror_threshold 1.1 --sample_num_per_expansion 80
```
#### Test model

```
python3 main.py --mode TEST --load_model --workload ./data/experiment/knob_tpcds_test.txt --db tpcds --load_model_name imdb_synthetic_temp
```
