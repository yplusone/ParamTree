import numpy as np

query_cparams = ['FilterNum', 'FilterOffset', 'FilterIntegerRatio', 'FilterFloatRatio', 'FilterStrRatio', 'FilterColumnNum',
                          'CondNum', 'CondOffset', 'CondIntegerRatio', 'CondFloatRatio', 'CondStrRatio', 'CondColumnNum',
                          'IndexCorrelation', 'IndexTreeHeight', 'IndexTreePages', 'IndexTreeUniqueValues',
                          'ParentOp', 'SoutAvg', 'Rows', 'Loops', 'LeftOp', 'RightOp',
                          'LeftSoutAvg', 'LeftRows', 'LeftLoops',
                          'RightSoutAvg', 'RightRows', 'RightLoops',
                          'InnerUnique', 'TablePages', 'TuplesPerBlock',
                          'Selectivity', 'BucketsNum', 'BatchesNum', 'Strategy'
                          ]
dbms_cparams = ['allow_system_table_mods', 'array_nulls', 'backend_flush_after', 'check_function_bodies', 'commit_delay', 'commit_siblings', 'cursor_tuple_fraction', 'deadlock_timeout', 'default_statistics_target', 'default_transaction_deferrable', 'default_transaction_read_only', 'effective_cache_size', 'effective_io_concurrency', 'escape_string_warning', 'exit_on_error', 'extra_float_digits', 'from_collapse_limit', 'geqo', 'geqo_effort', 'geqo_generations', 'geqo_pool_size', 'geqo_seed', 'geqo_selection_bias', 'geqo_threshold', 'gin_fuzzy_search_limit', 'gin_pending_list_limit', 'hash_mem_multiplier', 'idle_in_transaction_session_timeout', 'ignore_checksum_failure', 'jit', 'jit_dump_bitcode', 'jit_expressions', 'jit_inline_above_cost', 'jit_optimize_above_cost', 'jit_tuple_deforming', 'join_collapse_limit', 'lo_compat_privileges', 'lock_timeout', 'log_duration', 'log_executor_stats', 'log_lock_waits', 'log_min_duration_sample', 'log_min_duration_statement', 'log_parameter_max_length', 'log_parameter_max_length_on_error', 'log_parser_stats', 'log_planner_stats', 'log_replication_commands', 'log_statement_sample_rate', 'log_temp_files', 'log_transaction_sample_rate', 'logical_decoding_work_mem', 'maintenance_io_concurrency', 'maintenance_work_mem', 'max_stack_depth', 'min_parallel_index_scan_size', 'min_parallel_table_scan_size', 'operator_precedence_warning', 'parallel_leader_participation', 'parallel_setup_cost', 'parallel_tuple_cost', 'quote_all_identifiers', 'row_security', 'standard_conforming_strings', 'statement_timeout', 'synchronize_seqscans', 'tcp_keepalives_count', 'tcp_keepalives_idle', 'tcp_keepalives_interval', 'tcp_user_timeout', 'temp_buffers', 'temp_file_limit', 'trace_notify', 'trace_sort', 'track_activities', 'track_counts', 'track_io_timing', 'transaction_read_only', 'transform_null_equals', 'update_process_title', 'vacuum_cleanup_index_scale_factor', 'vacuum_cost_delay', 'vacuum_cost_limit', 'vacuum_cost_page_dirty', 'vacuum_cost_page_hit', 'vacuum_cost_page_miss', 'vacuum_freeze_min_age', 'vacuum_freeze_table_age', 'vacuum_multixact_freeze_min_age', 'vacuum_multixact_freeze_table_age', 'wal_compression', 'wal_init_zero', 'wal_recycle', 'wal_sender_timeout', 'wal_skip_threshold', 'work_mem', 'zero_damaged_pages']

all_cparams = query_cparams #+ dbms_cparams

features = {
            "Seq Scan":{
                "runtime_cost":{"node_features":all_cparams},            },
            "Index Scan":{
                "runtime_cost":{"node_features":all_cparams},            },
            "Index Only Scan":{
                "runtime_cost":{"node_features":all_cparams},            },
            "Sort":{
                "startup_cost":{"node_features":all_cparams},                
                "runtime_cost":{"node_features":all_cparams},            },
            "Hash Join":{
                "startup_cost":{"node_features":all_cparams},                
                "runtime_cost":{"node_features":all_cparams},            },
            "Nested Loop":{
                "runtime_cost":{"node_features":all_cparams},            },
            "Merge Join":{
                "runtime_cost":{"node_features":all_cparams},            },
            "Aggregate":{
                "startup_cost":{"node_features":all_cparams},                
                "runtime_cost":{"node_features":all_cparams},            },
        }
## For Running queries online
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

# For collecting statistics

schema_db_info = {'server':"127.0.0.1",
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

coefs = np.array([np.array([3.93205145e-05, 7.86410291e-05, 3.48516085e-04, 5.03552805e-03,
                            4.49506951e-02, 7.86410291e-05, 7.86410291e-05]), 0], dtype=object)
scale = 92.58

