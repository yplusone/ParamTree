
import re
import sys
sys.path.extend(["../","./"])
from tqdm import tqdm
from database_util.db_connector import *
import json
from benchmarker import benchmarker
import argparse
from util.util import print_args
dbms_cparams = ['cpu_tuple_cost','cpu_index_tuple_cost','cpu_operator_cost','seq_page_cost','random_page_cost','allow_system_table_mods', 'array_nulls', 'backend_flush_after', 'check_function_bodies', 'commit_delay', 'commit_siblings', 'cursor_tuple_fraction', 'deadlock_timeout', 'default_statistics_target', 'default_transaction_deferrable', 'default_transaction_read_only', 'effective_cache_size', 'effective_io_concurrency', 'escape_string_warning', 'exit_on_error', 'extra_float_digits', 'from_collapse_limit', 'geqo', 'geqo_effort', 'geqo_generations', 'geqo_pool_size', 'geqo_seed', 'geqo_selection_bias', 'geqo_threshold', 'gin_fuzzy_search_limit', 'gin_pending_list_limit', 'hash_mem_multiplier', 'idle_in_transaction_session_timeout', 'ignore_checksum_failure', 'jit', 'jit_dump_bitcode', 'jit_expressions', 'jit_inline_above_cost', 'jit_optimize_above_cost', 'jit_tuple_deforming', 'join_collapse_limit', 'lo_compat_privileges', 'lock_timeout', 'log_duration', 'log_executor_stats', 'log_lock_waits', 'log_min_duration_sample', 'log_min_duration_statement', 'log_parameter_max_length', 'log_parameter_max_length_on_error', 'log_parser_stats', 'log_planner_stats', 'log_replication_commands', 'log_statement_sample_rate', 'log_temp_files', 'log_transaction_sample_rate', 'logical_decoding_work_mem', 'maintenance_io_concurrency', 'maintenance_work_mem', 'max_stack_depth', 'min_parallel_index_scan_size', 'min_parallel_table_scan_size', 'operator_precedence_warning', 'parallel_leader_participation', 'parallel_setup_cost', 'parallel_tuple_cost', 'quote_all_identifiers', 'row_security', 'standard_conforming_strings', 'statement_timeout', 'synchronize_seqscans', 'tcp_keepalives_count', 'tcp_keepalives_idle', 'tcp_keepalives_interval', 'tcp_user_timeout', 'temp_buffers', 'temp_file_limit', 'trace_notify', 'trace_sort', 'track_activities', 'track_counts', 'track_io_timing', 'transaction_read_only', 'transform_null_equals', 'update_process_title', 'vacuum_cleanup_index_scale_factor', 'vacuum_cost_delay', 'vacuum_cost_limit', 'vacuum_cost_page_dirty', 'vacuum_cost_page_hit', 'vacuum_cost_page_miss', 'vacuum_freeze_min_age', 'vacuum_freeze_table_age', 'vacuum_multixact_freeze_min_age', 'vacuum_multixact_freeze_table_age', 'wal_compression', 'wal_init_zero', 'wal_recycle', 'wal_sender_timeout', 'wal_skip_threshold', 'work_mem', 'zero_damaged_pages']

def run_all_queries_save_raw_data(db,queries,savefile):
    template_set = set()
    with open(savefile,'w') as f:
        for query in tqdm(queries):
            sql = query['sql']
            res_dict = db.explain(sql, execute=True, timeout=600000)
            db.drop_cache()
            change_knob(knob_data)
            res_dict = db.explain(sql,execute=True,timeout=600000)
            res = {}
            res['planinfo'] = res_dict
            res['template'] = query['template']
            res['query'] = sql
            f.writelines(json.dumps(res)+"\n")
            template_set.add(query['template'])

    print("Number of template %d"%len(template_set))
    print(template_set)

def change_knob(data):
    for item in data:
        name = item.split(',')[0]
        if name not in dbms_cparams:
            continue
        value = item.split(',')[1].strip()[7:]
        if value == '\'off\'':
            value = 'off'
        elif value == '\'on\'':
            value = 'on'
        else:
            value = float(value)
        db.execute(f'set {name}={value};',set_env=True)