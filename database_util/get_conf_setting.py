import json

# settings = [
#     "temp_buffers",
#     "work_mem",
#     "effective_cache_size",
#     "jit",
#     "track_activities",
#     "track_counts",
#     "track_io_timing",
#     "enable_mergejoin",
#     "enable_nestloop",
#     "enable_hashjoin",
#     "enable_seqscan",
#     "enable_indexonlyscan",
#     "enable_indexscan",
#     "enable_material",
#     "enable_sort",
#     "log_transaction_sample_rate",
#     "log_min_duration_statement",
# ]
# settings = ["temp_buffers","work_mem","max_stack_depth","effective_cache_size","jit","track_activities","track_counts",
#             "track_io_timing","check_function_bodies",
#             "backend_flush_after","commit_delay","commit_siblings",
#             "cursor_tuple_fraction","deadlock_timeout","default_statistics_target","default_transaction_deferrable",
#             "default_transaction_read_only","escape_string_warning","extra_float_digits","from_collapse_limit",
#             "geqo","geqo_effort","geqo_generations","geqo_pool_size","geqo_seed","geqo_selection_bias",
#             "geqo_threshold","gin_fuzzy_search_limit","gin_pending_list_limit","hash_mem_multiplier",
#             "jit_dump_bitcode","jit_expressions","jit_inline_above_cost","jit_optimize_above_cost",
#             "jit_tuple_deforming","join_collapse_limit",
#             "log_duration","log_executor_stats",
#             "log_lock_waits","log_min_duration_sample","log_min_duration_statement","log_parameter_max_length",
#             "log_parameter_max_length_on_error","log_parser_stats","log_planner_stats","log_replication_commands",
#             "log_statement_sample_rate","log_temp_files",
#             "log_transaction_sample_rate","maintenance_io_concurrency","logical_decoding_work_mem",
#             "maintenance_work_mem",
#             "min_parallel_index_scan_size","min_parallel_table_scan_size",'allow_system_table_mods', 'array_nulls',
#             'effective_io_concurrency', 'exit_on_error', 'idle_in_transaction_session_timeout', 'ignore_checksum_failure',
#             'lo_compat_privileges', 'lock_timeout', 'operator_precedence_warning', 'parallel_leader_participation',
#             'parallel_setup_cost', 'parallel_tuple_cost', 'quote_all_identifiers', 'row_security', 'standard_conforming_strings',
#             'statement_timeout', 'synchronize_seqscans', 'tcp_keepalives_count', 'tcp_keepalives_idle', 'tcp_keepalives_interval',
#             'tcp_user_timeout', 'temp_file_limit', 'trace_notify', 'trace_sort', 'transaction_read_only', 'transform_null_equals',
#             'update_process_title', 'vacuum_cleanup_index_scale_factor', 'vacuum_cost_delay', 'vacuum_cost_limit',
#             'vacuum_cost_page_dirty', 'vacuum_cost_page_hit', 'vacuum_cost_page_miss', 'vacuum_freeze_min_age',
#             'vacuum_freeze_table_age', 'vacuum_multixact_freeze_min_age', 'vacuum_multixact_freeze_table_age',
#             'wal_compression', 'wal_init_zero', 'wal_recycle', 'wal_sender_timeout', 'wal_skip_threshold', 'zero_damaged_pages']

# restart settings
settings = ['shared_buffers', 'huge_pages', 'max_files_per_process', 'max_worker_processes', 'wal_buffers']


def get_conf_setting(db_connector):
    res = {}
    print(len(settings))
    for setting in settings:
        res[setting] = {}
        ans = db_connector.execute(
            f"select name,vartype,min_val,max_val,reset_val,pending_restart from pg_settings where name='{setting}'")[0]
        if ans[1] != 'bool':
            res[setting]['default'] = ans[4]
            res[setting]['type'] = ans[1]
            if ans[1] == "integer":
                res[setting]['type'] = "int"
                res[setting]['min'] = int(ans[2])
                res[setting]['max'] = int(ans[3])
            elif ans[1] == "real":
                res[setting]['type'] = "float"
                res[setting]['min'] = float(ans[2])
                if float(ans[3]) > 1e8:
                    res[setting]['max'] = 1e8
                else:
                    res[setting]['max'] = float(ans[3])
            res[setting]['restart'] = ans[5]

        elif ans[1] == 'bool':
            res[setting]['default'] = ans[4]
            res[setting]['min'] = "off"
            res[setting]['max'] = "on"
            res[setting]['type'] = ans[1]
            res[setting]['restart'] = ans[5]
        if setting == "max_stack_depth":
            res[setting]['max'] = 7680
    print(res)
    with open('./data/restart_conf.json', 'w') as f:
        f.write(json.dumps(res))
    return
