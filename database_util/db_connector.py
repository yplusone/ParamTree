from collections import *
import time
import psycopg2.extras
import psycopg2
import pandas as pd
import paramiko
import os

class Postgres():
    _connection = None
    _cursor = None

    def __init__(self, pg_url):
        self.pg_url = pg_url
        self._connection = psycopg2.connect(pg_url)
        self.execute

    def execute(self, query, set_env=False):
        '''
        Execute the query and return all the results at once
        '''
        cursor = self._connection.cursor(
            cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query)
        if not set_env:
            return cursor.fetchall()

    def explain(self, query, timeout=0):
        '''
        Execute an 'EXPLAIN ANALYZE' of the query
        '''
        if 'explain' not in query.lower():
            # if not query.lower().startswith('explain'):
            query = 'EXPLAIN (ANALYZE, COSTS, VERBOSE, BUFFERS, FORMAT JSON) ' + query

        if timeout >= 0:
            self.execute(f'SET statement_timeout = {timeout};', set_env=True)
        
        try:
            cursor = self._connection.cursor(
                cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute(query)
            q = cursor.fetchall()
        except Exception as e:
            print("Timeout!!!!", e)
            self._connection.close()
            self._connection = psycopg2.connect(self.pg_url)
            return None
        return q

    def discard_session_state(self):
        old_isolation_level = self._connection.isolation_level
        self._connection.set_isolation_level(0)
        self.execute("DISCARD ALL;",set_env=True)
        self._connection.set_isolation_level(old_isolation_level)

class Postgres_Connector:
    def __init__(self,server, pg,ssh): 
        self.server = server
        self.username = pg['username']
        self.password = pg['password']
        self.db_name = pg['db_name']
        self.ssh_username = ssh['username']
        self.ssh_password = ssh['password']
        self.pg_command_ctrl = pg['command_ctrl']
        self.ssh_port = ssh['port']

        self.count = 0
        if pg['db_name']:
            self.db_url = f"host={server} port={pg['port']} user={pg['username']} dbname={pg['db_name']} password={pg['password']} options='-c statement_timeout={12000000}' "
            self.init_db(pg['db_name'])
            # self.disable_parallel()

    def drop_cache(self):
        flag = True
        while flag:
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(hostname=self.server, port=self.ssh_port,
                            username=self.ssh_username, password=self.ssh_password,timeout=600)
                flag = False
            except:
                print("Sleep 10 Seconds......")
                time.sleep(10)
        stdin, stdout, stderr = ssh.exec_command(
            f"{self.pg_command_ctrl} stop;",get_pty=True)
        out, err = stdout.read(), stderr.read()
        if err:
            print(err)
        stdin, stdout, stderr = ssh.exec_command(
            "free && sync && sudo -S sh -c 'echo 3 >/proc/sys/vm/drop_caches' && free > /dev/null",get_pty=True)
        if self.server == "127.0.0.1" :
            stdin.write(f'{self.ssh_password}\n')
        out, err = stdout.read(), stderr.read()
        if err:
            print(err)
        stdin, stdout, stderr = ssh.exec_command(
            f"{self.pg_command_ctrl} start;",get_pty=True)

        out, err = stdout.read(), stderr.read()
        if err:
            print(err)
        ssh.close()

        self.init_db(self.db_name)

    def alter_system(self,command_list):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=self.server, port=self.ssh_port,
                    username=self.ssh_username, password=self.ssh_password,timeout=60)
        for command in command_list:
            stdin, stdout, stderr = ssh.exec_command(
                f"{command}",get_pty=True)
            out, err = stdout.read(), stderr.read()
            if err:
                print(err)
        stdin, stdout, stderr = ssh.exec_command(
            f"{self.pg_command_ctrl} restart;",get_pty=True)

        out, err = stdout.read(), stderr.read()
        if err:
            print(err)
        ssh.close()

        self.init_db(self.db_name)

    def init_db(self, db_name):
        db = self.db_url.format(db_name)
        PG = Postgres(db)
        self.db = PG
        return PG

    def disable_parallel(self):
        self.execute(
            'LOAD \'pg_hint_plan\';SET max_parallel_workers_per_gather=0;SET max_parallel_workers=0;', set_env=True)

    def explain(self, query, timeout=0,execute = False):
        if self.count>1 and self.count%20 == 0:
            time.sleep(30)
            self.count = 0
        if execute:
            self.count += 1
        if not execute:
            query = query.replace("ANALYZE,","")
        q = self.db.explain(query, timeout=timeout)
        if q is None or q == []:
            return
        return q[0][0][0]

    def execute(self, query, set_env=False):
        res = self.db.execute(query, set_env=set_env)
        return res

    def initial_tunning_knobs(self,knobs):
        self.knobs = knobs
        self.ordered_knob_list = self.knobs.names()

    def set_knob_value(self,name,val):
        knob = self.knobs[name]
        try:
            self.db.execute("set %s=%s;"%(name,knob.to_string(val)),set_env=True)
        except Exception as e:
            print(e)

    def set_knob_to_default(self,name):
        self.set_knob_value(name,self.knobs[name].userset)

    def get_knob_value(self,name):
        sql = "SELECT setting FROM pg_settings WHERE name = '{}';".format(name)
        try:
            value = self.execute(sql)[0][0]
        except:
            time.sleep(3)
            self.init_db(self.db_name)
            value = self.execute(sql)[0][0]
        return value

    def discard_session(self):
        self.db.discard_session_state()
