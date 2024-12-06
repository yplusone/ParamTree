# IMDB
## get data

```
wget http://homepages.cwi.nl/~boncz/job/imdb.tgz
tar -xvzf imdb.tgz -C imdb-datasets-ftp/
rm -f imdb.tgz
```

```
psql -c "DROP DATABASE IF EXISTS imdb"

psql -c "CREATE DATABASE imdb"
```

```sql
CREATE TABLE aka_name (  
id integer NOT NULL PRIMARY KEY,  
person_id integer NOT NULL,  
name character varying,  
imdb_index character varying(3),  
name_pcode_cf character varying(11),  
name_pcode_nf character varying(11),  
surname_pcode character varying(11),  
md5sum character varying(65)  
);  
  
CREATE TABLE aka_title (  
id integer NOT NULL PRIMARY KEY,  
movie_id integer NOT NULL,  
title character varying,  
imdb_index character varying(4),  
kind_id integer NOT NULL,  
production_year integer,  
phonetic_code character varying(5),  
episode_of_id integer,  
season_nr integer,  
episode_nr integer,  
note character varying(72),  
md5sum character varying(32)  
);  
  
CREATE TABLE cast_info (  
id integer NOT NULL PRIMARY KEY,  
person_id integer NOT NULL,  
movie_id integer NOT NULL,  
person_role_id integer,  
note character varying,  
nr_order integer,  
role_id integer NOT NULL  
);  
  
CREATE TABLE char_name (  
id integer NOT NULL PRIMARY KEY,  
name character varying NOT NULL,  
imdb_index character varying(2),  
imdb_id integer,  
name_pcode_nf character varying(5),  
surname_pcode character varying(5),  
md5sum character varying(32)  
);  
  
CREATE TABLE comp_cast_type (  
id integer NOT NULL PRIMARY KEY,  
kind character varying(32) NOT NULL  
);  
  
CREATE TABLE company_name (  
id integer NOT NULL PRIMARY KEY,  
name character varying NOT NULL,  
country_code character varying(6),  
imdb_id integer,  
name_pcode_nf character varying(5),  
name_pcode_sf character varying(5),  
md5sum character varying(32)  
);  
  
CREATE TABLE company_type (  
id integer NOT NULL PRIMARY KEY,  
kind character varying(32)  
);  
  
CREATE TABLE complete_cast (  
id integer NOT NULL PRIMARY KEY,  
movie_id integer,  
subject_id integer NOT NULL,  
status_id integer NOT NULL  
);  
  
CREATE TABLE info_type (  
id integer NOT NULL PRIMARY KEY,  
info character varying(32) NOT NULL  
);  
  
CREATE TABLE keyword (  
id integer NOT NULL PRIMARY KEY,  
keyword character varying NOT NULL,  
phonetic_code character varying(5)  
);  
  
CREATE TABLE kind_type (  
id integer NOT NULL PRIMARY KEY,  
kind character varying(15)  
);  
  
CREATE TABLE link_type (  
id integer NOT NULL PRIMARY KEY,  
link character varying(32) NOT NULL  
);  
  
CREATE TABLE movie_companies (  
id integer NOT NULL PRIMARY KEY,  
movie_id integer NOT NULL,  
company_id integer NOT NULL,  
company_type_id integer NOT NULL,  
note character varying  
);  
  
CREATE TABLE movie_info_idx (  
id integer NOT NULL PRIMARY KEY,  
movie_id integer NOT NULL,  
info_type_id integer NOT NULL,  
info character varying NOT NULL,  
note character varying(1)  
);  
  
CREATE TABLE movie_keyword (  
id integer NOT NULL PRIMARY KEY,  
movie_id integer NOT NULL,  
keyword_id integer NOT NULL  
);  
  
CREATE TABLE movie_link (  
id integer NOT NULL PRIMARY KEY,  
movie_id integer NOT NULL,  
linked_movie_id integer NOT NULL,  
link_type_id integer NOT NULL  
);  
  
CREATE TABLE name (  
id integer NOT NULL PRIMARY KEY,  
name character varying NOT NULL,  
imdb_index character varying(9),  
imdb_id integer,  
gender character varying(1),  
name_pcode_cf character varying(5),  
name_pcode_nf character varying(5),  
surname_pcode character varying(5),  
md5sum character varying(32)  
);  
  
CREATE TABLE role_type (  
id integer NOT NULL PRIMARY KEY,  
role character varying(32) NOT NULL  
);  
  
CREATE TABLE title (  
id integer NOT NULL PRIMARY KEY,  
title character varying NOT NULL,  
imdb_index character varying(5),  
kind_id integer NOT NULL,  
production_year integer,  
imdb_id integer,  
phonetic_code character varying(5),  
episode_of_id integer,  
season_nr integer,  
episode_nr integer,  
series_years character varying(49),  
md5sum character varying(32)  
);  
  
CREATE TABLE movie_info (  
id integer NOT NULL PRIMARY KEY,  
movie_id integer NOT NULL,  
info_type_id integer NOT NULL,  
info character varying NOT NULL,  
note character varying  
);  
  
CREATE TABLE person_info (  
id integer NOT NULL PRIMARY KEY,  
person_id integer NOT NULL,  
info_type_id integer NOT NULL,  
info character varying NOT NULL,  
note character varying  
);
```

## import data

```sql
copy aka_name from './imdb/aka_name.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY aka_title FROM './imdb/aka_title.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY cast_info FROM './imdb/cast_info.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY char_name FROM './imdb/char_name.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY comp_cast_type FROM './imdb/comp_cast_type.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY company_name FROM './imdb/company_name.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY company_type FROM './imdb/company_type.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY complete_cast FROM './imdb/complete_cast.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY info_type FROM './imdb/info_type.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY keyword FROM './imdb/keyword.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY kind_type FROM './imdb/kind_type.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY link_type FROM './imdb/link_type.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY movie_companies FROM './imdb/movie_companies.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY movie_info FROM './imdb/movie_info.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY movie_info_idx FROM './imdb/movie_info_idx.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY movie_keyword FROM './imdb/movie_keyword.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY movie_link FROM './imdb/movie_link.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY name FROM './imdb/name.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY person_info FROM './imdb/person_info.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY role_type FROM './imdb/role_type.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY title FROM './imdb/title.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');


```

## Create Index
```sql
create index company_id_movie_companies on movie_companies(company_id);

create index company_type_id_movie_companies on movie_companies(company_type_id);

create index info_type_id_movie_info_idx on movie_info_idx(info_type_id);

create index info_type_id_movie_info on movie_info(info_type_id);

create index info_type_id_person_info on person_info(info_type_id);

create index keyword_id_movie_keyword on movie_keyword(keyword_id);

create index kind_id_aka_title on aka_title(kind_id);

create index kind_id_title on title(kind_id);

create index linked_movie_id_movie_link on movie_link(linked_movie_id);

create index link_type_id_movie_link on movie_link(link_type_id);

create index movie_id_aka_title on aka_title(movie_id);

create index movie_id_cast_info on cast_info(movie_id);

create index movie_id_complete_cast on complete_cast(movie_id);

create index movie_id_movie_companies on movie_companies(movie_id);

create index movie_id_movie_info_idx on movie_info_idx(movie_id);

create index movie_id_movie_keyword on movie_keyword(movie_id);

create index movie_id_movie_link on movie_link(movie_id);

create index movie_id_movie_info on movie_info(movie_id);

create index person_id_aka_name on aka_name(person_id);

create index person_id_cast_info on cast_info(person_id);

create index person_id_person_info on person_info(person_id);

create index person_role_id_cast_info on cast_info(person_role_id);

create index role_id_cast_info on cast_info(role_id);
```

# TPCH

## get data
```sh
git clone https://github.com/electrum/tpch-dbgen.git
cd tpch-dbgen/
make
./dbgen -s 1 -f   
ls *.tbl

```

## Create Table
```sql
CREATE TABLE NATION  ( N_NATIONKEY  INTEGER NOT NULL,
                            N_NAME       CHAR(25) NOT NULL,
                            N_REGIONKEY  INTEGER NOT NULL,
                            N_COMMENT    VARCHAR(152));

CREATE TABLE REGION  ( R_REGIONKEY  INTEGER NOT NULL,
                            R_NAME       CHAR(25) NOT NULL,
                            R_COMMENT    VARCHAR(152));

CREATE TABLE PART  ( P_PARTKEY     INTEGER NOT NULL,
                          P_NAME        VARCHAR(55) NOT NULL,
                          P_MFGR        CHAR(25) NOT NULL,
                          P_BRAND       CHAR(10) NOT NULL,
                          P_TYPE        VARCHAR(25) NOT NULL,
                          P_SIZE        INTEGER NOT NULL,
                          P_CONTAINER   CHAR(10) NOT NULL,
                          P_RETAILPRICE DECIMAL(15,2) NOT NULL,
                          P_COMMENT     VARCHAR(23) NOT NULL );

CREATE TABLE SUPPLIER ( S_SUPPKEY     INTEGER NOT NULL,
                             S_NAME        CHAR(25) NOT NULL,
                             S_ADDRESS     VARCHAR(40) NOT NULL,
                             S_NATIONKEY   INTEGER NOT NULL,
                             S_PHONE       CHAR(15) NOT NULL,
                             S_ACCTBAL     DECIMAL(15,2) NOT NULL,
                             S_COMMENT     VARCHAR(101) NOT NULL);

CREATE TABLE PARTSUPP ( PS_PARTKEY     INTEGER NOT NULL,
                             PS_SUPPKEY     INTEGER NOT NULL,
                             PS_AVAILQTY    INTEGER NOT NULL,
                             PS_SUPPLYCOST  DECIMAL(15,2)  NOT NULL,
                             PS_COMMENT     VARCHAR(199) NOT NULL );

CREATE TABLE CUSTOMER ( C_CUSTKEY     INTEGER NOT NULL,
                             C_NAME        VARCHAR(25) NOT NULL,
                             C_ADDRESS     VARCHAR(40) NOT NULL,
                             C_NATIONKEY   INTEGER NOT NULL,
                             C_PHONE       CHAR(15) NOT NULL,
                             C_ACCTBAL     DECIMAL(15,2)   NOT NULL,
                             C_MKTSEGMENT  CHAR(10) NOT NULL,
                             C_COMMENT     VARCHAR(117) NOT NULL);

CREATE TABLE ORDERS  ( O_ORDERKEY       INTEGER NOT NULL,
                           O_CUSTKEY        INTEGER NOT NULL,
                           O_ORDERSTATUS    CHAR(1) NOT NULL,
                           O_TOTALPRICE     DECIMAL(15,2) NOT NULL,
                           O_ORDERDATE      DATE NOT NULL,
                           O_ORDERPRIORITY  CHAR(15) NOT NULL,  
                           O_CLERK          CHAR(15) NOT NULL, 
                           O_SHIPPRIORITY   INTEGER NOT NULL,
                           O_COMMENT        VARCHAR(79) NOT NULL);

CREATE TABLE LINEITEM ( L_ORDERKEY    INTEGER NOT NULL,
                             L_PARTKEY     INTEGER NOT NULL,
                             L_SUPPKEY     INTEGER NOT NULL,
                             L_LINENUMBER  INTEGER NOT NULL,
                             L_QUANTITY    DECIMAL(15,2) NOT NULL,
                             L_EXTENDEDPRICE  DECIMAL(15,2) NOT NULL,
                             L_DISCOUNT    DECIMAL(15,2) NOT NULL,
                             L_TAX         DECIMAL(15,2) NOT NULL,
                             L_RETURNFLAG  CHAR(1) NOT NULL,
                             L_LINESTATUS  CHAR(1) NOT NULL,
                             L_SHIPDATE    DATE NOT NULL,
                             L_COMMITDATE  DATE NOT NULL,
                             L_RECEIPTDATE DATE NOT NULL,
                             L_SHIPINSTRUCT CHAR(25) NOT NULL,
                             L_SHIPMODE     CHAR(10) NOT NULL,
                             L_COMMENT      VARCHAR(44) NOT NULL);

create extension pg_hint_plan;
create extension pageinspect;
su postgres


for i in `ls *.tbl`; do
    echo $i;
    sed -i 's/|$//' *.tbl;
    name=`echo $i| cut -d'.' -f1`;
    psql -h 127.0.0.1 -p 5432 -d tpch -c "COPY $name FROM '`pwd`/$i' DELIMITER '|' ENCODING 'LATIN1';";
done


\Copy region FROM '/home/usr/tpch-dbgen/region.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
\Copy nation FROM '/home/usr/tpch-dbgen/nation.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
\Copy part FROM '/home/usr/tpch-dbgen/part.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
\Copy supplier FROM '/home/usr/tpch-dbgen/supplier.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
\Copy customer FROM '/home/usr/tpch-dbgen/customer.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
\Copy lineitem FROM '/home/usr/tpch-dbgen/lineitem.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
\Copy partsupp FROM '/home/usr/tpch-dbgen/partsupp.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
\Copy orders FROM '/home/usr/tpch-dbgen/orders.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
```

## constraint
```sql
alter table PART add primary key(P_PARTKEY);

alter table SUPPLIER add primary key(S_SUPPKEY);

alter table PARTSUPP add primary key(PS_PARTKEY, PS_SUPPKEY);

alter table CUSTOMER add primary key(C_CUSTKEY);

alter table ORDERS add primary key(O_ORDERKEY);

alter table LINEITEM add primary key(L_ORDERKEY, L_LINENUMBER);

alter table NATION add primary key(N_NATIONKEY);

alter table REGION add primary key(R_REGIONKEY);

alter table SUPPLIER add CONSTRAINT f1 foreign key (S_NATIONKEY) references NATION(N_NATIONKEY);

alter table PARTSUPP add CONSTRAINT f2 foreign key (PS_PARTKEY) references PART(P_PARTKEY);

alter table PARTSUPP add CONSTRAINT f3 foreign key (PS_SUPPKEY) references SUPPLIER(S_SUPPKEY);

alter table CUSTOMER add CONSTRAINT f4 foreign key (C_NATIONKEY) references NATION(N_NATIONKEY);

alter table ORDERS add CONSTRAINT f5 foreign key (O_CUSTKEY) references CUSTOMER(C_CUSTKEY);

alter table LINEITEM add CONSTRAINT f6 foreign key (L_ORDERKEY) references ORDERS(O_ORDERKEY);

alter table LINEITEM add CONSTRAINT f7 foreign key (L_PARTKEY) references PART(P_PARTKEY);

alter table LINEITEM add CONSTRAINT f8 foreign key (L_SUPPKEY) references SUPPLIER(S_SUPPKEY);

alter table LINEITEM add CONSTRAINT f9 foreign key (L_PARTKEY, L_SUPPKEY) references PARTSUPP(PS_PARTKEY, PS_SUPPKEY);

alter table NATION add CONSTRAINT f10 foreign key (N_REGIONKEY) references REGION(R_REGIONKEY);
```

## Create Index

```sql
CREATE INDEX IDX_SUPPLIER_NATION_KEY ON SUPPLIER (S_NATIONKEY);

CREATE INDEX IDX_PARTSUPP_PARTKEY ON PARTSUPP (PS_PARTKEY);
CREATE INDEX IDX_PARTSUPP_SUPPKEY ON PARTSUPP (PS_SUPPKEY);

CREATE INDEX IDX_CUSTOMER_NATIONKEY ON CUSTOMER (C_NATIONKEY);

CREATE INDEX IDX_ORDERS_CUSTKEY ON ORDERS (O_CUSTKEY);

CREATE INDEX IDX_LINEITEM_ORDERKEY ON LINEITEM (L_ORDERKEY);
CREATE INDEX IDX_LINEITEM_PART_SUPP ON LINEITEM (L_PARTKEY,L_SUPPKEY);

CREATE INDEX IDX_NATION_REGIONKEY ON NATION (N_REGIONKEY);


-- aditional indexes

CREATE INDEX IDX_LINEITEM_SHIPDATE ON LINEITEM (L_SHIPDATE, L_DISCOUNT, L_QUANTITY);

CREATE INDEX IDX_ORDERS_ORDERDATE ON ORDERS (O_ORDERDATE);
```

# TPCDS

```sh
git clone <https://github.com/gregrahn/tpcds-kit.git>
cd tpcds-kit/tools
make OS=LINUX

create data tpcds
psql tpcds -f tpcds.sql

./dsdgen  --help

mkdir data
./dsdgen  -scale 1 -dir ./data/

cd data

#Import data
for i in `ls *.dat`; do
  table=${i/.dat/}
  echo "Loading $table..."
  sed 's/|$//' $i > /tmp/$i
  psql -h 127.0.0.1 tpcds -q -c "TRUNCATE $table"
  psql -h 127.0.0.1 tpcds -c "\\copy $table FROM '/tmp/$i' CSV DELIMITER '|'"
done

# Generate Queries
#!/bin/sh
for i in `seq 1 99`
do
./dsqgen  -DIRECTORY ../query_templates/ -TEMPLATE "query${i}.tpl" -DIALECT netezza -FILTER Y > ../sql/query${i}.sql
done
```

```sql
Create Index
CREATE INDEX c_customer_sk_idx ON customer(c_customer_sk);
CREATE INDEX d_date_sk_idx ON date_dim(d_date_sk);
CREATE INDEX d_date_idx ON date_dim(d_date);
CREATE INDEX d_month_seq_idx ON date_dim(d_month_seq);
CREATE INDEX d_year_idx ON date_dim(d_year);
CREATE INDEX i_item_sk_idx ON item(i_item_sk);
CREATE INDEX s_state_idx ON store(s_state);
CREATE INDEX s_store_sk_idx ON store(s_store_sk);
CREATE INDEX sr_returned_date_sk_idx ON store_returns(sr_returned_date_sk);
CREATE INDEX ss_sold_date_sk_idx ON store_sales(ss_sold_date_sk);

```