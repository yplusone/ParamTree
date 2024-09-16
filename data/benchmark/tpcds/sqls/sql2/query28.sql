-- start query 1 in stream 0 using template query28.tpl
select  *
from (select avg(ss_list_price) B1_LP
            ,count(ss_list_price) B1_CNT
            ,count(distinct ss_list_price) B1_CNTD
      from store_sales
      where ss_quantity between 0 and 5
        and (ss_list_price between 10 and 10+10 
             or ss_coupon_amt between 10692 and 10692+1000
             or ss_wholesale_cost between 36 and 36+20)) B1,
     (select avg(ss_list_price) B2_LP
            ,count(ss_list_price) B2_CNT
            ,count(distinct ss_list_price) B2_CNTD
      from store_sales
      where ss_quantity between 6 and 10
        and (ss_list_price between 132 and 132+10
          or ss_coupon_amt between 10820 and 10820+1000
          or ss_wholesale_cost between 33 and 33+20)) B2,
     (select avg(ss_list_price) B3_LP
            ,count(ss_list_price) B3_CNT
            ,count(distinct ss_list_price) B3_CNTD
      from store_sales
      where ss_quantity between 11 and 15
        and (ss_list_price between 187 and 187+10
          or ss_coupon_amt between 6100 and 6100+1000
          or ss_wholesale_cost between 34 and 34+20)) B3,
     (select avg(ss_list_price) B4_LP
            ,count(ss_list_price) B4_CNT
            ,count(distinct ss_list_price) B4_CNTD
      from store_sales
      where ss_quantity between 16 and 20
        and (ss_list_price between 104 and 104+10
          or ss_coupon_amt between 5526 and 5526+1000
          or ss_wholesale_cost between 41 and 41+20)) B4,
     (select avg(ss_list_price) B5_LP
            ,count(ss_list_price) B5_CNT
            ,count(distinct ss_list_price) B5_CNTD
      from store_sales
      where ss_quantity between 21 and 25
        and (ss_list_price between 113 and 113+10
          or ss_coupon_amt between 5619 and 5619+1000
          or ss_wholesale_cost between 74 and 74+20)) B5,
     (select avg(ss_list_price) B6_LP
            ,count(ss_list_price) B6_CNT
            ,count(distinct ss_list_price) B6_CNTD
      from store_sales
      where ss_quantity between 26 and 30
        and (ss_list_price between 118 and 118+10
          or ss_coupon_amt between 4202 and 4202+1000
          or ss_wholesale_cost between 27 and 27+20)) B6
limit 100;

-- end query 1 in stream 0 using template query28.tpl
