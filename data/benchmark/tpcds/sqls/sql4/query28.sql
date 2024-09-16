-- start query 1 in stream 0 using template query28.tpl
select  *
from (select avg(ss_list_price) B1_LP
            ,count(ss_list_price) B1_CNT
            ,count(distinct ss_list_price) B1_CNTD
      from store_sales
      where ss_quantity between 0 and 5
        and (ss_list_price between 19 and 19+10 
             or ss_coupon_amt between 4568 and 4568+1000
             or ss_wholesale_cost between 80 and 80+20)) B1,
     (select avg(ss_list_price) B2_LP
            ,count(ss_list_price) B2_CNT
            ,count(distinct ss_list_price) B2_CNTD
      from store_sales
      where ss_quantity between 6 and 10
        and (ss_list_price between 140 and 140+10
          or ss_coupon_amt between 4876 and 4876+1000
          or ss_wholesale_cost between 2 and 2+20)) B2,
     (select avg(ss_list_price) B3_LP
            ,count(ss_list_price) B3_CNT
            ,count(distinct ss_list_price) B3_CNTD
      from store_sales
      where ss_quantity between 11 and 15
        and (ss_list_price between 147 and 147+10
          or ss_coupon_amt between 8058 and 8058+1000
          or ss_wholesale_cost between 22 and 22+20)) B3,
     (select avg(ss_list_price) B4_LP
            ,count(ss_list_price) B4_CNT
            ,count(distinct ss_list_price) B4_CNTD
      from store_sales
      where ss_quantity between 16 and 20
        and (ss_list_price between 16 and 16+10
          or ss_coupon_amt between 10702 and 10702+1000
          or ss_wholesale_cost between 17 and 17+20)) B4,
     (select avg(ss_list_price) B5_LP
            ,count(ss_list_price) B5_CNT
            ,count(distinct ss_list_price) B5_CNTD
      from store_sales
      where ss_quantity between 21 and 25
        and (ss_list_price between 120 and 120+10
          or ss_coupon_amt between 8931 and 8931+1000
          or ss_wholesale_cost between 64 and 64+20)) B5,
     (select avg(ss_list_price) B6_LP
            ,count(ss_list_price) B6_CNT
            ,count(distinct ss_list_price) B6_CNTD
      from store_sales
      where ss_quantity between 26 and 30
        and (ss_list_price between 12 and 12+10
          or ss_coupon_amt between 13351 and 13351+1000
          or ss_wholesale_cost between 14 and 14+20)) B6
limit 100;

-- end query 1 in stream 0 using template query28.tpl
