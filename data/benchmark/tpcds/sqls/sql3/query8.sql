-- start query 1 in stream 0 using template query8.tpl
select  s_store_name
      ,sum(ss_net_profit)
 from store_sales
     ,date_dim
     ,store,
     (select ca_zip
     from (
      SELECT substr(ca_zip,1,5) ca_zip
      FROM customer_address
      WHERE substr(ca_zip,1,5) IN (
                          '20041','85748','75465','20543','16615','73817',
                          '62633','62160','14816','32504','37769',
                          '44474','89203','11175','34039','79756',
                          '89688','89899','99489','51081','87466',
                          '32214','43711','44178','23362','75034',
                          '19298','47188','51802','17102','49998',
                          '12105','20537','67802','30294','12064',
                          '19291','77442','26530','12817','37392',
                          '40339','29620','30784','28197','67899',
                          '32038','27949','59774','30378','54008',
                          '25414','60125','34606','63973','53306',
                          '44080','31541','55213','42710','75873',
                          '12555','40634','34073','76110','36913',
                          '67758','92225','26913','93788','17248',
                          '13297','35906','40717','15006','45796',
                          '19469','11739','50982','10733','87848',
                          '10283','40001','72210','48179','20139',
                          '23022','13031','21060','65601','16466',
                          '73414','74869','18699','66610','53780',
                          '96502','79306','91589','73624','83753',
                          '91045','51614','40491','73412','52137',
                          '39893','29621','59489','56656','57534',
                          '11400','94249','39944','57259','56795',
                          '35562','41198','86903','41237','52928',
                          '80672','61404','82067','54408','70993',
                          '15618','42022','13019','42887','67950',
                          '32449','30719','82263','55117','57817',
                          '93439','84137','57367','51410','42970',
                          '59069','18664','80581','20639','11481',
                          '11948','72171','92006','45641','41337',
                          '72205','46460','25958','88119','43088',
                          '61636','14158','24496','45883','28370',
                          '44872','33793','50309','25715','66502',
                          '27289','33533','89805','79357','85679',
                          '77302','12461','48768','56443','15124',
                          '30641','31926','25830','35975','39462',
                          '13935','45763','43828','74095','17156',
                          '33195','56035','39885','35126','52716',
                          '95851','31136','24342','49054','86956',
                          '70833','24340','58160','35163','37727',
                          '14592','37458','83895','72094','16803',
                          '23891','76453','36619','33318','64497',
                          '72242','24194','13538','51457','26884',
                          '67240','94140','18875','41348','39359',
                          '21958','20841','28391','50896','98671',
                          '18256','20700','71669','41610','23839',
                          '24387','26075','96953','27619','88411',
                          '44222','32592','87609','46509','61007',
                          '55263','61358','80347','75802','20507',
                          '72735','70099','67174','23347','63533',
                          '57397','49872','59949','60448','90901',
                          '98316','32018','29594','89080','68964',
                          '43032','17600','35772','57038','21541',
                          '90167','34781','60212','20989','98007',
                          '15282','88954','22769','20649','46491',
                          '34195','45819','93004','37991','75929',
                          '12084','40795','27049','62543','47850',
                          '84245','28552','49616','63098','17334',
                          '85388','79878','34027','63802','13054',
                          '98113','92320','24018','67641','22866',
                          '55723','36597','54714','26201','17686',
                          '86111','21198','31755','48997','17172',
                          '94703','39792','42855','43652','47710',
                          '53866','15134','99988','90689','58568',
                          '34617','31162','71774','20784','12640',
                          '51675','67217','94447','42118','76457',
                          '35578','42985','98266','22060','36733',
                          '18331','30337','20706','51842','32359',
                          '38583','16219','69115','74028','25252',
                          '60933','84251','25106','19695','31823',
                          '76696','84606','50434','34844','65255',
                          '58734','43948','27541','32984','57588',
                          '77314','79018','91945','42917','56367',
                          '93242','13560','60400','53681','81242',
                          '73234','97943','48913','70867','63084',
                          '47327','32506','84609','10080','27656',
                          '50180','70603','53069','51851','17182',
                          '11794','13509','81701','96267','58679',
                          '28514','12616','78591','50620','19993',
                          '36341','68478','57931','22842')
     intersect
      select ca_zip
      from (SELECT substr(ca_zip,1,5) ca_zip,count(*) cnt
            FROM customer_address, customer
            WHERE ca_address_sk = c_current_addr_sk and
                  c_preferred_cust_flag='Y'
            group by ca_zip
            having count(*) > 10)A1)A2) V1
 where ss_store_sk = s_store_sk
  and ss_sold_date_sk = d_date_sk
  and d_qoy = 1 and d_year = 2002
  and (substr(s_zip,1,2) = substr(V1.ca_zip,1,2))
 group by s_store_name
 order by s_store_name
 limit 100;

-- end query 1 in stream 0 using template query8.tpl
