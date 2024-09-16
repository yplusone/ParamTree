-- start query 1 in stream 0 using template query41.tpl
select  distinct(i_product_name)
 from item i1
 where i_manufact_id between 868 and 868+40 
   and (select count(*) as item_cnt
        from item
        where (i_manufact = i1.i_manufact and
        ((i_category = 'Women' and 
        (i_color = 'black' or i_color = 'coral') and 
        (i_units = 'Oz' or i_units = 'Gross') and
        (i_size = 'large' or i_size = 'extra large')
        ) or
        (i_category = 'Women' and
        (i_color = 'rose' or i_color = 'orchid') and
        (i_units = 'Box' or i_units = 'Tsp') and
        (i_size = 'N/A' or i_size = 'economy')
        ) or
        (i_category = 'Men' and
        (i_color = 'linen' or i_color = 'pink') and
        (i_units = 'Ton' or i_units = 'Bundle') and
        (i_size = 'petite' or i_size = 'medium')
        ) or
        (i_category = 'Men' and
        (i_color = 'tomato' or i_color = 'brown') and
        (i_units = 'Tbl' or i_units = 'Bunch') and
        (i_size = 'large' or i_size = 'extra large')
        ))) or
       (i_manufact = i1.i_manufact and
        ((i_category = 'Women' and 
        (i_color = 'seashell' or i_color = 'dim') and 
        (i_units = 'Ounce' or i_units = 'Dram') and
        (i_size = 'large' or i_size = 'extra large')
        ) or
        (i_category = 'Women' and
        (i_color = 'salmon' or i_color = 'mint') and
        (i_units = 'Case' or i_units = 'Pound') and
        (i_size = 'N/A' or i_size = 'economy')
        ) or
        (i_category = 'Men' and
        (i_color = 'blue' or i_color = 'goldenrod') and
        (i_units = 'Cup' or i_units = 'Pallet') and
        (i_size = 'petite' or i_size = 'medium')
        ) or
        (i_category = 'Men' and
        (i_color = 'magenta' or i_color = 'smoke') and
        (i_units = 'Gram' or i_units = 'Dozen') and
        (i_size = 'large' or i_size = 'extra large')
        )))) > 0
 order by i_product_name
 limit 100;

-- end query 1 in stream 0 using template query41.tpl
