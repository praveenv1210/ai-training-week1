import pandas as pd
import numpy as np

customer="""
customer_id,customer_name,city
101,arun ,chennai
102,priya ,mumbai
103,rahul,chennai
105,divya,bangalore
"""
with open("customer.csv","w")as f:
    f.write(customer)
customer_df=pd.read_csv("customer.csv")
print(customer_df)
df=pd.read_csv("order1.csv")

inner_join=pd.merge(customer_df,df,on="customer_id",how="inner")

print(inner_join)
print("inner join rows:",len(inner_join))

left_join=pd.merge(customer_df,df,on="customer_id",how="left")

print(left_join)
print("left join rows:",len(left_join))


no_order=left_join[
    left_join['order_id'].isna()
    ]
print(no_order[['customer_id','customer_name','city']])

outer_join=pd.merge(df,customer_df,on="customer_id",how="outer",indicator=True)
print(outer_join)

unmatched_rows=outer_join[outer_join['_merge']=='left_only']
print(unmatched_rows[[ 'customer_id','customer_name','amount']])