import pandas as pd
import numpy as np

order="""
order_id,customer_id,amount
1,101,500
2,102,
3,101,700
3,101,700
4,103,300
5,104,900
"""
with open("order1.csv","w")as f:
    f.write(order)
df=pd.read_csv("order1.csv")
print(df)
print("missing values:")
print(df.isna().sum())
print("duplicate rows:")
print(df.duplicated().sum())
df=df.drop_duplicates()
print("after removing duplicates:",df)
median_amount=df['amount'].median()
print("median amount:")
print(median_amount)
df['amount']=df['amount'].fillna(median_amount)
print("after filling missing values:",df)
df['amount']=pd.to_numeric(df['amount'])
df['order_id']=df['order_id'].astype(str)
print("data types:",df.dtypes)
result=df.groupby('customer_id').agg(
    total=('amount','sum'),orders=(
        'order_id','count'))
print("customer report:",result)