import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("charts",exist_ok=True)

df=pd.read_csv("employee1.csv")
print(df)
df.columns=df.columns.str.strip().str.lower()
print(df.columns.tolist())
# def find_column(possible_columns,columns):
#     for column in possible_columns:
#         if column in columns:
#             return column
#     raise ValueError(f" column not found: {possible_columns} "
#                      )
id_column='id'
salary_column='salary'
department_column='dept'
print(df.columns.tolist())
# df[salary_column]=pd.to_numeric(df[salary_column],errors='coerce')
# df[id_column]=pd.to_numeric(df[id_column],errors='coerce')
# df= df.dropna(subset=[id_column,salary_column,department_column])
os.makedirs("output",exist_ok=True)

avg_salary=df.groupby('dept')['salary'].mean()
plt.figure(figsize=(9,6))
bars=plt.bar(avg_salary.index,avg_salary.values)
plt.title("Average Salary by Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")
plt.bar_label(bars, fmt='%.2f')
plt.tight_layout()
plt.savefig("charts/bar_chart.png",dpi=300)

plt.show()


print("bar chart saved to charts")

plt.figure(figsize=(9,6))
plt.hist(df['salary'],bins=5,edgecolor='black')
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig("charts/histogram.png",dpi=300) 
plt.show()
plt.close()
print("histogram saved to charts")  

plt.figure(figsize=(9,6))
print(df.columns.tolist())
for department, group in df.groupby('dept'):
    print(department,group.columns.tolist())
    plt.scatter(group['id'],group['salary'],label=department,s=80)
plt.title("Salary vs Employee ID by Department")
plt.xlabel("Employee ID")   
plt.ylabel("Salary")

plt.legend(title="Department")
plt.grid(True)
plt.tight_layout()    
plt.savefig("charts/scatter_plot.png",dpi=300)
plt.show()  
plt.close()    
print("scatter plot saved to charts")

month=["Jan","Feb","Mar","Apr","May","Jun"]
sales=[12,15,14,18,22,25]
plt.figure(figsize=(9,6))
plt.plot(month,sales,marker='o',linestyle='-',label="monthly_Sales")
plt.title("Monthly Sales jan to jun")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("charts/line_chart.png",dpi=300)
plt.show()
print("line chart saved to charts")
