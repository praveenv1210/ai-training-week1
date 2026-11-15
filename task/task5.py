import pandas as pd
import numpy as np


employee="""
    id,name,dept,salary,city
    1,arun,sales,30000,chennai
    2,priya,hr,45000,chennai
    3,rahul,it,60000,bangalore
    4,divya,it,75000,chennai
    5,karthick,finance,90000,mumbai
    6,meena,sales,52000,bangalore                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    7,suresh,it,68000,chennai
    8,anitha,hr,48000,mumbai"""
    
with open("employee1.csv","w")as f:
    f.write(employee)
employee=pd.read_csv('employee1.csv')
print(employee.head())
print(employee.info())
print(employee.describe())

name=employee["name"]
name_salary=employee[["name","salary"]]

chennai_employee=employee.loc[employee["city"]=="chennai","name"].values
print("\nchennai:",",".join(chennai_employee))

sal=(employee["salary"]>50000).sum()
emp=employee.loc[employee["dept"]=="it","name"].values
print("salary above 50000:",sal,"employee")
print("it department:",",".join(emp))