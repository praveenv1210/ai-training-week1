import csv
from importlib.resources import path
from pathlib import Path 
def calculate_bonus(salary,rate=0.1,*extras,**options):
    bonus = (salary * rate)+ sum(extras)
    if "cap" in options:
        bonus = min(bonus, options["cap"])
    return bonus
def log_call(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} returned: {result}")
        return result
    return wrapper
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
    def get_details(self):
        return f"Employee Name: {self.name}, Salary: {self.salary}-team Size: {self.team_size}"
    @property
    def annual_salary(self):
        return self._salary * 12        
class Manager(Employee):
    def __init__(self, name, salary, team_size):
        super().__init__(name, salary)
        self.team_size = team_size
        @log_call
        def get_details(self):

         return f"Manager Name: {self.name}, Salary: {self.salary}-Team Size: {self.team_size}"


employees=[]
file_path=Path(__file__).parent / "employees.csv"
with open(file_path, 'r', newline="") as file: 
     reader = csv.DictReader(file)

     for row in reader:
        employee={
                "name": row['name'],
                "department": row['department'],
                "salary": int(row['salary'])
                
            }      
        employees.append(employee)

for emp in employees:
   
    salary = emp["salary"]
    if salary < 50000:
        category = "Low salary"
    elif salary < 70000:
        category = "Medium salary"
    else:
        category = "High salary"
        
    print(f"{emp['name']}-{category}")    



manager = Manager("divya", 75000, 5)   
print(manager.get_details())

bonus = calculate_bonus(30000,0.1,500,1000, cap=4000)         
        