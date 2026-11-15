class InsuffientFundsError(Exception):
  pass

class account:
  def __init__(self,name,balance,min_balance=500):
    self.name=name
    self.balance=balance


  def withdraw(self,amount):
    if amount > self.balance:
      raise InsuffientFundsError("insuffienit balance:",self.balance)
    self.balance -=amount



class Savingsaccount(account):
  def __init__(self, name, balance,min_balance):
    super().__init__(name, balance) 
    self.min_balance=min_balance



  def withdraw(self, amount):
    if self.balance-amount < self.min_balance:
      raise InsuffientFundsError("insuffientfunds Error:",self.balance)
    super().withdraw(amount)

priya =Savingsaccount("priya",2000,1200)
try:
  priya.withdraw(1200)
  print("withdrawal amount")
except InsuffientFundsError as e:
  print ("error:",e)     

try:
  priya.withdraw(800)
  print("priya withdrew 800 balance:",priya.balance)
except InsuffientFundsError as e:
  print ("error:",e)

print(isinstance(priya,account))
