class InsufficientFundsError(Exception):
    pass

class account:
    def __init__(self,owner,balance):
        self.owner=owner
        self.balance=balance

    def deposit(self,amount):

        self.balance += amount
        
        print(self.owner,"deposited amount")
      
        print("current amount:",self.balance)
        
        print("/////****************/////")

    def withdraw(self,amount):
        amount=(int(input("Enter withdrawal amount:")))
        
        if amount > self.balance:
         
            raise InsufficientFundsError("insuffient balance:",self.balance,"requested:",amount)   
        self.balance -= amount
      
        print("withdrawal successfully:",amount)
        print("///*****************///")

      
         
        print("balance amount:",self.balance) 


acc=account("arunraj",400000)
try:
    acc.deposit(500)
    acc.withdraw(50000)  
except InsufficientFundsError as e:
    print("error:",e)
finally:
    print("transaction successfully")    


       
