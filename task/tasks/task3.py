a=10
b=10.5
c="python"
d=[10,20,30]
e={"name":"arunraj"}
f={10,20,30}
g=(10,20,30)

print(type(a))
print(type(b))
print(type(c))
print(type(d))
print(type(e))
print(type(f))
print(type(g))


sales=[12000,0,8500,15000,-200,9800]

total=0
for amount in sales:
    if amount<= 0:
        print("skipped invalid value",amount)
        continue
    total+=amount
    print("added:",amount,"tatal:",total)

    if total > 30000:
        break
if total > 30000:
    print("target reached.stoped at:",total)
elif total == 30000:
    print("target reached successfully:",total)
else:
    print("target unreached,total:",total) 
