a=10
b=10.5
c="python"
d=[10,20,30]
e={"Name":"John","Age":30}
f={10,20,30}
g=(10,20,30)   

print("data types")
print(a,type(a))
print(b,type(b))
print(c,type(c))
print(d,type(d))
print(e,type(e))
print(f,type(f))
print(type(g))

sales=[12000,0,8500,15000,-200,9800]
total=0

for sale in sales:
    if sale <=0:
        print(f"skipped invalid value : {sale}")
        continue
    total += sale
    print(f"added {sale} total {total}")
    if total  > 30000:
        break
if total > 30000:
    print(f"Target reached.stopped at {total}")
else:
     print(f"Target not reached.final total: {total}")