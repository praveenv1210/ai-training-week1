price=[500,1200,800,2500,300]
def apply_discount(price, discount=10):
    return price - (price * discount / 100)
print("apply discount(1000)->",apply_discount(1000))
print("apply discount(1000,25)->",apply_discount(1000,25))

def total(*args):
    return sum(args)
print("total(5,10,15)->",total(5,10,15))

def describe_person(**kwargs):
    for key,value in kwargs.items():
        print(f"{key}:{value}")
above_1000=[price for price in price if price > 1000]
print("prices above 1000:", above_1000)

gst=list(map(lambda x: round(x*1.18,2),price))
print("prices with gst:", gst)

below_1000=list(filter(lambda x: x<1000,price))
print("prices below 1000:", below_1000)