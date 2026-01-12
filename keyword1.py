age=input("enter your age:")
age=int(age)
if age<0:
    print("please enter a valid age")
elif age<18:
    print("you are a minor")
elif age>=18 and age<65:
    print("you are an adult")
else:
    print("you are a sinor citizen")

x=6
print(type(x))
x='Hello'
print(type(x))

s="10010"
c=int(s,2)
print("after converting to integer base 2:",end="")
print(c)
e=float(s)
print("after convertin to float:",end="")
print(e)


import keyword
print("the list of keywords are:")
print(keyword.kwlist) 


expr=10+20*30
print(expr)


