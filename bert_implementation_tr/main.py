# şimdilik main isminde sonradan düzenlemeler yapılacak

"""

"""
import os 
print(__file__)
print("---------------------------")
print(os.path.abspath(__file__))
print("---------------------------")
print(os.path.dirname(__file__))


# shallow copy
fruits = ['orange', 'apple', 'pear', 'banana', ['kiwi', 'apple', 'banana']]
a = fruits.copy()

print(fruits)
print(a)

a[0] = "domat"

print("-"*10)
print(fruits)
print(a)

fruits[0] = "hıyar"

print("-"*10)
print(fruits)
print(a)

# shallow copy effect!
fruits[4][0] = "illuminati is real"

print("-"*10)
print(fruits)
print(a)

print(id(fruits), id(a))


string1, string2, string3 = '', 'THammer Dance', 'aaaa'
non_null = string1 or string2 or string3
print(non_null)