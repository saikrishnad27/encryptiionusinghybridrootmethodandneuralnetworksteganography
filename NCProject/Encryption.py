import sympy as sym
import math
import numpy as np6
import time
import os, psutil
from numpy.linalg import norm
start = time.time()


def key_A_gen(a, p ,g):   # Key generated by the sender A
  x = int(pow(g,a,p))
  return x

def key_B_gen(b, p, g):  # Key generated by the receiver B
  y = int(pow(g,b,p))
  return y

def secret_key_gen_A(a, b, p, g):
  y = key_B_gen(b,p,g)           # The key that A will get after key exchange
  secret_key = int(pow(y,a,p))   # w.r.t the sender A after key exchange
  return secret_key

def secret_key_gen_B(a, b, p, g):
  x = key_A_gen(a,p,g)           # The key that B will get after key exchange
  secret_key = int(pow(x,b,p))   # w.r.t the sender A after key exchange
  return secret_key

  # Both the persons will be agreed upon the public keys p and g
# Let p be a prime number
# Let g be a primitive root of p
p = 13  
g = 6

# The sender A is allocated a private key a 
# The receiver B is allocated a private key b
a = 4
b = 2

# Key generated by Sender A
key_A = key_A_gen(a, p, g)

# Key generated by Receiver B
key_B = key_B_gen(b, p, g)

# Now the keys generated are exchanged between each other
# Therefore, A will be key_B and B will get key_A

# Secret key generated by A
secret_key_A = secret_key_gen_A(a, b, p, g)

# Secret key generated by B
secret_key_B = secret_key_gen_B(a, b, p, g)

# Testing and outputing the secret_key
if(secret_key_A == secret_key_B):
  secret_key = secret_key_A
else:
  secret_key = 0
  print("Algorithm is wrong")


def func(x, secret_key, ascii):
    return secret_key * x * x * x + 4*x*x+secret_key*math.cos(x) - 2.7 * x - ascii


def derivfunc(x):
    return 3 * secret_key * x * x - 2.7 + 8*x +secret_key*math.sin(x)


def bisection(a, b, e, secret_key, ascii):
    if (func(a, secret_key, ascii) * func(b,secret_key, ascii) >= 0):
        return
    c = a
    iter=1
    while ((b - a) >= e):

        # Find middle point
        c = (a + b) / 2

        # Check if middle point is root
        if (func(c, secret_key, ascii) == 0.0):
            break
        if (func(c, secret_key, ascii) * func(a, secret_key, ascii) < 0):
            b = c
        else:
            a = c
        iter+=1
    return c, iter

def newton_raphson(x, secret_key, ascii):
    h = func(x, secret_key, ascii) / derivfunc(x)
    iter = 1
    while abs(h) >= 0.0001:
        h = func(x, secret_key, ascii) / derivfunc(x)
        x = x - h
        iter+=1

    return x, iter

def hybrid(x1,x2,E,secret_key,ascii):
    i=1
    e=100
    while(i<100):
        temp1=0
        temp=(x1+x2)/2
        h = func(temp, secret_key, ascii) / derivfunc(temp)
        h1=temp-h
        if x1<h1 and h1<x2:
            temp1=h1
        else:
            temp1=temp
        if abs(func(temp1,secret_key,ascii))<E:
            return temp1,i
        if func(x1,secret_key,ascii)*func(temp1,secret_key,ascii)<0:
            x2=temp1
        else:
            x1=temp1
        i=i+1
        
def secant(x1, x2, E, secret_key, ascii):
    n = 0
    xm = 0
    x0 = 0
    c = 0
    if (func(x1,secret_key,ascii) * func(x2,secret_key,ascii) < 0):
        while True:

            
            x0 = ((x1 * func(x2,secret_key,ascii) - x2 * func(x1,secret_key,ascii)) /
                  (func(x2,secret_key,ascii) - func(x1,secret_key,ascii)))


            c = func(x1,secret_key,ascii) * func(x0,secret_key,ascii)

            
            x1 = x2
            x2 = x0

            
            n += 1

            if (c == 0):
                break
            xm = ((x1 * func(x2,secret_key,ascii) - x2 * func(x1,secret_key,ascii)) /
                  (func(x2,secret_key,ascii) - func(x1,secret_key,ascii)))

            if (abs(xm - x0) < E):
                break
    return x0,n



file = open("Characters.txt", "r")

characters = []
Ascii = []

for line in file:
    for character in line:
        characters.append(character)
        Ascii.append(ord(character))

# print(Ascii)

cipher = []



# 
# for i in Ascii:
#     x0 = -1
#     cipher.append(newton_raphson(x0, secret_key, int(i)))
# j=1
for i in Ascii:
   x1 = 0
   x2 = 5
   cipher.append(hybrid(x1,x2,0.00001,secret_key,int(i)))
   #cipher.append(secant(x1,x2,0.001,secret_key, int(i)))

ciptxt=''
print("for the execution of the improvised hybrid algorithm")
for i in cipher:
    ciptxt+=str(i[0])+'@'
fi=open('cipher.txt','w')
fi.write(ciptxt)
end = time.time()
total_time = end-start
print(" the time used was:")
print(total_time)
print("memory used in gb")
print(psutil.Process(os.getpid()).memory_info().rss / 1024 **3 )