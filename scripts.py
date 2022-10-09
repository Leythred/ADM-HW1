#Say "Hello, World!" With Python
A = "Hello, World!"
print(A)



#Python If-Else
x=int(input())
if x<=0:
    print("retry")
elif x>100:
    print("Retry")
elif not (x/2).is_integer():
    print("Weird")
elif x<2:
    print("Weird")
elif x<=5:
    print("Not Weird")
elif x<=20:
    print("Weird")
elif x<=100:
    print("Not Weird")

    
    
#Arithmetic Operators
x = int(input())
y = int(input())
print(x+y)
print(x-y)
print(x*y)



#Python: Division
a = int(input())
b = int(input())
print(a//b)
print(a/b)



#Loops
n = int(input())
x = range(n)
for i in x:
    print(i*i)

    
    
#Write a function
def is_leap(year):
    leap = False
    if (year/400).is_integer():
        leap = True
    elif (year/100).is_integer():
        leap = False
    elif (year/4).is_integer():
        leap = True
    
    return leap



#Print Function
n = int(input())
s = ""
x = range(1, n+1)
for i in x:
    s += str(i)
print(s)



#Find the Runner-Up Score!
n = int(input())
arr = list(map(int, input().split()))
x=[]
for i in range(n):
    x.append(arr[i]-max(arr))
y=[a for a in x if a != 0]
z=max(y)
print(z+max(arr)) 



#Nested Lists
name=[]
score=[]
n = int(input())
for i in range(n):
    name.append(input())
    score.append(float(input()))
x=dict(zip(name, score))
sx=sorted(x.items(), key=lambda x: x[1])
list = []
a=0
for i in range(n):
    if sx[i][1] == sx[0][1]:
        a = a+1
for i in range(n):
    if sx[i][1] == sx[a][1]:
        list.append(sx[i][0])       
slist=sorted(list)
for i in range(len(slist)):
    print(slist[i])

    
    
#Finding the percentage
n = int(input())
sm = {}
for _ in range(n):
    name, *line = input().split()
    scores = list(map(float, line))
    sm[name] = scores
qn = input()
a=(sum(sm[qn]))/len(scores)
print("{:.2f}".format(a))



#sWAP cASE
def swap_case(s):
    a=[]
    for i in range(len(s)):
        if s[i].isupper()==True:
            a.append(s[i].lower())
        elif s[i].islower()==True:
            a.append(s[i].upper())
        else:
            a.append(s[i])
    b="".join(a)
    return(b)



#String Split and Join
def split_and_join(line):
    a = line.split(" ")
    b = "-".join(a)
    return(b)



#What's Your Name?
def print_full_name(first, last):
    # Write your code here
    a=[first, last]
    b=" ".join(a)
    c=["Hello", b]
    d=" ".join(c)
    e=[d, "! You just delved into python."]
    f="".join(e)
    print(f)

    
    
#Mutations
def mutate_string(string, position, character):
    a=string[:position] + character + string[position+1:]
    return(a)



#Find a string
def count_substring(string, sub_string):
    x=0
    j=0
    res=[]
    for i in range(len(string)):
        x=string[j:].find(sub_string)
        j=j+1+x
        if x != -1:
            res.append(x)
    return(len(res))



#String Validators
s = input()
x=[0,0,0,0,0]
for i in range(len(s)):
    if s[i].isalnum()==True:
        x[0]+=1
    if s[i].isalpha()==True:
        x[1]+=1
    if s[i].isdigit()==True:
        x[2]+=1
    if s[i].islower()==True:
        x[3]+=1
    if s[i].isupper()==True:
        x[4]+=1
for i in range(len(x)):
    if x[i]==0:
        print("False")
    else:
        print("True")

        
        
#Text Wrap
def wrap(string, max_width):
    a=textwrap.wrap(string, width=max_width)
    for i in range(len(a)-1):
        print(a[i])
    s=a[len(a)-1]
    return(s)



#Introduction to Sets
def average(array):
    x = set(array)
    m = len(x)
    a = sum(x)/m
    return("%0.3f" % a)



#Symmetric Difference
x=int(input())
n=set(map(int, input().split()))
y=int(input())
m=set(map(int, input().split()))
a=m.difference(n)
b=n.difference(m)
c=a.union(b)
s=list(c)
s.sort()
for i in range(len(s)):
    print(s[i])
    
    
    
#No Idea!
nm=list(map(int, input().split()))
arr=list(map(int, input().split()))
A=set(map(int, input().split()))
B=set(map(int, input().split()))
more=[x for x in arr if x in A]
less=[x for x in arr if x in B]
h=len(more)-len(less)
print(h)



#Set .add()
n=int(input())
s=set([])
for i in range(n):
    c=input()
    s.add(c)
print(len(s))



#Set .discard(), .remove() & .pop()
N = int(input())
stg = set(map(int, input().split()))
n = int(input())
for i in range(n):
    S = list(input().split())
    if S[0]=="pop":
        stg.pop()
    if S[0]=="remove":
        l = int(S[1])
        if l in stg:
            stg.remove(l)
    if S[0]=="discard":
        l = int(S[1])
        stg.discard(l)
print(sum(stg))



#Set .union() Operation
n = int(input())
stg1 = set(map(int, input().split()))
m = int(input())
stg2 = set(map(int, input().split()))
A = stg1.union(stg2)
print(len(A))



#Set .intersection() Operation
n = int(input())
stg1 = set(map(int, input().split()))
m = int(input())
stg2 = set(map(int, input().split()))
A = stg1.intersection(stg2)
print(len(A))



#Set .difference() Operation
n = int(input())
stg1 = set(map(int, input().split()))
m = int(input())
stg2 = set(map(int, input().split()))
A = stg1.difference(stg2)
print(len(A))



#Set .symmetric_difference() Operation
n = int(input())
stg1 = set(map(int, input().split()))
m = int(input())
stg2 = set(map(int, input().split()))
A = stg1.symmetric_difference(stg2)
print(len(A))



#Set Mutations
N = int(input())
stg = set(map(int, input().split()))
n = int(input())
for i in range(n):
    S = list(input().split())
    stgn = set(map(int, input().split()))
    if S[0]=="intersection_update":
        stg.intersection_update(stgn)
    if S[0]=="update":
        l = int(S[1])
        stg.update(stgn)
    if S[0]=="symmetric_difference_update":
        l = int(S[1])
        stg.symmetric_difference_update(stgn)
    if S[0]=="difference_update":
        l = int(S[1])
        stg.difference_update(stgn)
print(sum(stg))



#The Captain's Room
    #this one completed only few of the test cases because of the time limit, it gave me 2.93 points
n = int(input())
stg = list(map(int, input().split()))
seti = list(set(stg))
x = []
y = []
for i in range(len(seti)):
    x.append(stg.count(seti[i]))
    y.append(seti[i])
for j in range(len(x)):
    if x[j]==1:
        print(y[j])

        

#Check Subset
N = int(input())
for i in range(N):
    n1 = int(input())
    stg1 = set(map(int, input().split()))
    n2 = int(input())
    stg2 = set(map(int, input().split()))
    if len(stg2.intersection(stg1))==len(stg1):
        print("True")
    else:
        print("False")
    

    
#collections.Counter()
from collections import Counter
n = int(input())
stg = list(map(int, input().split()))
C = Counter(stg)
N = int(input())
s = 0
for i in range(N):
    a = list(map(int, input().split()))
    if C[a[0]] != 0:
        C[a[0]] -= 1
        s += a[1]
print(s)



#DefaultDict Tutorial
from collections import defaultdict
nm = list(map(int, input().split()))
n = nm[0]
m = nm[1]
D = defaultdict(list)
x=[]
for i in range(n):
    a = (input())
    D[a].append(i+1)
for j in range(m):
    b = (input())
    x.append(D[b])
for h in range(m):
    if len(x[h]) != 0:
        print(" ".join(map(str,x[h])))
    else:
        print(-1)
        


#Collections.namedtuple()
from collections import namedtuple
N = int(input())
col = list(input().split())
k = col.index("MARKS")
x=[]
for i in range(N):
    std = namedtuple("std", col)
    inpt = list(input().split())
    student = std(ID=0,NAME=0,CLASS=0,MARKS=inpt[k])
    x.append(int(student.MARKS))
print(sum(x)/N)



#Word Order
import collections
N = int(input())
D = {}
for i in range(N):
    string = input()
    if string not in D:
        D[string] = 0
    D[string] += 1
print(len(D.keys()))
print(" ".join(map(str,D.values())))



#Collections.deque()
from collections import deque
N = int(input())
stg = deque()
for i in range(N):
    S = list(input().split())
    if S[0]=="pop":
        stg.pop()
    if S[0]=="popleft":
        stg.popleft()
    if S[0]=="append":
        l = int(S[1])
        stg.append(l)
    if S[0]=="appendleft":
        l = int(S[1])
        stg.appendleft(l)
print(" ".join(map(str,stg)))



#Arrays
def arrays(arr):
    a = numpy.empty(len(arr))
    for i in range(len(arr)):
        a[i] = float(arr[len(arr)-1-i])
    return(a)



#Shape and Reshape
import numpy as np
string = list(map(int,input().split()))
a = np.array(string)
print(np.reshape(a,(3,3)))



#Transpose and Flatten
import numpy as np
nm = list(map(int,input().split()))
a = np.empty((nm[0],nm[1]), int)
for i in range(nm[0]):
    a[i] = list(map(int, input().split()))
print(np.transpose(a))
print(a.flatten())



#Concatenate
import numpy as np
nmp = list(map(int,input().split()))
a = np.empty((nmp[0],nmp[2]), int)
b = np.empty((nmp[1],nmp[2]), int)
for i in range(nmp[0]):
    a[i] = list(map(int, input().split()))
for j in range(nmp[1]):
    b[j] = list(map(int, input().split()))
print(np.concatenate((a,b), axis = 0))



#Zeros and Ones
import numpy as np
xyz = list(map(int,input().split()))
print(np.zeros((xyz), int))
print(np.ones((xyz), int))



#Eye and Identity
import numpy as np
np.set_printoptions(legacy="1.13")
xy = list(map(int,input().split()))
print(np.eye(xy[0],xy[1], k=0))



#Array Mathematics
import numpy as np
nm = list(map(int,input().split()))
a = np.empty((nm[0],nm[1]), int)
b = np.empty((nm[0],nm[1]), int)
for i in range(nm[0]):
    a[i] = list(map(int,input().split()))
for j in range(nm[0]):
    b[j] = list(map(int,input().split()))
print(np.add(a,b))
print(np.subtract(a,b))
print(np.multiply(a,b))
print(a//b)
print(np.mod(a,b))
print(np.power(a,b))



#Floor, Ceil and Rint
import numpy as np
np.set_printoptions(legacy="1.13")
arr = list(map(float,input().split()))
a = np.array(arr)
print(np.floor(a))
print(np.ceil(a))
print(np.rint(a))



#Sum and Prod
import numpy as np
nm = list(map(int,input().split()))
a = np.empty((nm[0],nm[1]), int)
for i in range(nm[0]):
    a[i] = list(map(int,input().split()))
print(np.prod(np.sum(a, axis=0)))



#Min and Max
import numpy as np
nm = list(map(int,input().split()))
a = np.empty((nm[0],nm[1]), int)
for i in range(nm[0]):
    a[i] = list(map(int,input().split()))
print(np.max(np.min(a, axis=1)))



#Mean, Var, and Std
import numpy as np
nm = list(map(int,input().split()))
a = np.empty((nm[0],nm[1]), int)
for i in range(nm[0]):
    a[i] = list(map(int,input().split()))
print(np.mean(a, axis=1))
print(np.var(a, axis=0))
print(round(np.std(a),11))



#Dot and Cross
import numpy as np
N = int(input())
a = np.empty((N,N), int)
b = np.empty((N,N), int)
for i in range(N):
    a[i] = list(map(int,input().split()))
for i in range(N):
    b[i] = list(map(int,input().split()))
c = np.transpose(b)
z = np.empty((N,N), int)
for i in range(N):
    for j in range(N):
        z[i][j]=(np.dot(a[i],c[j]))
print(z)



#Inner and Outer
import numpy as np
a = list(map(int,input().split()))
b = list(map(int,input().split()))
A = np.array(a)
B = np.array(b)
print(np.inner(A, B))
print(np.outer(A, B))



#Polynomials
import numpy as np
a = list(map(float,input().split()))
x = float(input())
A = np.array(a)
print(np.polyval(A,x))



#Linear Algebra
import numpy as np
N = int(input())
a = np.empty((N,N), float)
for i in range(N):
    a[i] = list(map(float,input().split()))
print(round(np.linalg.det(a),2))



#Calendar Module
import calendar as cd
date = list(map(int,input().split()))
day = cd.weekday(date[2],date[0],date[1])
if day == 0:
    print("MONDAY")
if day == 1:
    print("TUESDAY")
if day == 2:
    print("WEDNESDAY")
if day == 3:
    print("THURSDAY")
if day == 4:
    print("FRIDAY")
if day == 5:
    print("SATURDAY")
if day == 6:
    print("SUNDAY")
    
    
    
#Time Delta
import math
import os
import random
import re
import sys
from datetime import datetime

def time_delta(t1, t2):
    d1 = datetime.strptime(t1, "%a %d %b %Y %H:%M:%S %z")
    d2 = datetime.strptime(t2, "%a %d %b %Y %H:%M:%S %z")
    return(abs(int((d1-d2).total_seconds())))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = str(time_delta(t1, t2))

        fptr.write(delta + '\n')

    fptr.close()

    
    
#Map and Lambda Function
cube = lambda x: x*x*x

def fibonacci(n):
    fib = [0,1]
    if n>=2:
        for i in range(n-2):
            fib.append(fib[i+1]+fib[i])
    elif n==0: fib=[]
    else: fib=[0]
    return(fib)



#Re.split()
regex_pattern = r"[,.]"



#Capitalize!
def solve(s):
    a = s.split(" ")
    b = []
    for i in range(len(a)):
        b.append(a[i].capitalize())
    return(" ".join(b))



#Birthday Cake Candles
import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    count = 0
    c = max(candles)
    for i in range(len(candles)):
        if candles[i]==c:
            count += 1
    return(count)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

   
    
#Number Line Jumps
import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    if v2-v1 == 0:
        return("NO")
    
    t = (x1-x2)/(v2-v1)
    
    if t/math.ceil(abs(t))==1:
        return("YES")
    else:
        return("NO")

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

    
    
#Viral Advertising
import math
import os
import random
import re
import sys

def viralAdvertising(n):
    a = 2
    s = 2
    for i in range(1,n):
        b = math.floor(a*3/2)
        s += b
        a = b
    return(s)
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()



#Insertion Sort - Part 1
import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    t=arr[n-1]
    for i in range(1,n):
        if arr[n-1-i]>t:
            arr[n-i]=arr[n-1-i]
        else:
            arr[n-i]=t
        print(*arr, sep=" ")
        if arr[n-1-i]<t:
            break
    if arr[0]>t:
        arr[0]=t
        print(*arr, sep=" ")   
    
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)
    


#Insertion Sort - Part 2
import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    for i in range(1,n):
        t = arr[i]
            
        j=i-1
            
        while t<arr[j]:
            arr[j+1]=arr[j]
            arr[j]=t
            j-=1
            if j <= -1:
                break

        print(*arr, sep=" ")

    if arr[0]>t:
        arr[1]=arr[0]
        arr[0]=t
        print(*arr, sep=" ")

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
    

    
#Recursive Digit Sum
    #also this one exceeded the time limit for some test cases, it gave me 16.67 points
import math
import os
import random
import re
import sys

def superDigit(n, k):     #this works but takes too much time
    a = str(n)*k
    while len(a) >= 2:
        a_num = 0
        l=[int(i) for i in a]
        a_num = sum(l)
        a = str(a_num)
    return(int(a))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


