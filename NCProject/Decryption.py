import math
fi=open('NNcipher.txt','r')

def func(x, secret_key):
    return secret_key * x * x * x + 4*x*x+secret_key*math.cos(x) - 2.7 * x

msg=[]

de=['']
c=0

for i in fi:
    for j in str(i):
        if(j=='@'):
            c+=1
            de.append('')
        else:
            de[c]+=str(j)

de.pop()
for i in de:
    # print(i)
    msg.append(chr(round(func(float(i), 3))))

pltxt=''
for i in msg:
    pltxt+=i

fid=open('decrypted.txt','w')
fid.write(pltxt)
