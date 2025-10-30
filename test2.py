import sys
input=sys.stdin.readline

n=int(input().strip())
a=list(map(int,input().split()))

# 压缩
vals=sorted(set(a))
id={v:i+1 for i,v in enumerate(vals)}  # 1-based
rk=[id[v] for v in a]
U=len(vals)

class BIT:
    def __init__(self,n):
        self.n=n
        self.t=[0]*(n+1)
    def add(self,i,v):
        while i<=self.n:
            self.t[i]+=v
            i+=i&-i
    def sum(self,i):
        s=0
        while i>0:
            s+=self.t[i]
            i-=i&-i
        return s
    def range(self,l,r):
        if r<l: return 0
        return self.sum(r)-self.sum(l-1)

L=[0]*(U+1)
R=[0]*(U+1)
for x in rk: R[x]+=1

bit=BIT(U)
def curC(x): return L[x]*R[x]

ans=0
for x in rk:
    ans += bit.range(x+1, U)
    bit.add(x, -curC(x))
    L[x]+=1; R[x]-=1
    bit.add(x,  curC(x))

print(ans)
