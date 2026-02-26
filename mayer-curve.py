import math as m
import numpy as np
import matplotlib.pyplot as plt

n = 50
t = np.linspace(0,1,n)
p = 0.0 # 0-1 parameter for gap in middle of Mayer curve


# Helper functions
def dist(a,b):
    return m.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)


# curve segments
C1 = lambda t: [m.cos(m.pi*t),m.sin(m.pi*t)+1]
C2 = lambda t: [-1,1-t]
C3 = lambda t: [(1/2-p/50)*(m.cos(m.pi*(t+1))-1)-p/25,1/2*m.sin(m.pi*(t+1))]
C4 = lambda t: [-p/25,t/2]
C5 = lambda t: [(1/4-p/25)*(m.cos(m.pi/2*t)-1)-p/25,1/4*m.sin(m.pi/2*t)+1/2]
C6 = lambda t: [1/4*m.sin(m.pi*(t-1))-1/4,1/4*m.cos(m.pi*(t-1))+1]
C7 = lambda t: [1/2*(t-1/2),1.25]
C8 = lambda t: [1/4*m.sin(m.pi*t)+1/4,1/4*m.cos(m.pi*t)+1]
C9 = lambda t: [(1/4-p/25)*m.cos(m.pi/2*(t+1))+0.25,1/4*m.sin(m.pi/2*(t+1))+1/2]
C10 = lambda t: [p/25,1/2*(1-t)]
C11 = lambda t: [(1/2-p/50)*m.cos(m.pi*(t+1))+1/2+p/50,1/2*m.sin(m.pi*(t+1))]
C12 = lambda t: [1,t]


# calculating distances
rawl = np.zeros(12)
rawl[0] = m.pi
rawl[1] = 1
rawl[2] = m.pi/2
rawl[3] = 1/2
rawl[4] = m.pi/8
rawl[5] = m.pi/4
rawl[6] = 1/2
rawl[7] = m.pi/4
rawl[8] = m.pi/8
rawl[9] = 1/2
rawl[10] = m.pi/2
rawl[11] = 1

cuml = np.zeros(12)
cumsum = 0
for i in range(12):
    cumsum += rawl[i]
    cuml[i] += cumsum

totall = cuml[11]

print("rawl: {}".format(rawl))
print("totall: {}".format(totall))
l = rawl/totall
cuml /= totall

print("l: {}".format(l))
print("cuml: {}".format(cuml))

# Stitching it together
C = np.zeros((n,2))
for i in range(n):
    if t[i] <= cuml[0]:
        C[i,:] = C1((t[i])/l[0])
    elif t[i] <= cuml[1]:
        C[i,:] = C2((t[i]-cuml[0])/l[1])
    elif t[i] <= cuml[2]:
        C[i,:] = C3((t[i]-cuml[1])/l[2])
    elif t[i] <= cuml[3]:
        C[i,:] = C4((t[i]-cuml[2])/l[3])
    elif t[i] <= cuml[4]:
        C[i,:] = C5((t[i]-cuml[3])/l[4])
    elif t[i] <= cuml[5]:
        C[i,:] = C6((t[i]-cuml[4])/l[5])
    elif t[i] <= cuml[6]:
        C[i,:] = C7((t[i]-cuml[5])/l[6])
    elif t[i] <= cuml[7]:
        C[i,:] = C8((t[i]-cuml[6])/l[7])
    elif t[i] <= cuml[8]:
        C[i,:] = C9((t[i]-cuml[7])/l[8])
    elif t[i] <= cuml[9]:
        C[i,:] = C10((t[i]-cuml[8])/l[9])
    elif t[i] <= cuml[10]:
        C[i,:] = C11((t[i]-cuml[9])/l[10])
    else:
        C[i,:] = C12((t[i]-cuml[10])/l[11])

# plotting
fig,ax = plt.subplots()

ax.plot(C[:,0],C[:,1])

plt.show()
