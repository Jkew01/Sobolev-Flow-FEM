# Manifest
#
# The purpose of this file is to perform a FEM for the classic curvature flow
# straight classic style
#
# Easy right?
#
# Written by Jonas Kew
#


# The imports
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt # probably don't use plots directly here
import matplotlib.animation as animation
import math as m
import numpy.linalg as la
import scipy as sp
from tempfile import TemporaryFile
import time
outfile = TemporaryFile()


start = time.time()
# GLOBAL VARS
n = 300 # our square works for 45
h = 1/n
t = np.linspace(0.0,1.0,n)
dt = 10**(-2) # 10**-1
b = 1/9600 # speed parameter
#a = 1
#lda = 0.25

debug = False
genGif = True
plotLength = False
chase = False


# UTILS

# NOTE - Consider using other numeric schemes for integration
# finite element integration
def feint_old(f,h,intrv):
    b = intrv[0]
    a = b+h
    cumsum = 0
    cumerr = 0
    i = 0
    while True:
        if a > intrv[1]+10**-14: # warning! really high error tolerance
            #print("[!] Error - interval overshoot - {}".format(a))
            return cumsum,cumerr # do proper error handling

        val1,val2 = integrate.quad(f,b,a)
        #if i%100==0:
        #    print("{},{},{}".format(a,b,h))
        #val1 = (f(i)*h)
        #val2 = 0 # sketchy
        cumsum = cumsum+val1
        #print("{},{},{},{},{},{},{}".format(b,a,h,f(i),t[i],u0x[i],len(t)))
        #input("  press space...  ")
        cumerr += val2
        i += 1
        #print("({},{}) - {} , {}".format(b,a,cumsum,cumerr))
        b = a
        a = b+h
        if abs(a-h - intrv[1]) <= 10**-14:
            return cumsum,cumerr
        if i >=len(t):
            return cumsum,cumerr

# new method does trapezoidal rule with linspace functions
def feint(f,h,intrv):
    cumsum = 0
    for i in range(len(t)-1):
        cumsum += (f(i)+f(i+1))*h/2
    return cumsum
        
# finite difference differentiation
def fd(f,x,fh=0.00001):
    return (f(x+fh) - f(x))/fh

# This integration is perfect for products of linear functions (like nodal basis and nodal approx)
# i - center of nodal basis
def simpsonint(i,data):
    if i == 0:
        return (2*data[0]+data[1])*h/6.15 # should actually be 6??
    elif i == n-1:
        return (2*data[n-1]+data[n-2])*h/6.15
    return (data[i-1]+4*data[i]+data[i+1])*h/6.15

# This version works for arbitrary products of linear functions
def simpsonintegral(a,b,c,d):
    return h*(a*c + (a+b)*(c+d) + b*d)/6

# for arclength reparam
def dist(a,b):
    return m.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
# takes C, a n*2 matrix
def length(C):
    cumsum = 0
    for i in range(n-1):
        cumsum += dist(C[i,:],C[i+1,:])
    return cumsum
def I(a,b,t):
    return (b-a)*t+a
def linC(x,C):
    lowernode = m.floor(x*n)
    uppernode = m.ceil(x*n)
    remainder = x%(1/n)
    return I(C[lowernode,:],C[uppernode,:],remainder*n)
def alinC(x,C):
    totallength = length(C)
    scaledx = x*totallength
    if x == 1:
        return C[n-1,:]
    cacheddistance = 0
    #print("Scaledx: {},len:{}".format(scaledx,totallength))
    for i in range(n-1):
        #print("({}) - cdist:{},cdist[+1]:{}".format(i,cacheddistance,cacheddistance+dist(C[i,:],C[i+1,:])))
        if scaledx >= cacheddistance and scaledx <= cacheddistance + dist(C[i,:],C[i+1,:]):
            #print("Found to be in this domain")
            lowernode = C[i,:]
            uppernode = C[i+1,:]
            remainderdist = scaledx - cacheddistance
            
            interpol = remainderdist/dist(lowernode,uppernode)
            #print("Interpol:{},lnode:{},unode:{}".format(interpol,lowernode,uppernode))
            if interpol <= 1 and interpol >= 0:
                #print("\n--------------")
                return I(lowernode,uppernode,interpol)
            #print("\nErr analysis:\n")
        cacheddistance += dist(C[i,:],C[i+1,:])
        #print("\n")
def reparam(C):
    totallength = length(C)
    lengthperelement = totallength/(n-1)
    sigma = np.zeros((n,2))
    for i in range(n):
        sigma[i,:] = alinC(1/(n-1)*i,C)
    return sigma

    print("[!] - ERROR -- Check alinC(x)\n-----------------")
    1/0


# sign function: returns -1 or 1. Note: we assume 0 is positive
def sgn(x):
    if x<0:
        return -1
    return 1

class Viewport:
    x = [-1.1,1.1]
    y = [-1.1,1.1]
viewport = Viewport()

# finite difference |C'(t)|**2, needed?
#def cdsquared(f,g,x,y):
    # return m.sqrt((x1-x2)**2 + (y1,y2)**2)/h
#    return m.sqrt(fd(f,x)**2+fd(f,y)**2)

# nodal basis approximation
def nodalapprox(x,t,data):
    h = (t[1]-t[0])
    lower = m.floor((x-t[0])/h)
    upper = m.ceil((x-t[0])/h)
    interpol = x%h
    # NOTE - THIS IS DANGEROUS, INVESTIGATE
    if lower < 0 or upper > n:
        return 0
    #print("lower: {}\nupper: {}\n---------".format(lower,upper))
    return (data[upper]-data[lower])*interpol+data[lower]


print(nodalapprox(.36,[0.0,0.1,0.2,0.3,0.4],[0,1,20,40,-40]))
print("\n\n---------------------\nEND OF UNIT TEST\n---------------------\n\n")

# Green's Kernel for the Sobolev Gradient Flow. lda refers to lambda and l are parameters.
def greenKernel(x,lda,l):
    try:
        ans = -(  m.cosh( (abs(x)-l/2)/(lda*l) )  )/( 2*lda*l*m.sinh(1/(2*lda)) )
    except:
        print("Error: Division by zero in the Greens kernel.\nVars: x = {}, lda = {}, l: {}".format(x,lda,l))
        exit()
        ans = 0
    return ans

# Our good friend the circular convolution. i refers to our nodal basis shift
def convolution(f,offset,lda,l):
    output = np.zeros(n)
    for j in range(n):
        #output[i] = feint_old(lambda x: nodalapprox(x-t[i],t,f)*greenKernel((x-t[i])*l,lda,l)*l,h,[0.0,1.0])[0]
        output[j] = feint_old(lambda x: f(x)*greenKernel((x-t[j]+offset)*l,lda,l)*l,h,[0.0,1.0])[0]
    return output

# Discrete circular convolution as suggested in the provided paper (Homogeneous Sobolev Gradient Flow of the length functional)
# u = X, the curve
# v = Greens Kernel
def convolution2(u,v):
    output = np.zeros((n,2))
    for k in range(n):
        for i in range(n-1): # n-1 seems to fix an issue with the boundary node being convolved twice
            output[k,0] += u[i,0]*v[k-i]
            output[k,1] += u[i,1]*v[k-i]
            print("using convolution2 not self.convolution2")
            print(0/0)
    return output

def curvatures(C):
    curvatures = np.zeros(n)
    a = ((C[1,:]-C[0,:]))/h
    b = ((C[0,:]-C[n-2,:]))/h
    curvatures[0] = dist(a,b)/h
    curvatures[n-1] = curvatures[0]
    for i in range(n-2):
        a = ((C[i,:]-C[i+1,:]))/h
        b = ((C[i+1,:]-C[i+2,:]))/h
        curvatures[i+1] = dist(a,b)/h
    return curvatures

# checks for convexity assuming embedded curve (not intersecting) and polygon
# uses z component of the cross product
def isConvex(C):
    crossprods = np.zeros(n)
    dx1 = C[0,0] - C[n-2,0]
    dy1 = C[0,1] - C[n-2,1]
    dx2 = C[1,0] - C[0,0]
    dy2 = C[1,1] - C[0,1]
    crossprods[0] = dx1*dy2-dy1*dx2
    for i in range(n-2):
        dx1 = C[i+1,0] - C[i,0]
        dy1 = C[i+1,1] - C[i,1]
        dx2 = C[i+2,0] - C[i+1,0]
        dy2 = C[i+2,1] - C[i+1,1]
        crossprods[i] = dx1*dy2-dy1*dx2
    # remove zero crossprods. These can be removed while describing the same polygon
    # as they occur for colinear node triples
    # If either min or max is zero then the whole curve is either stricly positive or negative crossprod
    # meaning it is convex
    if min(crossprods)*max(crossprods) >= 0:
        return True
    return False

###########################
# FEM
###########################
class fem:
    # VARIABLES
    h = 0.005 # needs to agree with fem.h, class needed

    def __init__(self,h,interval,dirichlet,f,u0x,u0y,debug=False):
        self.h = h
        self.u0x = u0x
        self.u0y = u0y
        self.setQ()
        self.interval = interval
        self.dirichlet = dirichlet
        self.f = f
        self.debug = debug
        self.A = np.zeros((2*n,2*n)) # 2 times for each set of nodal basis functions corresponding to x and y
        #self.generate_bilinear_form(regen=True) # set to true when changing bilinear form
        self.updateDiscLengthVector()
        # self.generateGreensMatrix()
        self.generate_bilinear_form()
        #self.invA = la.inv(self.A)
        #self.area = self.getSignedArea()

    # UTILS

    # returns the nodes which most closely sandwich the given value
    def getAdjacentNodeIndices(self,x):
        return m.floor((x-self.interval[0])/self.h),m.ceil((x-self.interval[0])/self.h)

    def getLength(self):
        # nodes 0 and n-1 should be the same value due to coupling
        length = 0#m.sqrt((self.u0x[0]-self.u0x[n-1])**2 + (self.u0y[0]-self.u0y[n-1])**2)*0
        for i in range(n-1):
            length += m.sqrt((self.u0x[i]-self.u0x[i+1])**2 + (self.u0y[i]-self.u0y[i+1])**2)
        return length # scale for more interesting behaviour visually

    def getSignedArea(self):
        fh=0.00001
        return feint_old(lambda x: (nodalapprox(x+fh,t,self.u0y) - nodalapprox(x,t,self.u0y))/fh * nodalapprox(x,t,self.u0x) -
                         (nodalapprox(x+fh,t,self.u0x) - nodalapprox(x,t,self.u0x))/fh * nodalapprox(x,t,self.u0y),
                         self.h,[0,1])[0]/2

    def getDiscCumLen(self,x):
        output = 0
        for i in range(x-1):
            output += m.sqrt((self.u0x[i]-self.u0x[i+1])**2 + (self.u0y[i]-self.u0y[i+1])**2)
        return output
    # Returns arclength distance between two nodes
    def getDiscDist(self,a,b):
        # sort a,b
        if a > b:
            c = b
            b = a
            a = c
        output = 0
        count = 0
        for i in range(b-a):
            if i==0 and b == n-1 and debug:
                print(i+a)
            count += 1
            output += m.sqrt((self.u0x[i+a]-self.u0x[i+a+1])**2 + (self.u0y[i+a]-self.u0y[i+a+1])**2)
        if debug:
            print(count)
        return output

    def convolution(self,lda,l):
        output = np.zeros((n,2))
        f = lambda x: self.phi(x) #lambda x: nodalapprox(x,t,self.u0x)
        g = lambda x: self.phi(x) #lambda x: nodalapprox(x,t,self.u0y)
        for j in range(n):
            #output[i] = feint_old(lambda x: nodalapprox(x-t[i],t,f)*greenKernel((x-t[i])*l,lda,l)*l,h,[0.0,1.0])[0]
            # output[j,0] = feint_old(lambda x: f(x)*greenKernel((x-t[j])*l,lda,l)*l,h,[0.0,1.0])[0]
            # output[j,1] = feint_old(lambda x: g(x)*greenKernel((x-t[j])*l,lda,l)*l,h,[0.0,1.0])[0]

            output[j,0] = feint_old(lambda x: f(x)*greenKernel((self.getDiscCumLen(j)-t[j])*l,lda,l)*l,h,[0.0,1.0])[0]
            output[j,1] = feint_old(lambda x: g(x)*greenKernel((self.getDiscCumLen(j)-t[j])*l,lda,l)*l,h,[0.0,1.0])[0]
        return output

    def convolution2(self,u,l):
        uhat = np.zeros((n,2))
        for i in range(n):
            uhat[i] = u[i] * self.avgArc(i)
        output = np.zeros((n,2))
        #for k in range(n):
        #    for i in range(n-1): # n-1 seems to fix an issue with the boundary node being convolved twice
        #        output[k,0] += u[i,0]*self.greensMatrix[k,i]
        #        output[k,1] += u[i,1]*self.greensMatrix[k,i]
        self.greensMatrix[:,n-1] = np.zeros(n)
        self.greensMatrix[n-1,:] = np.zeros(n)
        output = self.greensMatrix@(uhat)
        output[n-1] = output[0]
        return output

    def setQ(self):
        q = np.zeros(n)
        for i in range(n):
            fd1 = np.sqrt((self.u0x[(i+1)%n]-self.u0x[i])**2+(self.u0y[(i+1)%n]-self.u0y[i])**2)
            fd2 = np.sqrt((self.u0x[(i-1)%n]-self.u0x[i])**2+(self.u0y[(i-1)%n]-self.u0y[i])**2)
            if i == 0:
                fd1 = np.sqrt((self.u0x[1]-self.u0x[0])**2+(self.u0y[1]-self.u0y[0])**2)/1
                fd2 = np.sqrt((self.u0x[n-2]-self.u0x[0])**2+(self.u0y[n-2]-self.u0y[0])**2)/1
            elif i==n-1:
                fd1 = np.sqrt((self.u0x[1]-self.u0x[n-1])**2+(self.u0y[1]-self.u0y[n-1])**2)/1
                fd2 = np.sqrt((self.u0x[n-2]-self.u0x[n-1])**2+(self.u0y[n-2]-self.u0y[n-1])**2)/1
                    
            q[i] = n*(fd1+fd2)/2
        self.q = q

    def generate_bilinear_forms_old(self, regen=False):
        # generate the bilinear form once for performance
        N = n
        nodes = t
        A = np.zeros((N,N))
        if regen:
            print("Generating bilinear form!")
            hasPrinted = False
            for i in range(N):
                if i > N/2 and not hasPrinted:
                    print("Halfway complete on the bilinear form")
                    hasPrinted = True
                for j in range(N):
                    # WARNING THIS ASSUMES ONLY ADJACENT NODES ARE RELATED, GENERALLY TRUE THOUGH
                    if abs(i-j) <= 1 or (i*j == 0 and i+j==N-1):
                        #A[i,j] = feint_old(lambda x: fd(self.phi,x-nodes[i])*fd(self.phi,x-nodes[j]),self.h,self.interval)[0]
                        A[i,j] = feint_old(lambda x: fd(self.phi,x-nodes[i])*fd(self.phi,x-nodes[j])*dt
                                           + (self.phi(x-nodes[i]))*self.phi(x-nodes[j]),
                                           self.h,self.interval)[0]

            # POISSON PROBLEM
            # hack it - curt of https://math.stackexchange.com/questions/4094103/finite-element-method-poisson-equation-in-1d-with-inhomogenous-boundary-conditio
            # i.e. (so I don't forget), OVERWRITE the end points of the nodal basis matrix to agree with reality (corresponding to the end nodes of the mesh).
            # This is also specific to the PDE, so check carefully
            #A[0,0] = 1
            #A[0,1] = 0
            #A[N-1,N-1] = 1 # even correct?
            #A[N-1,N-2] = 0

            # for periodic/circle domains, the same mods are needed for the last col
            #A[0,N-1] = 0
            #A[0,N-2] = 0
            #A[N-1,0] = 0
            #A[n-1,1] = 0

            # HEAT PROBLEM
            # Specifically, this enforces the rule that the end nodes are equal
            # Thanks to https://www.featool.com/tutorial/2017/10/30/Periodic-Boundary-Conditions-and-the-Solver-Hook-Functionality/
            # Add the first and last rows
            A[0,:] = A[0,:] + A[n-1,:]
            # Use last row (now put into first) to enforce the end terms matching
            A[n-1,:] = np.zeros(n)
            A[n-1,[0,n-1]] = [1 , -1]
            
            self.A = A
            self.save_bilinear_form()
        else:
            self.load_bilinear_form()


    def save_bilinear_form(self):
        with open('bilinear_form.npy', 'wb') as file:
            np.save(file,self.A)
    def load_bilinear_form(self):
        with open('bilinear_form.npy', 'rb') as file:
            self.A = np.load(file)

    # nodal basis at zero
    def phi(self,x):
        if x >= -h and x < 0:
            return 1+x/h
        elif x >= 0 and x <= h:
            return 1-x/h
        else:
            return 0
        
    def phi_(self,x):
        if x%1 >= 1-h and x%1 < 1:
            return 1+((x%1)-1)/(h)
        elif x%1 >= 0 and x%1 <= h:
            return 1-(x%1)/h
        else:
            return 0


    def updateDiscLengthVector(self):
        self.discLengths = np.zeros(n-1)
        for i in range(n-1):
            self.discLengths[i] = dist([self.u0x[i+1],self.u0y[i+1]],[self.u0x[i],self.u0y[i]])

    def avgArc(self,i):
        if i == 0 or i == n-1:
            a = np.array([self.u0x[1],self.u0y[1]])
            b = np.array([self.u0x[0],self.u0y[0]])
            c = np.array([self.u0x[n-2],self.u0y[n-2]])
        else:
            a = np.array([self.u0x[i-1],self.u0y[i-1]])
            b = np.array([self.u0x[i],self.u0y[i]])
            c = np.array([self.u0x[i+1],self.u0y[i+1]])
        return (dist(a,b)+dist(b,c))/2

    def getDiscDist(self,a,b):
        output = 0
        if a > b:
            c = a
            a = b
            b = c
        for i in range(b-a):
            output += self.discLengths[i+a]
        return output

    def generateGreensMatrix(self):
        self.greensMatrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                self.greensMatrix[i,j] = greenKernel(self.getDiscDist(i,j),lda,self.getLength()) #*self.getDiscDist(i,j)


    # generates the x and y bilinear forms according to current data
    def generate_bilinear_form(self):
        nodes = t
        self.A = np.zeros((2*n,2*n))
        l = self.getLength()
        for i in range(n):
            # This object is needed for the Simpsons rule
            phiArray = np.zeros(n)
            phiArray[i] = 1
            #if i == 0:
            #    print("Params:\n  lda = {}\n  l = {}\n  a = {}\n".format(lda,l,a))
            #rowConvolution = convolution(self.phi,nodes[i],lda,l)
            #greensArray = np.zeros(n)
            u0 = np.zeros((n,2))
            u0[:,0] = phiArray#self.u0x
            u0[:,1] = phiArray#self.u0y
            #for k in range(n):
                # We need an expression for the arclength distance
                #sigmax = self.getDiscCumLen(k+1)
                #greensArray[k] = greenKernel((1/n)*k*0+sigmax,lda,l)
            #greensArray*=1 # scale by l
            #rowConvolution = convolution2(u0,greensArray)
            #rowConvolution = self.convolution2(u0,l)
            #q = np.sqrt((self.u0x)**2+(self.u0y)**2)
            #q[0] = (q[1]+q[n-2])/2
            #q[n-1]=q[0]
            for j in range(n):
                notlocal = False
                # The convolution makes this no longer just local
                if abs(i-j) <= 1 or (i*j == 0 and i+j==n-1) or notlocal:
                    intrvv = [nodes[i]-h,float(max(nodes[i]+h,1.0))]
                    # xx basis functions
                    # foo1 = feint_old(lambda x: sgn(fd(self.phi,x-nodes[j]))*fd(self.phi,x-nodes[i]),self.h,self.interval)[0]
                    # foo2 = feint_old(lambda x: abs(fd(self.phi,x-nodes[i]))*(self.phi(x-nodes[i]))*self.phi(x-nodes[j])/(dt),self.h,self.interval)[0]

                    #self.A[i,j] = feint_old(lambda x: (self.phi(x-nodes[j]))/((lda**2) * (l**2))*(nodalapprox(x,t,rowConvolution)+self.phi(x-nodes[i]))*dt + (self.phi(x-nodes[j])*self.phi(x-nodes[i])),
                    #self.h,intrvv)[0]
                    #q = n * np.sqrt((self.u0x[(i+1)%n]-self.u0x[i])**2+(self.u0y[(i+1)%n]-self.u0y[i])**2) # speed in last time step
                    #if i == 0:
                    #    q = n * np.sqrt((self.u0x[n-2]-self.u0x[0])**2+(self.u0y[n-2]-self.u0y[0])**2)/1
                    #elif i==n-1:
                    #    q = n * np.sqrt((self.u0x[n-2]-self.u0x[n-1])**2+(self.u0y[n-2]-self.u0y[n-1])**2)/1
                    #print("q check: {}".format(q))
                    self.A[j,i] = simpsonint(j, self.q[j]*phiArray )
                    if i==j:
                        self.A[j,i] += (1+dt)*1/(self.q[j])/h
                    elif abs(i-j) == 1:
                        self.A[j,i] -= (1+dt)/(self.q[j])/h/2
                    elif i+j == n-1 and i*j==0: # this case accounts for when they each equal the end points, which are indentified.
                        self.A[j,i] -= (1+dt)/(self.q[j])/h/2 # cancels double adding of term on boundary.
                    #0/0 # Oh No!
                    #self.A[j,i] = simpsonint(j, )
                    #if (i%10==0 and j==i):
                    #    print("(i,j) = ({},{}) -- Parts: {}, {}".format(i,j,feint_old(lambda x: dt*(self.phi(x-nodes[j]))/((lda**2) * (l**2))*(nodalapprox(x,t,rowConvolution)+self.phi(x-nodes[i])),self.h,self.interval)[0],
                    #                                     feint_old(lambda x: (self.phi(x-nodes[j])*self.phi(x-nodes[i])),self.h,self.interval)[0]))
                    # xy basis functions
                    #self.A[i,j+n] = feint_old(lambda x: fd(self.phi,x-nodes[j])
                    #                          + 0*(self.phi(x-nodes[i]))*(self.phi(x-nodes[j]))/dt,
                    #                          self.h,self.interval)[0]
                    # yx basis functions
                    #self.A[i+n,j] = feint_old(lambda x: fd(self.phi,x-nodes[j])
                    #                         + 0*(self.phi(x-nodes[i]))*self.phi(x-nodes[j])/dt,
                    #                         self.h,self.interval)[0]
                    # yy basis functions
                    #self.A[i+n,j+n] = feint_old(lambda x: (self.phi(x-nodes[j]))/((lda**2) * (l**2))*(nodalapprox(x,t,rowConvolution)+self.phi(x-nodes[i]))*dt + (self.phi(x-nodes[j])*self.phi(x-nodes[i])),
                    #self.h,intrvv)[0]
                    self.A[j+n,i+n] = simpsonint(j, self.q[j]*phiArray )
                    if i==j:
                        self.A[j+n,i+n] += (1+dt)*1/(self.q[j])/h # prev 2
                    elif abs(i-j) == 1:
                        self.A[j+n,i+n] -= (1+dt)/(self.q[j])/h/2
                    elif i+j == n-1 and i*j==0: # this case accounts for when they each equal the end points, which are indentified.
                        self.A[j+n,i+n] -= (1+dt)/(self.q[j])/h/2
                    #if i==0 and j == 0:
                    #    print("(0,0)  -->  G:{},phi:{}".format(simpsonint(j,rowConvolution),simpsonint(j,phiArray)))

        
        # apply periodic boundary conditions:
        self.A[0,:] = (self.A[0,:] + self.A[n-1,:])
        self.A[n,:] = (self.A[n,:] + self.A[2*n-1,:])
        # Use last row (now put into first) to enforce the end terms matching
        self.A[n-1,:] = np.zeros(2*n)
        self.A[2*n-1,:] = np.zeros(2*n)
        self.A[n-1,[0,n-1]] = [1 , -1]
        self.A[2*n-1,[n,2*n-1]] = [1, -1]


    # NOTE try and find and organise subroutines
    # Main FEM routine
    def solve(self):
        self.updateDiscLengthVector()
        #self.generateGreensMatrix()
        global t
        # parameter space (usually denoted t).
        #x = np.linspace(interval[0]*m.pi,interval[1]*m.pi,1000)
        x = t
        nodes = t
        phix = np.ones(len(x))
        for i in range(len(x)):
            phix[i] = self.phi(x[i])
        y = np.ones(len(x))
        for i in range(len(x)):
            y[i] = fd(self.phi,x[i])

        # dependent on length, which varies with time
        self.generate_bilinear_form()
        A = self.A

        F = np.zeros(2*n)
        #print("dt check: {}".format(dt))
        #q = np.sqrt(self.u0x**2+u0y**2)
        #q[0] = (q[1]+q[n-2])/2
        #q[n-1]=q[0]
        qsum = 0

        
        for i in range(n):
            #F[i] = feint(lambda x: self.f(x)*self.phi(x-nodes[i]),h,self.interval)#[0]
            #F[i] = feint_old(lambda x: self.phi(x-nodes[i])*nodalapprox(x,t,self.u0x)/dt,h,self.interval)[0]
            #F[i+n] = feint_old(lambda x: self.phi(x-nodes[i])*nodalapprox(x,t,self.u0y)/dt,h,self.interval)[0]
            #q = n * np.sqrt((self.u0x[(i+1)%n]-self.u0x[i])**2+(self.u0y[(i+1)%n]-self.u0y[i])**2) # speed in last time step
            #if i == 0:
            #    q = n * np.sqrt((self.u0x[n-2]-self.u0x[0])**2+(self.u0y[n-2]-self.u0y[0])**2)/1
            #elif i==n-1:
            #    q = n * np.sqrt((self.u0x[n-2]-self.u0x[n-1])**2+(self.u0y[n-2]-self.u0y[n-1])**2)/1
            #qsum+=q
            F[i] = simpsonint(i,self.q[i]*self.u0x)
            F[i+n] = simpsonint(i,self.q[i]*self.u0y)


            fd1x = n*(self.u0x[(i)%n]-self.u0x[(i-1)%n])/2
            fd2x = n*(self.u0x[(i+1)%n]-self.u0x[(i)%n])/2
            fd1y = n*(self.u0y[(i)%n]-self.u0y[(i-1)%n])/2
            fd2y = n*(self.u0y[(i+1)%n]-self.u0y[(i)%n])/2
            
            
            #if i==j:
            F[i]   += (fd1x-fd2x)/(self.q[i])
            F[i+n] += (fd1y-fd2y)/(self.q[i])
            #elif abs(i-j) == 1:
            #    F[i] -= 1/(self.q[i])/h/2
            #    F[i+n] -= 1/(self.q[i])/h/2
            #elif i+j == n-1 and i*j==0: # this case accounts for when they each equal the end points, which are indentified.
            #    F[i] -= 1/(self.q[i])/h/2
            #    F[i+n] -= 1/(self.q[i])/h/2
                
            
        #for i in range(N):
        #    print(self.phi(t[i]-nodes[3]),end=", ")
        #    print("[!] - {},{}".format(i,F[i]))
        #input("pause! here is the F[i]")
        #print("'Mean' Curvature: {}".format(qsum/n))



        # POISSON PROBLEM

        #F[0] = self.dirichlet[0] # LHS boundary value
        #F[N-1] = self.dirichlet[1] # RHS boundary value

        #print(self.u0[0])
        #F[0] = self.u0x[0]
        #F[n-1] = self.u0x[n-1]
        #F[n] = self.u0y[0]
        #F[2*n-1] = self.u0y[n-1]


        # HEAT PROBLEM
        F[0] = F[0] + F[n-1]
        F[n] = F[n] + F[2*n-1]
        F[n-1] = 0
        F[2*n-1] = 0

        if self.debug:
            print("\n\nF:")
            print(F)

        #AB = np.array(np.zeros((3,N)))
        #for j in range(N):
        #    for i in range(N):
                #AB[i,j] = A[j+i-1,j]
        #        if 1+i-j < 3 and 1+i-j>=0:
        #            AB[1+i-j,j] = A[i,j]
        #if debug:
        #    print(AB)
        #U = sp.linalg.solve_banded((1,1),AB,F,check_finite=False)
        #print("\n")
        #print(A)
        #print(F)


        
        U=la.solve(self.A,F)
        #U = self.invA@F


        
        #print("U on the boundary: ({},{})".format(U[0],U[n-1]))
        #print("more dets on U: {},{},    {},{}".format(A[0,0],A[0,1],A[n-1,n-2],A[n-1,n-1]))
        #print("\n\n")
        #print(A)
        #print("\n\n")
        #print(F)
        #print("\n\n")
        if self.debug:
            print("\n\nU:")
            print(U)
            print(A@la.solve(A,F)-F)
            print("\n\nAU-F: ")
            print(A@U-F)

        #print("\n\nU:  ")
        #print(U)
            print("\n\ndt: {}".format(dt))

        #x = np.linspace(self.interval[0],self.interval[1],100)
        #x=t
        # This allows us to generate more points within the nodal basis approx
        #y = np.zeros(len(x))
        #for i in range(n-1):
            #print("{} \in ({},{})".format(x[i],getAdjacentNodeIndices(x[i])[0],getAdjacentNodeIndices(x[i])[1]))
        #    y[i] = (U[self.getAdjacentNodeIndices(x[i])[0]]*self.phi(x[i]+1-h*self.getAdjacentNodeIndices(x[i])[0]) + 
        #        U[self.getAdjacentNodeIndices(x[i])[1]]*self.phi(x[i]+1-h*self.getAdjacentNodeIndices(x[i])[1]))
        #y[n-1] /=2 # adjacent node for the last end node spits out the same val so you get double your result
        if self.debug:
            print("\n\n[!]")
            print(y)
            print("[!]")
        #return x,y

            # boundaries really need to match
        #print("U checks: {}, {}\n         {}, {}".format(U[0],U[n-1],U[n],U[2*n-1]))
        #U[n-1] = U[0]

        #self.u0 = np.copy(U)
        #self.u0x = np.copy(U[0:n])
        #self.u0y = np.copy(U[n:2*n])

        u0 = np.zeros((n,2))
        u0[:,0] = U[0:n]
        u0[:,1] = U[n:2*n]

        # keep the curve centered
        #u0[:,0] -= np.mean(u0[:,0])
        #u0[:,1] -= np.mean(u0[:,1])
        
        #ru0 = reparam(u0)
        #self.u0x = ru0[:,0]
        #self.u0y = ru0[:,1]

        self.u0x = u0[:,0]
        self.u0y = u0[:,1]
        
        #print(U)
        #print(F)

        # for area preservation
        #areaScaleFactor = self.area/self.getSignedArea()
        #U *= areaScaleFactor**(1/2)

        #self.u0x = np.copy(U[0:n])
        #self.u0y = np.copy(U[n:2*n])
        #print("Area is given by: {}".format(self.getSignedArea()))
            
        self.setQ() # update speeds with new u0x,u0y data
        
        return t,U[0:n],U[n:2*n] # prev was y


############################################################################################################################
# TESTING
############################################################################################################################


#plt.figure(figsize=(10,6))
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Test plot')
#plt.grid(True)

def func(x):
    return 0*(1-x**2)

#h = 0.005
#interval = [-1.0,1.0]
#dirichlet = [0.0,-1.0]

#myfem = fem(h,interval,dirichlet,func)

#x,y = myfem.solve()
#plt.plot(x,y)
#plt.show()

# HEAT

fig,ax = plt.subplots()
if not plotLength:
    ax.set(xlim=viewport.x,ylim=viewport.y,xlabel="x",ylabel="y")
else:
    ax.set(xlabel="x",ylabel="y")

# time step
#dt = 10**-2

# initial data
#t = np.linspace(0.0,1.0,int((1.0-0.0)/h))
# just a circle
u0x = np.zeros(len(t))
u0y = np.zeros(len(t))
u0x = 2*np.cos(2*m.pi*t)+0*0.6*np.cos(40*m.pi*t) #+0.1*np.sin(16*m.pi*t) #-np.ones(len(t))#+0.4*np.cos(4*m.pi*t)
u0y = 2*np.sin(2*m.pi*t)+2*np.sin(4*m.pi*t)#+0.1*np.cos(16*m.pi*t)#+0.4*np.sin(4*m.pi*t)

# for testing Sobolev against known solution
circlex = 0.25*0+np.cos(2*m.pi*t)
circley = 0.25*0+np.sin(2*m.pi*t)


useSpiral = False
useSquare = False
useCircle = False
useTriangle = True

if useCircle:
    u0x = circlex
    u0y = circley
elif useSquare:
    i = 0
    u0x[i] = 0
    u0y[i] = 0
    s = 10
    l = [1,2,3,4,5,6,7,8,9,10,11]

    u0x[1:12] = [x/(s+1) for x in l]
    u0y[1:12] = np.zeros(11)

    u0x[12:23] = np.ones(11)
    u0y[12:23] = [x/(s+1) for x in l]

    u0x[23:34] = [(s+1-x)/(s+1) for x in l]
    u0y[23:34] = np.ones(11)

    u0x[34:45] = np.zeros(11)
    u0y[34:45] = [(s+1-x)/(s+1) for x in l]

    # unconvex it
    #u0x[40] += 0.2

    print(u0x[23:33])

elif useSpiral:
    spin = 6 # some sort of even integer ig
    C1 = lambda x: np.array(m.sqrt(spin*m.pi*x)*np.array([m.cos(spin*m.pi*x),m.sin(spin*m.pi*x)]))
    C2 = lambda x: np.array(-m.sqrt((spin-1)*m.pi*(1-x))*np.array([m.cos((spin-1)*m.pi*(1-x)),m.sin((spin-1)*m.pi*(1-x))]))
    C3 = lambda x: ((C1(1)[0]-C2(0)[0])*np.array([m.cos(m.pi*x),m.sin(m.pi*x)]) + (C1(1)+C2(0)))/2
    #spiral = np.array(map(lambda x: C1(x/0.49) if x <= 0.49 else C3((x-0.49)/0.02) if x <= 0.51 else C2((x-0.51)/0.49),t))

    # transforms the 3 curves to fit a piecewise 0-1 parameter space and line up smoothly. We can move the splitter values
    # to change how many points each of the parts gets though if a reparam by arclength is done later this will matter but not as much
    def spiral(x):
        if x <= 0.45:
            return C1(x/0.45)
        elif x <= 0.55:
            return C3((x-0.45)/0.1)
        else:
            return C2((x-0.55)/0.45)

    for i in range(n):
        u0x[i] = spiral(t[i])[0]
        u0y[i] = spiral(t[i])[1]
elif useTriangle:
    # verts
    A = [1,0]
    B = [-1/2,m.sqrt(3)/2]
    C = [-1/2,-m.sqrt(3)/2]
    C1 = lambda x: np.array([x*A[0]+(1-x)*B[0], x*A[1]+(1-x)*B[1]])
    C2 = lambda x: np.array([x*B[0]+(1-x)*C[0], x*B[1]+(1-x)*C[1]])
    C3 = lambda x: np.array([x*C[0]+(1-x)*A[0], x*C[1]+(1-x)*A[1]])
    def triangle(x):
        if x<= 1/3:
            return C1(x*3)
        elif x <= 2/3:
            return C3((x-1/3)*3)
        else:
            return C2((x-2/3)*3)
    for i in range(n):
        u0x[i] = triangle(t[i])[0]
        u0y[i] = triangle(t[i])[1]
    # we do spiral

# We do some scaling
#u0x *= 4
#u0y *= 4

#i += 1
#while (i < 10):
#    u0x[i] = 4*t[i]
#    u0y[i] = 0
#    i += 1
#while(i < 20):
#    u0x[i] = 1
#    u0y[i] = 4*t[i-10]
#    i += 1
#while(i < 30):
#    u0x[i] = 1-4*t[i-20]
#    u0y[i] = 1
#    i += 1
#while(i < 40):
#    u0x[i] = 0
#    u0y[i] = 1-4*t[i-30]
#    i += 1
#u0x[n-1] = 0
#u0y[n-1] = 0
#u0x *= 4
#u0y *= 4

#u0x = np.sin(2*m.pi*t)+np.ones(n)*2
#u0y = 8*t*(1-t)

def heatFx(i):
    #return (u0x[i]-1)/dt # t should be u
    return u0x[i]/dt
def heatFy(i):
    #return (u0y[i]-1)/dt
    return u0y[i]/dt

#heatfemx = fem(h,[0.0,1.0],[1.0,1.0],heatFx,u0x,debug=False)
#heatfemy = fem(h,[0.0,1.0],[0.0,0.0],heatFy,u0y,debug=False)

classicfem = fem(h,[0.0,1.0],[1.0,1.0],heatFx,u0x,u0y,debug=False)

if chase:
    ax.set(xlim=[1.1*min(u0x),1.1*max(u0x)],ylim=[1.1*min(u0y),1.1*max(u0y)])

if plotLength:
    C = np.zeros((n,2))
    C[:,0] = u0x
    C[:,1] = u0y
    lengths = [length(C)]
else:
    curve = ax.plot(u0x,u0y,'-',label="FEM",color="black")[0]
if useCircle and False:
    if not plotLength:
        circle = ax.plot(circlex,circley,'o',label="true solution",color="blue")[0]
    else:
        trueLengths = [2*m.pi]
    ax.legend()
if genGif:
    gifout = animation.ImageMagickFileWriter(fps=10)
    gifout.setup(fig,"output_csf.gif")

#debugcurve=ax.plot(4*(t-1/2),0*t,label="debugcurve",color='black')[0]
#benchmark = ax.plot(t,np.cos(2*np.pi*t),label="true solution",color="red")[0]

#def phi_a(x):
#    output = np.zeros(len(x))
#    for i in range(len(x)):
#        if x[i]%1 >= 1-h and x[i]%1 < 1:
#            output[i] = 1+((x[i]%1)-1)/(h)
#        elif x[i]%1 >= 0 and x[i]%1 <= h:
#            output[i] = 1-(x[i]%1)/h
#        else:
#            output[i] = 0
#    return output
#phitest = ax.plot(t,phi_a(t),label="phi(t-frame)")[0]

#ax.plot(t,(u0x-1)/dt,label="F")
#uxc = ax.plot(t,u0x,label="ux")[0]
#uyc = ax.plot(t,u0y,label="uy")[0]

#ax.legend()

#phit = np.linspace(-10,10,5000)
#phity = np.zeros(5000)
#for i in range(5000):
#    phity[i] = heatfemx.phi(phit[i])
#ax.plot(phit,phity,label="phi")
#fig.show() useful for plotting the starting curve
def update(frame):
    global t
    t,ux = heatfemx.solve()
    #t,uy = heatfemy.solve()
    #ux = t
    #uy = t
    #print("silliest billy")
    #input("safety: ")

    #u0x = ux
    #u0y = uy

    #print(ux)
    #curve.set_xdata(ux)
    #curve.set_ydata(uy)

    #uxc.set_ydata(ux)
    #uyc.set_ydata(uy)

    #debugcurve.set_xdata(t+np.cos(frame))
    #debugcurve.set_ydata(t+np.sin(frame))

    if genGif:
        gifout.grab_frame()

    return (curve)

#input("pause..")
#ani = animation.FuncAnimation(fig=fig,func=update,frames=5,interval=1,repeat=False)
#update(1)
#update(1)
#update(1)
#update(1)
#plt.show()

if genGif:
    gifout.grab_frame()
frame = 0
print("length init: {}".format(len(u0x)))
frames = 200
frameNo = 100 # how many snapshots we want in image mode
for i in range(frames):
    #update(i)
    #t,ux = heatfemx.solve()
    #t,uy = heatfemy.solve()Though, we've used convolutions before on models which are local, any chance we can model the flow without the Greens function and kernel?

    t,ux,uy = classicfem.solve()
    if chase:
        ax.set(xlim=[1.1*min(ux),1.1*max(ux)],ylim=[1.1*min(uy),1.1*max(uy)])
    if i%1==0: # for more steps with less frames in gif
        if i%10 == 0:
            print("\n--------\nComputing Frame: {}".format(frame))
            debug = True
            print(classicfem.getDiscDist(0,n-1))
            debug = False

            print("lengths: {},{}".format(len(ux),len(uy)))
            C = np.zeros((n,2))
            C[:,0] = ux
            C[:,1] = uy
            print("inf(kappa): {}".format(min(curvatures(C))))
            print("Convex?: {}. [!] NOTE that this assumes nonself intersecting".format(isConvex(C)))
         
        #ux = t
        #uy = t
        #print("silliest billy")
        #input("safety: ")

        #print(u0y[7])
        #print(uy[7])
        #u0x = ux
        #u0y = uy

        #print(ux)
        if plotLength:
            C = np.zeros((n,2))
            C[:,0] = ux
            C[:,1] = uy
            lengths.append(length(C))
        elif genGif:
            curve.set_xdata(ux)
            curve.set_ydata(uy)
        elif i%(frames//frameNo)==0:
            ax.plot(ux,uy,color="black")

        if useCircle and False: #one day
            r0 = 1 # initial radius
            #a = 0.05 # investigate, ask Glen
            #lda = 1 # just use global def to keep same with FEM
            # assuming a!=2
            b = ((2*m.pi)**a)/(1+(2*m.pi*lda)**2)
            radius = (r0**(2-a) - (2-a)*b*(i+1)*dt)**(1/(2-a))
            if plotLength:
                trueLengths.append(radius*2*m.pi)
            elif genGif:
                circle.set_xdata(circlex*radius)
                circle.set_ydata(circley*radius)
            elif i%(frames//frameNo)==0:
                ax.plot(circlex*radius,circley*radius,'o',color="blue")
            print("True Solution Radius: {}".format(radius))
            print("True Solution Length: {}".format(radius*2*m.pi))

        #uxc.set_ydata(ux)
        #benchmark.set_ydata(np.exp(-dt*(frame+1))*(np.cos(2*np.pi*t)))
        #phitest.set_ydata(phi_a(t-frame/31))
        #uyc.set_ydata(uy)

        #debugcurve.set_xdata(t+np.cos(frame))
        #debugcurve.set_ydata(t+np.sin(frame))


        if genGif and i%(frames//frameNo)==0:
            gifout.grab_frame()
        #print(u0y[7])
        frame+=1
print("Program finished in {} seconds".format(round(1000*(time.time() - start))/1000))
if plotLength:
    ax.plot(np.linspace(0,(frames+1)*dt/2,frames+1),lengths,label="FEM") # highly suspect scaling happening
    if useCircle and False:
        ax.plot(np.linspace(0,frames+1,frames+1),trueLengths,label="True Solution")
if genGif:
    gifout.finish()
else:
    plt.show()
#print(heatfemy.u0[7])



