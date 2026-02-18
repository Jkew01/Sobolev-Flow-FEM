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
outfile = TemporaryFile()

# GLOBAL VARS
n = 100
h = 1/n
t = np.linspace(0.0,1.0,n)
dt = 1 # 10**-1


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
        if a > intrv[1]+10**-14:
            return "[!] Error - interval overshoot - {}".format(a) # do proper error handling

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

# sign function: returns -1 or 1. Note: we assume 0 is positive
def sgn(x):
    if x<0:
        return -1
    return 1

class Viewport:
    x = [-1.4,1.4]
    y = [-1.4,1.4]
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
    #print("lower: {}\nupper: {}\n---------".format(lower,upper))
    return (data[upper]-data[lower])*interpol+data[lower]


print(nodalapprox(.36,[0.0,0.1,0.2,0.3,0.4],[0,1,20,40,-40]))
print("\n\n---------------------\nEND OF UNIT TEST\n---------------------\n\n")


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
        self.interval = interval
        self.dirichlet = dirichlet
        self.f = f
        self.debug = debug
        self.A = np.zeros((2*n,2*n)) # 2 times for each set of nodal basis functions corresponding to x and y
        #self.generate_bilinear_form(regen=True) # set to true when changing bilinear form
        self.generate_bilinear_form()
        self.invA = la.inv(self.A)
        self.area = self.getSignedArea()

    # UTILS

    # returns the nodes which most closely sandwich the given value
    def getAdjacentNodeIndices(self,x):
        return m.floor((x-self.interval[0])/self.h),m.ceil((x-self.interval[0])/self.h)

    def getSignedArea(self):
        fh=0.00001
        return feint_old(lambda x: (nodalapprox(x+fh,t,self.u0y) - nodalapprox(x,t,self.u0y))/fh * nodalapprox(x,t,self.u0x) -
                         (nodalapprox(x+fh,t,self.u0x) - nodalapprox(x,t,self.u0x))/fh * nodalapprox(x,t,self.u0y),
                         self.h,[0,1])[0]/2

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
                        A[i,j] = feint_old(lambda x: fd(self.phi,x-nodes[i])*fd(self.phi,x-nodes[j])
                                           + (self.phi(x-nodes[i]))*self.phi(x-nodes[j])/dt,
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


    # generates the x and y bilinear forms according to current data
    def generate_bilinear_form(self):
        nodes = t
        self.A = np.zeros((2*n,2*n))
        print(sgn(fd(self.phi,0.01)))
        print(sgn(fd(self.phi,-0.01)))
        for i in range(n):
            for j in range(n):
                if abs(i-j) <= 1 or (i*j == 0 and i+j==n-1):
                    # xx basis functions
                    foo1 = feint_old(lambda x: sgn(fd(self.phi,x-nodes[j]))*fd(self.phi,x-nodes[i]),self.h,self.interval)[0]
                    foo2 = feint_old(lambda x: abs(fd(self.phi,x-nodes[i]))*(self.phi(x-nodes[i]))*self.phi(x-nodes[j])/(dt),self.h,self.interval)[0]

                    self.A[i,j] = feint_old(lambda x: sgn(fd(self.phi,x-nodes[j]))*fd(self.phi,x-nodes[i])/dt
                                            + abs(fd(self.phi,x-nodes[i]))*(self.phi(x-nodes[i]))*self.phi(x-nodes[j]) ,#/(dt),
                                            self.h,self.interval)[0]
                    # xy basis functions
                    #self.A[i,j+n] = feint_old(lambda x: fd(self.phi,x-nodes[j])
                    #                          + 0*(self.phi(x-nodes[i]))*(self.phi(x-nodes[j]))/dt,
                    #                          self.h,self.interval)[0]
                    # yx basis functions
                    #self.A[i+n,j] = feint_old(lambda x: fd(self.phi,x-nodes[j])
                    #                         + 0*(self.phi(x-nodes[i]))*self.phi(x-nodes[j])/dt,
                    #                         self.h,self.interval)[0]
                    # yy basis functions
                    self.A[i+n,j+n] = feint_old(lambda x: sgn(fd(self.phi,x-nodes[j]))*fd(self.phi,x-nodes[i])/dt
                                              + abs(fd(self.phi,x-nodes[i]))*(self.phi(x-nodes[i])*self.phi(x-nodes[j])) ,#/(dt),
                                              self.h,self.interval)[0]

        # apply periodic boundary conditions:
        self.A[0,:] = self.A[0,:] + self.A[n-1,:]
        self.A[n,:] = self.A[n,:] + self.A[2*n-1,:]
        # Use last row (now put into first) to enforce the end terms matching
        self.A[n-1,:] = np.zeros(2*n)
        self.A[2*n-1,:] = np.zeros(2*n)
        self.A[n-1,[0,n-1]] = [1 , -1]
        self.A[2*n-1,[n,2*n-1]] = [1, -1]
        #print(self.A)
                    

    # NOTE try and find and organise subroutines
    # Main FEM routine
    def solve(self):
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


        #self.generate_bilinear_form()
        A = self.A

        F = np.zeros(2*n)
        print("dt check: {}".format(dt))
        for i in range(n):
            #F[i] = feint(lambda x: self.f(x)*self.phi(x-nodes[i]),h,self.interval)#[0]
            F[i] = feint_old(lambda x: abs(fd(self.phi,x-nodes[i]))*nodalapprox(x,t,self.u0x)*self.phi(x-nodes[i])/(dt**0),h,self.interval)[0]
            F[i+n] = feint_old(lambda x: abs(fd(self.phi,x-nodes[i]))*nodalapprox(x,t,self.u0y)*self.phi(x-nodes[i])/(dt**0),h,self.interval)[0]
        #for i in range(N):
        #    print(self.phi(t[i]-nodes[3]),end=", ")
        #    print("[!] - {},{}".format(i,F[i]))
        #input("pause! here is the F[i]")



        # POISSON PROBLEM

        #F[0] = self.dirichlet[0] # LHS boundary value
        #F[N-1] = self.dirichlet[1] # RHS boundary value

        #print(self.u0[0])
        #F[0] = self.u0[0]
        #F[N-1] = self.u0[N-1]


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


        
        #U=la.solve(self.A,F)
        U = self.invA@F


        
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
        print("U checks: {}, {}\n         {}, {}".format(U[0],U[n-1],U[n],U[2*n-1]))
        #U[n-1] = U[0]

        #self.u0 = np.copy(U)
        self.u0x = np.copy(U[0:n])
        self.u0y = np.copy(U[n:2*n])
        #print(U)
        #print(F)

        # for area preservation
        #areaScaleFactor = self.area/self.getSignedArea()
        #U *= areaScaleFactor**(1/2)

        #self.u0x = np.copy(U[0:n])
        #self.u0y = np.copy(U[n:2*n])
        #print("Area is given by: {}".format(self.getSignedArea()))
            
        
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
ax.set(xlim=viewport.x,ylim=viewport.y,xlabel="x",ylabel="y")

# time step
#dt = 10**-2

# initial data
#t = np.linspace(0.0,1.0,int((1.0-0.0)/h))
# just a circle
u0x = np.zeros(len(t))
u0y = np.zeros(len(t))
u0x = np.cos(2*m.pi*t) #+0.1*np.sin(16*m.pi*t) #-np.ones(len(t))#+0.4*np.cos(4*m.pi*t)
u0y = np.sin(2*m.pi*t)+np.sin(4*m.pi*t)#+0.1*np.cos(16*m.pi*t)#+0.4*np.sin(4*m.pi*t)

def heatFx(i):
    #return (u0x[i]-1)/dt # t should be u
    return -u0x[i]/dt
def heatFy(i):
    #return (u0y[i]-1)/dt
    return -u0y[i]/dt

#heatfemx = fem(h,[0.0,1.0],[1.0,1.0],heatFx,u0x,debug=False)
#heatfemy = fem(h,[0.0,1.0],[0.0,0.0],heatFy,u0y,debug=False)

classicfem = fem(h,[0.0,1.0],[1.0,1.0],heatFx,u0x,u0y,debug=False)

curve = ax.plot(u0x,u0y,label="curve")[0]
gifout = animation.ImageMagickFileWriter(fps=10)
gifout.setup(fig,"output_classic.gif")

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

    uxc.set_ydata(ux)
    #uyc.set_ydata(uy)

    #debugcurve.set_xdata(t+np.cos(frame))
    #debugcurve.set_ydata(t+np.sin(frame))


    gifout.grab_frame()

    return (curve)

#input("pause..")
#ani = animation.FuncAnimation(fig=fig,func=update,frames=5,interval=1,repeat=False)
#update(1)
#update(1)
#update(1)
#update(1)
#plt.show()

gifout.grab_frame()
frame = 0
print("length init: {}".format(len(u0x)))
for i in range(40):
    print("\n--------\nComputing Frame: {}".format(frame))
    #update(i)
    #t,ux = heatfemx.solve()
    #t,uy = heatfemy.solve()
    t,ux,uy = classicfem.solve()

    print("lengths: {},{}".format(len(ux),len(uy)))
 
    #ux = t
    #uy = t
    #print("silliest billy")
    #input("safety: ")

    #print(u0y[7])
    #print(uy[7])
    #u0x = ux
    #u0y = uy

    #print(ux)
    curve.set_xdata(ux)
    curve.set_ydata(uy)

    #uxc.set_ydata(ux)
    #benchmark.set_ydata(np.exp(-dt*(frame+1))*(np.cos(2*np.pi*t)))
    #phitest.set_ydata(phi_a(t-frame/31))
    #uyc.set_ydata(uy)

    #debugcurve.set_xdata(t+np.cos(frame))
    #debugcurve.set_ydata(t+np.sin(frame))


    gifout.grab_frame()
    #print(u0y[7])
    frame+=1
gifout.finish()
print("foo")
#print(heatfemy.u0[7])



