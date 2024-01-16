# -*- coding: utf-8 -*-
"""
Conversion of data files from Poul La Cour Tunnel

Created on Mon Jan  8 13:21:44 2024

@author: keme
"""

import numpy as np
import matplotlib.pyplot as plt

# define filename and number of pressure taps
filename = 'PLCT/Data/sNACA_FoilData/sNACA_Re3M_2.txt'
#filename = 'PLCT/Data/sNACA_FoilData/sNACA_Re5M_2.txt'
#filename = 'PLCT/Data/sNACA_FoilData/sNACA_Re3M_4.txt'
#filename = 'PLCT/Data/sNACA_FoilData/sNACA_Re5M_4.txt'

nf = 96   # number of pressure taps
LE = 46    # number of pressure taps on lower side 
           # (from tailing edge to leading edge)

# read the data file
with open(filename) as file: 
    lines = file.readlines()
    
# handle first part with three columns
n1 = 234 + nf  # number of lines
data = np.zeros((n1, 3))
for i in range(n1):
    linefields = lines[i].split()
    for j in range(3):
        data[i, j] = float(linefields[j])
# split into variables
aoAVec  = data[0,:]   # AOA 
u0Vec   = data[1,:]   # Tunnel velocity [m/s]
tempVec = data[2,:]   # Temperature [C]
reVec   = data[3,:]   # Reynolds no.  [-]
q0Vec   = data[4,:]   # Pdyn tunnel ref [Pa]
atm     = data[5,:]   # Atmospheric pressure [Pa]
pProfMat = data[6:6+nf]  # AirWing tap pressure [Pa]
n = 6 + nf
pWallPresMat = data[n:n+128] # Wall pressures [Pa]
n += 128
pTotWRMat = data[n:n+96] # WakeRake total pressure       
n += 96
fgMat = data[n:n+4]     # Force gage
                        # (normal upstream, tangential upstream,
                        #  normal downstream, tangential downstream) 
n += 4
# handle second part with four columns
n2 = 224 + nf
data = np.zeros((n2, 4))
for i in range(n2):
    linefields = lines[n1 + i].split()
    for j in range(4):
        data[i, j] = float(linefields[j])
# split into variables
profOrifMat = data[0:nf]  # AirWing tap coords [m] (p/s,x,y,z)
n = nf
wallPresGeoMat = data[n:n+128]  # Wall tab coordinates [m]
n+= 128
wrPTotOriMat = data[n:n+96]    # WakeRake coords [m] (0,x,y,0)

# Plot some of the data
j = 2  # select case

n = 1
plt.figure(n)
plt.clf()
plt.plot(np.arange(3)+1, aoAVec, 'ko-')
plt.xlabel('Measurement no.')
plt.ylabel('AOA [deg]')
plt.title('Angle of Attach')

n+=1
plt.figure(n)
plt.clf()
plt.plot(profOrifMat[:LE,1], profOrifMat[:LE,2], 'ro-', markersize=2)
plt.plot(profOrifMat[LE:,1], profOrifMat[LE:,2], 'bo-', markersize=2)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Airfoil geometry')
plt.axis('image')


n+=1

plt.figure(n)
plt.clf()
plt.plot(profOrifMat[:LE,1], -pProfMat[:LE,j], 'ro-', markersize=2)
plt.plot(profOrifMat[LE:,1], -pProfMat[LE:,j], 'bo-', markersize=2)
plt.xlabel('x [m]')
plt.ylabel('Airfoil local pressure [Pa]')
plt.title('Airfoil pressure')

n+=1
m = 48 # fine resolution line
q = 16 # coarse resolution line
plt.figure(n)
plt.clf()
plt.plot(wallPresGeoMat[:m,1], pWallPresMat[:m,j], 'ro-', markersize=2)
plt.plot(wallPresGeoMat[m:2*m,1], pWallPresMat[m:2*m,j], 'bo-', markersize=2)
plt.plot(wallPresGeoMat[2*m:2*m+q,1], pWallPresMat[2*m:2*m+q,j], 'go-', markersize=2)
plt.plot(wallPresGeoMat[2*m+q:2*m+2*q,1], pWallPresMat[2*m+q:2*m+2*q,j], 'mo-', markersize=2)
plt.xlabel('x [m]')
plt.ylabel('wall local pressure [Pa]')
plt.title('PLCT wall pressure')

n+=1
plt.figure(n)
plt.clf()
plt.plot(pTotWRMat[:,j], -wrPTotOriMat[:,2], 'bo-', markersize=2)
#plt.axis([0, 5000, -0.4, 0.4])
plt.xlabel('Rake pressure [Pa]')
plt.ylabel('y [m]')
plt.title('Wake rake pressure')

#plt.show()

# Calculation of the Coeffecients
# Wall Coefficient of lift - Cl
Deltax = 0.1

n = 48

Lower = 0
Upper = 0
for i in range(n//2):
    Lower = Lower + Deltax * (pWallPresMat[i+1,j] + pWallPresMat[i,j])

for i in range(n//2, n):
    Upper = Upper + Deltax * (pWallPresMat[i+1,j] + pWallPresMat[i,j])

L = 1/2 * Lower - 1/2 * Upper

rho = atm[j] / (287.058 * (tempVec[j]+273.15))

Cl = L / (1/2 * rho * u0Vec[j]**2)

print("Wall coeffecient of lift is", Cl)


#Foil coeffecient of drag

