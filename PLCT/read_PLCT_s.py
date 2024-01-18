# -*- coding: utf-8 -*-
"""
Conversion of data files from Poul La Cour Tunnel

Created on Mon Jan  8 13:21:44 2024

@author: keme
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io

# define filename and number of pressure taps
filename = 'PLCT/Data/sNACA_FoilData/sNACA_Re3M_2.txt'
#filename = 'PLCT/Data/sNACA_FoilData/sNACA_Re5M_2.txt'
#filename = 'PLCT/Data/sNACA_FoilData/sNACA_Re3M_4.txt'
#filename = 'PLCT/Data/sNACA_FoilData/sNACA_Re5M_4.txt'
filename_lst = ['PLCT/Data/sNACA_FoilData/sNACA_Re3M_2.txt',
                'PLCT/Data/sNACA_FoilData/sNACA_Re5M_2.txt',
                'PLCT/Data/sNACA_FoilData/sNACA_Re3M_4.txt',
                'PLCT/Data/sNACA_FoilData/sNACA_Re5M_4.txt']

nf = 96   # number of pressure taps
LE = 46    # number of pressure taps on lower side 
           # (from tailing edge to leading edge)
AOA_lst = []
Cl_w_lst = []
Cl_f_lst = []
Cl_g_lst = []
Cd_lst = []

for filename in filename_lst:
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
    j = 1  # select case

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
    cord = 0.9
    span = 1.999

    # Wall Coefficient of lift - Cl
    #n = 48
    #wallPresGeoMat[:m,1], pWallPresMat[:m,j], 'ro-'
    #wallPresGeoMat[m:2*m,1], pWallPresMat[m:2*m,j],
    for j in range(3):
        Lower = 0
        Upper = 0

        for i in range(int(m)-1):
            Lower = Lower + (wallPresGeoMat[i+1,1]-(wallPresGeoMat[i,1])) * (pWallPresMat[i+1,j] + pWallPresMat[i,j])
            #print(i,wallPresGeoMat[i+1,1]-(wallPresGeoMat[i,1]),'Lower')


        for i in range(m, 2*m-1):
            Upper = Upper + (wallPresGeoMat[i+1,1]-(wallPresGeoMat[i,1])) * (pWallPresMat[i+1,j] + pWallPresMat[i,j])
            #print(i,wallPresGeoMat[i+1,1]-(wallPresGeoMat[i,1]))
        

        L_w = 1/2 * Lower - 1/2 * Upper

        rho = atm[j] / (287.058 * (tempVec[j]+273.15))

        Cl_w = L_w / (1/2 * rho * u0Vec[j]**2 * cord)
        Cl_w_lst.append(Cl_w)

        print("Wall coeffecient of lift is", Cl_w)


        #Foil coeffecient of lift
        F_px = 0
        F_py = 0
        for i in range(nf-1):
            F_px = F_px + 1/2*(pProfMat[i+1,j] + pProfMat[i,j])*(profOrifMat[i,2]-profOrifMat[i+1,2])
            F_py = F_py + 1/2*(pProfMat[i+1,j] + pProfMat[i,j])*(profOrifMat[i,1]-profOrifMat[i+1,1])
        
        L_f = F_py * np.cos(aoAVec[j]*np.pi/180) - F_px*np.sin(aoAVec[j]*np.pi/180)
        Cl_f = L_f / (1/2 * rho * u0Vec[j]**2 * cord )
        Cl_f_lst.append(Cl_f)

        print("Foil coeffecient of lift is", Cl_f)

        print(np.trapz(pProfMat[:,j]))


        #Gauge coeffecient of lift
        #Combined normal (To wing) force
        Fn = fgMat[0,j] + fgMat[2,j]

        #Combined Tangential (To wing force)
        Ft = fgMat[1,j] + fgMat[3,j]

        L_g = Fn * np.cos(aoAVec[j]*np.pi/180) - Ft * np.sin(aoAVec[j]*np.pi/180)

        Cl_g = L_g / (1/2 * rho * u0Vec[j]**2 * cord * span)
        Cl_g_lst.append(Cl_g)

        print("Gauge coeffecient of lift is",Cl_g)

        #Wake Rake coeffecient of drag
        Cd = 0
        for i in range(len(pTotWRMat[:,j])-1):

            u_i = np.sqrt(abs(pTotWRMat[i,j])/(0.5*rho))
            u_i1 = np.sqrt(abs(pTotWRMat[i+1,j])/(0.5*rho))
            Cd = Cd + (u_i/u0Vec[j]*(1-u_i/u0Vec[j])+u_i1/u0Vec[j]*(1-u_i1/u0Vec[j]))*(wrPTotOriMat[i+1,2]-wrPTotOriMat[i,2])/cord
        
        Cd_lst.append(Cd)

        AOA_lst.append(aoAVec[j])

        print("Wake Rake coeffecient of drag is", Cd)

        #theta = rho * u0Vec[j]**2 * cord* np.pr

T8_1 = pd.DataFrame({'AOA': AOA_lst[:3], 'Cl_w3M': Cl_w_lst[:3], 'Cl_w5M': Cl_w_lst[3:6], 'Cl_f3M': Cl_f_lst[:3], 'Cl_f5M': Cl_f_lst[3:6], 'Cl_g3M': Cl_g_lst[:3], 'Cl_g5M': Cl_g_lst[3:6], 'Cd3M': Cd_lst[:3], 'Cd5M': Cd_lst[3:6]}) 
T8_2 = pd.DataFrame({'AOA': AOA_lst[6:9], 'Cl_w3M': Cl_w_lst[6:9], 'Cl_w5M': Cl_w_lst[9:12], 'Cl_f3M': Cl_f_lst[6:9], 'Cl_f5M': Cl_f_lst[9:12], 'Cl_g3M': Cl_g_lst[6:9], 'Cl_g5M': Cl_g_lst[9:12], 'Cd3M': Cd_lst[6:9], 'Cd5M': Cd_lst[9:12]})
print(T8_1)
print(T8_2)
T8_1.to_csv('T8_1.m', sep='\t', index=False,header=False)
T8_2.to_csv('T8_2.m', sep='\t', index=False,header=False)


stop = True
