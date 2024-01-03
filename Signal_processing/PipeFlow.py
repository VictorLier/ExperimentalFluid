from math import ceil


#Rough estimate of T_u = T_L
def RoughTu(Diameter, Centerline_Velocity):
    D = Diameter
    U_cl = Centerline_Velocity
    T_L = D / U_cl
    Time = T_L
    return Time


#Time length of one block - When TBlock >> 20 T_u - to avoid windowing
def TBlock(T_u):
    Time = T_u*20
    return Time

T_u = RoughTu(0.1,1)
print("Time Length for each block should be more than" ,TBlock(T_u), "sec.")


#Variability for the power spectrum is Epsilon_st^2 = 1/N_b
#N_b number of blocks

#For a variablility of xx % how many blocks is needed

def NumberOfRecords(Procent):
    N_b = ceil(1/Procent)
    return N_b 

print("For a variability of 5 procent the number of blocks have to be" ,NumberOfRecords(0.05), "For 1 procent it is", NumberOfRecords(0.01))



#Probe cut of frequency

def ProbeCutOff(Length, Width, Flowspeed):
    U = Flowspeed
    Fc_L = U/(2*Length)
    Fc_W = U/(2*Width)

    Fc = min(Fc_L,Fc_W)
    return Fc

print("The frequncy of the probe cut-off is", ProbeCutOff(0.001,0.0000005,1))
print("A Aliasing filter above the probe cut-off is recommended") #Tror Victor
