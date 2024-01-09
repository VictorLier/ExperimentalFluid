import numpy as np
# Geometries
d_nozzle_exit = 0.04733 # m
d_wire = 5*10**(-9) # m
l_wire = 0.00125 # m

# Desired accuracy
epsilon = 0.01 # %

# Estimated velocity
U = 10 # m/s

# Conservative estimate of integral time scale
Tu = d_nozzle_exit/U # s

# Sampling rate
fs = 1/(2*Tu) # Hz

# Estimate of number of independed samples
N_indep = np.ceil(1/epsilon**2)

# T tot
T_tot = 2*Tu*N_indep # s

stop = True


