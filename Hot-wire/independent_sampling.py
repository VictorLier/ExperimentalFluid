import numpy as np
# Geometries
d_nozzle_exit = 0.1 # m
d_wire = 7*10^(-9) # m
l_wire = 0.003 # m

# Desired accuracy
epsilon = 0.05 # %

# Estimated velocity
U = 10 # m/s

# Conservative estimate of integral time scale
Tu = d_nozzle_exit/U # s

# Samling rate
fs = 2*Tu # Hz

# Estimate of number of independed samples
N_indep = np.ceil(1/np.sqrt(epsilon))

# T block
T_block = 2*Tu*N_indep # s

stop = True

