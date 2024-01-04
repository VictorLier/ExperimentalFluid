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
fs = U/(2*l_wire) # Hz

# Estimate of number of independed samples
N_b = np.ceil(1/np.sqrt(epsilon))

# T block
T_block = 1000*Tu*N_b # s

stop = True


