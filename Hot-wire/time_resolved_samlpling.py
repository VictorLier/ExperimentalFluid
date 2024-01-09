import numpy as np
# Geometries
d_nozzle_exit = 0.1 # m
d_wire = 7*10**(-9) # m
l_wire = 0.003 # m
d_pipe = 0.01 # m

#Air properties
nu = 0.00001512

# Desired accuracy
epsilon = 0.05 # %

# Estimated velocity
U = 10 # m/s

# Conservative estimate of integral time scale
Tu = d_nozzle_exit/U # s

# Samling rate
fs = min(U/(2*l_wire), U/(2*d_wire)) # Hz

# Estimate of number of independed samples
N_b = np.ceil(1/epsilon**2)

# T block
T_block = 200*Tu

# T tot
T_tot = T_block*N_b + 2*Tu*N_b # s

# Expected vortex frequency
Re = U * d_pipe / nu

St = 0.198*(1-19.7/Re)

f = St * U/d_pipe

stop = True


