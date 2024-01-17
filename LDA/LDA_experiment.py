import numpy as np
import matplotlib.pyplot as plt

l = 532e-9 # wavelength [m]
a = 310e-3 # [m]
b = 28e-3 # [m]

theta = 2*np.arctan(b/a)

df = l/(2*np.sin(theta/2))

print(df)
stop = True

