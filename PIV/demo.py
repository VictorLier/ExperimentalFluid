# demo of par2vel
import numpy as np
import matplotlib.pyplot as plt
from par2vel.camera import One2One
from par2vel.field import Field2D
from par2vel.artimage import ArtImage, OseenUfunc
from par2vel.piv2d import dxfield

cam = One2One((512,512))
cam.noise_mean = 0.05
cam.noise_rms = 0.01
ai = ArtImage(cam)
ai.random_particles(0.02)  # particle density in particles per pixel
ai.displace_particles(OseenUfunc(200,8,[256,256,0.0]), 1)
ai.generate_images()
fld = Field2D(cam)
fld.squarewindows(32, 0.5)
dxfield(ai.Im[0], ai.Im[1], fld)

plt.figure(1)
plt.clf()
plt.imshow(ai.Im[0], cmap='gray')
plt.figure(2)
plt.clf()
plt.imshow(ai.Im[1], cmap='gray')
plt.figure(3)
plt.clf()
plt.quiver(fld.x[0,:,:],fld.x[1,:,:],fld.dx[0,:,:],fld.dx[1,:,:], pivot='mid')
plt.axis('image')
plt.show()