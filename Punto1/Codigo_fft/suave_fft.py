import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math

####################### CODIGO DEL REPOSITORIO (usado para comparar resultados) ##################
######## vemos que las imagenes producidas de nuestra implementacion (suave.png) 
# y las de este codigo (suave_fft.png) SON LAS MISMAS                       ##########

from scipy import fftpack
from scipy.signal import convolve2d
# read image
img = plt.imread('gris.png')
print(np.shape(img))

# prepare an 1-D Gaussian convolution kernel
t = np.linspace(-200, 200, 30)
bump = np.exp(-(1.0*(10**-3))*t**2)
bump = bump / np.trapz(bump) # normalize the integral to 1

# make a 2-D kernel out of it
kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
print(img.shape[:2])

# padded fourier transform, with the same shape as the image
kernel_ft = fftpack.fft2(kernel, shape=img.shape[:2], axes=(0, 1))

# convolve
img_ft = fftpack.fft2(img, axes=(0, 1))
img2_ft = kernel_ft[:, :, np.newaxis] * img_ft
img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real

plt.imshow(fftpack.fftshift(img2))
plt.savefig("suave_ftt.png")


