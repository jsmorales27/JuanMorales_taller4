import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math

def gaussiana(a,sigma,t):
	curva=a*np.e**((-1/2)*((t/sigma)**2))
	curva=curva/np.trapz(curva)
	return curva

def fourier(y):
	N=float(len(y))
	G=[]
	for n in range (0, len(y)):
		suma=[]
		n1=float(n)
		for k in range (0, len(y)-1):
			distribucion=np.cos(2*math.pi*k*n1/N)-1j*np.sin(2*math.pi*k*n1/N)
			suma.append(y[k]*distribucion)
		G.append(np.sum(np.array(suma)))
	return np.array(G)

img = plt.imread('totti.png')
print np.shape(img)

from scipy import fftpack
# Hacer la funcion gaussiana con una parametrizacion t
t = np.linspace(-10, 10, 30)
gauss1d = gaussiana(1.0,np.sqrt(10.0),t)

#curva gaussiana de dos dimensiones a partir de la multiplicacion de vectores gaussianos
gauss2d = gauss1d[:, np.newaxis] * gauss1d[np.newaxis, :]
print gauss1d[:, np.newaxis].shape
print gauss1d[np.newaxis, :].shape
print(gauss2d.shape)

# padded fourier transform, with the same shape as the image
kernel_ft = fftpack.fft2(gauss2d, shape=img.shape[:2], axes=(0, 1))

# convolve
img_ft = fftpack.fft2(img, axes=(0, 1))
img2_ft = kernel_ft[:, :, np.newaxis] * img_ft
img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real

plt.imshow(img2)
plt.savefig("capitano2.png")

#visualizar gaussiana dos dimensional
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
for x in range(0,30):
	for y in range(0,30):
		ax.scatter(x,y,gauss2d[x,y], s=5, color="red")
plt.savefig("gauss2d.png")
plt.clf()

plt.imshow(gauss2d)
plt.savefig("fotogauss.png")
plt.clf()

plt.imshow(   abs(fftpack.fftshift(kernel_ft)) )
plt.savefig("fotogaussft.png")
plt.clf()






