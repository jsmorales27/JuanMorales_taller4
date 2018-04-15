import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math

def gaussiana(a,sigma,t):
	curva=a*np.e**((-1/2)*((t/sigma)**2))
	curva=curva/np.trapz(curva)
	return curva

def fourier(matriz, tamano):
	N=float(matriz.shape[1])
	M=float(matriz.shape[0])
	G1=np.zeros(tamano)
	G = G1.astype(complex)
	for u in range (0, matriz.shape[0]):
		for v in range (0,matriz.shape[1]):
			sumax=[]
			for x in range (0, matriz.shape[0]-1):
				distribucionx=np.cos(2*math.pi*x*u/M)-1j*np.sin(2*math.pi*x*u/M)
				sumay=[]
				for y in range (0, matriz.shape[1]-1):
					distribuciony=np.cos(2*math.pi*y*v/N)-1j*np.sin(2*math.pi*y*v/N)
					sumay.append(matriz[x,y]*distribuciony)
				sumay_total=np.sum(np.array(sumay))
				sumax.append(sumay_total*distribucionx)
			sumax_total=np.sum(np.array(sumax))
			G[u,v]=sumax_total
	return G

img = plt.imread('cristiano.png')
print np.shape(img)

from scipy import fftpack
# Hacer la funcion gaussiana con una parametrizacion t
t = np.linspace(-10, 10, 256)
gauss1d = gaussiana(1.0,np.sqrt(10.0),t)

#curva gaussiana de dos dimensiones a partir de la multiplicacion de vectores gaussianos
gauss2d = gauss1d[:, np.newaxis] * gauss1d[np.newaxis, :]
print gauss1d[:, np.newaxis].shape
print gauss1d[np.newaxis, :].shape
print(gauss2d.shape)

# hacer la transformada de fourier bidimensional de la matriz gaussiana, con la dim de salida igual a dim img 
kernel_ft = fftpack.fft2(gauss2d, shape=img.shape[:2], axes=(0, 1))
gauss2d_transformada=fourier(gauss2d, np.shape(gauss2d))

plt.imshow(   abs(fftpack.fftshift(gauss2d_transformada)))
plt.savefig("fotogauss_trans.png")
plt.clf()


# convolve
img_ft = fftpack.fft2(img, axes=(0, 1))
img2_ft = kernel_ft[:, :, np.newaxis] * img_ft
img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real

plt.imshow(img2)
plt.savefig("capitano3.png")

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






