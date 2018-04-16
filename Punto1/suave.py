import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math

######### LOS RESULTADOS (_propios) SE PUEDEN COMPARAR CON LOS DEL CODIGO EN GITHUB (_fft) #################
#########                   Que estan en la otra carpeta (Resultados_fft)                  #################  

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
				sumay_total=np.sum(np.array(sumay), axis=0)
				sumax.append(sumay_total*distribucionx)
			sumax_total=np.sum(np.array(sumax), axis=0)
			G[u,v]=sumax_total
	return G

##############################  Funciona para imagenes cuadradas, es decir nxn ###############################
##############################  Funciona para imagenes pequenas, es decir n<40 ###############################
                             #       (para 60x60 ya toma entre 5min y 10 min)
print "\n######## Programa para suavizado de Imagenes CUADRADAS max(30x30) ###### \n"
imagen=raw_input("Nombre del archivo de imagen (gris.png): " )
img = plt.imread(imagen)
plt.imshow(img)
plt.savefig("original.png")

# Hacer la funcion gaussiana con una parametrizacion t
t = np.linspace(-200, 200, img.shape[1])
sigma=float(input("Introduzca la varianza de la gaussiana deseada (recomendada 500): "))
gauss1d = gaussiana(1.0,np.sqrt(sigma),t)

#curva gaussiana de dos dimensiones a partir de la multiplicacion de vectores gaussianos
gauss2d = gauss1d[:, np.newaxis] * gauss1d[np.newaxis, :]
print "Dimensiones de imagen : ", np.shape(img)
print "Dimensiones gaussiana : ", gauss2d.shape

# hacer la transformada de fourier bidimensional de la matriz gaussiana y la matriz de la imagen
gauss2d_fourier=fourier(gauss2d, np.shape(gauss2d))
img_fourier = fourier(img, np.shape(img))

from scipy import fftpack

# Multiplicar las matrices en el espacio de fourier y luego hacer la transformada inversa
img2_fourier = gauss2d_fourier[:, :, np.newaxis] * img_fourier

img2 = fftpack.ifft2(img2_fourier, axes=(0,1)) ##### USO DEL PAQUETE CON LA TRANSFORMADA INVERSA
                                                #####    (No se estudio en clase detenidamente)
plt.imshow(fftpack.fftshift(img2).real) ##### Uso del paquete para poner (0,0) en centro de la imagen
plt.savefig("suave.png")

#visualizar gaussiana dos dimensional y su transformada

plt.imshow(abs(fftpack.fftshift(img2_fourier)))
plt.savefig("imagen_fourier_filtrada.png")
plt.clf()

plt.imshow(gauss2d)
plt.savefig("gauss.png")
plt.clf()

plt.imshow(abs(fftpack.fftshift(gauss2d_fourier)))
plt.savefig("gauss_ft_propia.png")
plt.clf()







