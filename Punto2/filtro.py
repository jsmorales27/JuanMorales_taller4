import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math

print "\n#############   Programa de filtrado de Imagenes CUADRADAS      ##############"

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

def pasa_bajas(imagen):
	img = plt.imread(imagen)
	plt.imshow(img)
	plt.savefig(imagen[:-4]+"_original.png")
	gauss2d=filtro(img.shape[1])
	gauss2d_fourier=fourier(gauss2d, np.shape(gauss2d))
	img_fourier = fourier(img, np.shape(img))
	from scipy import fftpack
	img2_fourier = gauss2d_fourier[:, :, np.newaxis] * img_fourier
	img2 = fftpack.ifft2(img2_fourier, axes=(0,1)) ### Uso de transformada inversa del paquete
	plt.imshow(fftpack.fftshift(img2).real)        ### Uso de fftshift para centrar (0,0)
	plt.title("SE ALCANZA A VER UN CHAPLIN DIFUSO")
	plt.savefig("pasa_bajas.png")

def pasa_altas(imagen):
	img = plt.imread(imagen)
	plt.imshow(img)
	plt.savefig(imagen[:-4]+"_original.png")
	gauss2d=filtro(img.shape[1])
	gauss2d_fourier=fourier(gauss2d, np.shape(gauss2d))
	img_fourier = fourier(img, np.shape(img))
	from scipy import fftpack
	#Normalizamos la gaussiana y solo tomamos la parte real, luego multiplicamos como un pasa bajas
	gauss2d_fourier_norm = np.abs(gauss2d_fourier.real) / np.abs(gauss2d_fourier.real).max()
	img2_parcial_fourier = gauss2d_fourier_norm[:, :, np.newaxis] * img_fourier
	# A la transformada inicial de la imagen le restamos la multiplicacion analoga al pasa bajas
	img2_fourier=img_fourier-img2_parcial_fourier
	
	img2 = fftpack.ifft2(img2_fourier, axes=(0,1)) ### Uso de transformada inversa del paquete
	plt.imshow(abs(img2))
	plt.title("SE ALCANZA A VER EL CONTORNO DE CHAPLIN")
	plt.savefig("pasa_altas.png")

def filtro (tamano):
	t = np.linspace(-200, 200, tamano)
	gauss1d = gaussiana(1.0,np.sqrt(500),t)
	gauss2d = gauss1d[:, np.newaxis] * gauss1d[np.newaxis, :]
	return gauss2d

tipo=raw_input("\nEscoga el filtro que desea (bajo o alto): ")

if (tipo=="bajo"):
	print "\n## Filtro pasa bajas"
	img=raw_input("Nombre del archivo de imagen cuadrada: " )
	print "Recuerde que el tamano maximo para un buen funcionamiento es 30x30 ....."
	pasa_bajas(img)
elif (tipo=="alto"):
	print "\n## Filtro pasa altas"
	img=raw_input("Nombre del archivo de imagen cuadrada: ")
	print "Recuerde que el tamano maximo para un buen funcionamiento es 30x30 ....."
	pasa_altas(img)
else:
	print "Valor de filtro no valido"
