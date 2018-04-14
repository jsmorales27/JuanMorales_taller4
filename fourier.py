import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math

n = 128 # numero de puntos en todo el intervalo
f = 120.0 #  frecuencia en Hz
dt = 1 / (f * 32 ) #32 samples per unit frequency
t = np.linspace( 0, (n-1)*dt, n)
y = np.cos(2 * np.pi * f * t)

plt.plot(t,y)
plt.savefig("coseno.png")
plt.clf()

def fourier(y, dt):
	N=float(len(y))
	G=[]
	sub_frecuencia=[]
	for n in range (0, len(y)):
		suma=[]
		n1=float(n)
		sub_frecuencia.append(n1/N)
		for k in range (0, len(y)-1):
			distribucion=np.cos(2*math.pi*k*n1/N)-1j*np.sin(2*math.pi*k*n1/N)
			suma.append(y[k]*distribucion)
		G.append(np.sum(np.array(suma)))
	fa=1.0/dt
	w = np.linspace(0, fa/2, len(G)/2, endpoint=False)
	return np.array(G)[:len(G)/2], w

transformada, frecuencias = fourier(y, dt)
plt.plot(frecuencias, np.abs(transformada))
plt.savefig("coseno_fourier.png")

i=transformada.argmax()
print frecuencias[i]
