import cv2
import numpy as np

class pySaliencyImage:
	def __init__(self):
		return None
	#--------------------Extraccion de colores--------------------
	def SMExtractRGBI(self, inputImage):
		#Convierte la imagen en un array
		src = np.float32(inputImage) * 1./255
		#Regresa una lista de acuerdo al separador indicado (por defecto ' ')
		(B, G, R) = cv2.split(src)
		#Extrae la intensidad de la imagen 
		I = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) #Lo convierte a escala de grises
		return R, G, B, I
	#Mapa de rasgos intensificados
	def IFMGetFM(self, I): #Recibe la imagen en escala de grises
		return self.FMGaussianPyrCSD(I)
	#Construccion de una piramide gausiana y toma de diferencias centro elvonvente
	def FMGaussianPyrCSD(self, src):
		GaussianMaps = self.FMCreateGaussianPyr(src) #Contiene la imagen en escala de grises y la imagen reducida en escala de grises
		dst = self.FMCenterSurroundDiff(GaussianMaps)
		return dst
	#--------------------Mapa de caracteristicas--------------------
	#Construyendo la piramide gausiana
	def FMCreateGaussianPyr(self, src):
		dst = list()
		dst.append(src) #Agrega la imagen gris a la lista
		for i in range(1,9): #Empieza en 1 y termina en 8
			nowdst = cv2.pyrDown(dst[i-1]) #Reduce el tamano y resolucion de la imagen
			dst.append(nowdst) #Agrega a la lista la nueva imagen reducida
		return dst
	##Toma de diferencias centro elvonvente
	def FMCenterSurroundDiff(self, GaussianMaps):
		dst = list()
		for s in range(2,5): #Empieza en 2 y termina en 4
			now_size = GaussianMaps[s].shape #Crea un arreglo con las dimensiones de la imagen
			now_size = (now_size[1], now_size[0]) #(width, height)
			tmp = cv2.resize(GaussianMaps[s+3], now_size, interpolation=cv2.INTER_LINEAR) #Fuente, tamano deseado, remuestreo
			nowdst = cv2.absdiff(GaussianMaps[s], tmp) #Diferencia entre GaussianMaps y tmp
			dst.append(nowdst) #Agrega el elemento a la lista
			tmp = cv2.resize(GaussianMaps[s+4], now_size, interpolation=cv2.INTER_LINEAR)
			nowdst = cv2.absdiff(GaussianMaps[s], tmp)
			dst.append(nowdst)
		return dst