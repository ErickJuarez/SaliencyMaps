import cv2
import sys
import matplotlib.pyplot as plt
import pySaliencyMap
from os import scandir

def ls2(path): 
    return [obj.name for obj in scandir(path) if obj.is_file() if obj.name.endswith('.jpg')]

def mapas(file):
	try:
	    #--------------------Lee el archivo--------------------
	    img = cv2.imread('images/imagesEntrenar/'+file)
	    #img = cv2.imread('images/imagesEntrenar/'+sys.argv[1]+'.jpg')

	    #--------------------Inicializa las variables--------------------
	    imgsize = img.shape #Crea un arreglo con las dimensiones de la imagen
	    img_height = imgsize[0] #Asigna el valor de altura de la posicion 0 de imgsize
	    img_width  = imgsize[1] #Asigna el valor de ancho de la posicion 1 de imgsize
	    sm = pySaliencyMap.pySaliencyMap(img_width, img_height) #Inicializa la clase

	    #--------------------Llamada a los metodos de obtencion de rasgos--------------------
	    saliency_map = sm.SMGetSM(img) #Obtiene el mapa de poderacion
	    binarized_map = sm.SMGetBinarizedSM(img) #Obtiene el mapa binarizado
	    salient_region = sm.SMGetSalientRegion(img) #Obtiene la region poderada

	    #--------------------Ploteo del resultado de lops metodos de obtencion de rasgos--------------------
	    #plt.subplot(2,2,1), plt.imshow(img, 'gray') #Original en escala de grises
	    plt.subplot(2,2,1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #Original
	    plt.title('Imagen de entrada')
	    #cv2.imshow("input",  img) #Muestra imagen en grande
	    plt.subplot(2,2,2), plt.imshow(saliency_map, 'gray')
	    plt.title('Mapa de ponderacion')
	    #cv2.imshow("output", map) #Muestra imagen en grande
	    plt.subplot(2,2,3), plt.imshow(binarized_map)
	    plt.title('Mapa de ponderacion binarizada')
	    #cv2.imshow("Binarized", binarized_map) #Muestra imagen en grande
	    plt.subplot(2,2,4), plt.imshow(cv2.cvtColor(salient_region, cv2.COLOR_BGR2RGB))
	    cv2.imwrite('images/mapas/'+file,salient_region)
	    #cv2.imwrite('images/mapas/'+sys.argv[1]+'.jpg',salient_region)
	    plt.title('Region ponderada')
	    #cv2.imshow("Segmented", segmented_map) #Muestra imagen en grande

	    #plt.show() #Muestra las 4 imagenes

	    #--------------------Cierra el programa--------------------
	    #cv2.waitKey(0)
	    #cv2.destroyAllWindows()
	except:
	 	print("Imagen muy pequena")
#Main
if __name__ == '__main__':
    files=ls2("images/imagesEntrenar/")
    for file in files:
        print(file)
        mapas(file)