import numpy as np
from scipy import linalg
import cv2
import scipy
import scipy.misc
from PIL import Image
import numpy as np
import os
from tqdm import tqdm


##### Variaveis globais para realizar determinadas acoes #####
showImages = False
resizeImages = False

'''Funcao responsavel por receber um array de imagens, uma opcao para apresentar imagens e dar resize nas mesmas.
Para cada imagem do array sera calculada a sua matriz relativa a imagem esbranquicada e adicionada ao array de retorno.'''
def whiten_images(images, show, resize):
    global showImages, resizeImages
    showImages = show
    resizeImages = resize
    whitened_images = []
    for img in images:
        whitened_images.append(whiten_image(img))
    return whitened_images

'''Funcao responsavel por receber uma imagem e retornar a matriz equivalente a mesma, porem esbranquicada.'''
def whiten_image(img):
    x = loadData(img)
    width, height = x.shape
    shaped_x = reshapeImage(x)
    xPCAWhite, U = PCAWhitening(shaped_x)
    xZCAWhite = ZCAWhitening(xPCAWhite, U)
    final_xZCAWhite = shapeImageWhitened(xZCAWhite, width, height)
    return final_xZCAWhite

'''Funcao responsavel por ler a imagem original em tons de cinza, dar resize na mesma e apresenta-la 
(se as opcoes tiverem sido escolhidas). Retorna uma matriz 2D (pois transformamos a imagem para tons 
de cinza) NxM, onde N e M sao as dimensoes da imagem apos ter sido redimensionada (ou nao).'''
def loadData(img):
    img_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    if resizeImages:
        img_gray = cv2.resize(img_gray, (67, 540))
    
    if showImages: 
        showImage(img_gray, 'original')
    return img_gray
    
'''Funcao responsavel por receber a matriz no formato 1 x N*M de valores float e realizar os passos do
algoritmo de branqueamento do PCA. Sao eles: calcular sigma e seus autovalores (matriz U de rotacao, etc), 
achar xRot (dados rotacionados), xHat(dados em dimensao reduzida 1) e, finalmente, computar a matriz PCA 
utilizando a formula estabelecida. Retorna a matriz PCA e U.'''
def PCAWhitening(x):
    sigma = x.dot(x.T) / x.shape[1]
    U, S, Vh = linalg.svd(sigma)

    xRot = U.T.dot(x)

    #Reduz o numero de dimensoes de 2 pra 1
    k = 1
    xRot = U[:,0:k].T.dot(x)
    xHat = U[:,0:k].dot(xRot)
    
    epsilon = 1e-5
    xPCAWhite = np.diag(1.0 / np.sqrt(S + epsilon)).dot(U.T).dot(x) #formula do PCAWhitening
    return xPCAWhite, U

'''Funcao responsavel por retornar a matriz ZCAWhitening a partir da PCAWhitening e U, atraves
da formula estabelecida (UxPCAWhite).'''
def ZCAWhitening(xPCAWhite, U):
    xZCAWhite = U.dot(xPCAWhite) #formula da ZCAWhitening
    return xZCAWhite

'''Funcao responsavel por receber a matriz 2D NxM da imagem e retornar uma nova matriz 1 x N*M, sem alteracao
dos valores da mesma. Todos os valores foram transformados para float, pois quando lidamos com int temos problema 
de overflow em algumas contas.'''
def reshapeImage(img):
    vector = img.flatten(1)
    x = vector.reshape(1,len(vector))
    x = x.astype('float64')
    return x

'''Funcao responsavel por abrir uma janela do sistema com o titulo escolhido para apresentar a imagem do parametro.'''
def showImage(img, title):
    cv2.imshow(title,img)
    cv2.waitKey(0)

'''Funcao responsavel por transformar a matriz ZCAWhite que possui dimensao 1xN*M em uma nova matriz, equivalente a original
de dimensoes NxM, sem alteracao de seus valores. Apresenta a imagem final caso a opcao seja escolhida.'''
def shapeImageWhitened(xZCAWhite,width, height):
    reshaped = xZCAWhite.reshape(height,width)
    reshaped_t = reshaped.T
    if showImages:
        showImage(reshaped_t, 'whitened')
        
    return reshaped_t
    
    
if __name__ == '__main__':
    #whiten_images(['flower.jpg'], True, False)

	train_folder = '../data/Train48_contrast/'
	valid_folder = '../data/Valid48_contrast/'
	test_folder = '../data/Test48_contrast/'

	train_files = os.listdir(train_folder)
	valid_files = os.listdir(valid_folder)
	test_files = os.listdir(test_folder)

	train_images = [file for file in train_files if 'img' in file]
	valid_images = [file for file in valid_files if 'img' in file]
	test_images = [file for file in test_files if 'img' in file]

	# Training set
	for img in tqdm(train_images):
		tmp = whiten_images(['../data/Train48_contrast/' + img], False, False)[0]
		tmp = np.reshape(tmp, (48,48,1))
		np.save('../data/Train48_zca/' + img.split('.')[0] + '.npy', tmp)

	# Validation set
	for img in tqdm(valid_images):
		tmp = whiten_images(['../data/Valid48_contrast/' + img], False, False)[0]
		tmp = np.reshape(tmp, (48,48,1))
		np.save('../data/Valid48_zca/' + img.split('.')[0] + '.npy', tmp)

	# Test set
	for img in tqdm(test_images):
		tmp = whiten_images(['../data/Test48_contrast/' + img], False, False)[0]
		tmp = np.reshape(tmp, (48,48,1))
		np.save('../data/Test48_zca/' + img.split('.')[0] + '.npy', tmp)
