# load necessary modules
from keras_vggface.vggface import VGGFace
from keras.engine import  Model
from keras.layers import Input
from keras.preprocessing import image
import numpy as np
import os
from utils import utilities
from tqdm import tqdm


# Extract features from the layer fc7/relu
layer_name = 'fc7/relu'
vgg_model = VGGFace(model='vgg16', include_top=True, pooling='max')
out = vgg_model.get_layer(layer_name).output
vgg_model_new = Model(vgg_model.input, out)

input_folders = ['../../data/ferplus_rgb/Train224/', '../../data/ferplus_rgb/Valid224/', '../../data/ferplus_rgb/Test224/']
output_folders = ['../../data/transfer_learning/VGG16_Face_fc7/Train/', '../../data/transfer_learning/VGG16_Face_fc7/Valid/', '../../data/transfer_learning/VGG16_Face_fc7/Test/']


for i in range(3):
	# get the files
	images = os.listdir(input_folders[i])
	for img in tqdm(images):
		img_path = input_folders[i] + img
		im = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(im)
		x = np.expand_dims(x, axis=0)
		x = utilities.preprocess_input(x, version=1) # or version=2

		features = vgg_model_new.predict(x)
		np.save(output_folders[i] + img.split('.')[0] + '.npy', features)
