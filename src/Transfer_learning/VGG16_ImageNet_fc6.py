from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import os
from tqdm import tqdm

# Extract features from the layer fc6/relu
base_model = VGG16(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs= base_model.get_layer('fc1').output)

input_folders = ['../../data/ferplus_rgb/Train224/', '../../data/ferplus_rgb/Valid224/', '../../data/ferplus_rgb/Test224/']
output_folders = ['../../data/transfer_learning/VGG16_ImageNet_fc6/Train/', '../../data/transfer_learning/VGG16_ImageNet_fc6/Valid/', '../../data/transfer_learning/VGG16_ImageNet_fc6/Test/']

for i in range(3):
	# get the files
	images = os.listdir(input_folders[i])
	for img in tqdm(images):
		img_path = input_folders[i] + img
		im = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(im)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		features = model.predict(x)
		np.save(output_folders[i] + img.split('.')[0] + '.npy', features)
