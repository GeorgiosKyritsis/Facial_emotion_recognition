# load necessary modules
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
from tqdm import tqdm


# Extract features from the last layer of ResNet50
# and save them to one file per image
model = ResNet50(weights='imagenet', include_top=False)

input_folders = ['../../data/ferplus_rgb/Train224/', '../../data/ferplus_rgb/Valid224/', '../../data/ferplus_rgb/Test224/']
output_folders = ['../../data/transfer_learning/ResNet50_ImageNet/Train/', '../../data/transfer_learning/ResNet50_ImageNet/Valid/', '../../data/transfer_learning/ResNet50_ImageNet/Test/']

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
