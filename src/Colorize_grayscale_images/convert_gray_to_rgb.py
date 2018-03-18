import tensorflow as tf
import skimage.transform
from skimage.io import imsave, imread
import os
import time
from tqdm import tqdm

def load_image(path):
    img = imread(path)
    img = skimage.transform.resize(img, (224, 224))
    return img

# Input and output folder
# The inputs are the folders with the grayscale images
# Outputs are the folders where the rgb images are saved
gray_folders = ['../../data/ferplus/Train48/', '../../data/ferplus/Valid48/', '../../data/ferplus/Test48/']
rgb_folders = ['../../data/ferplus_rgb/Train224/', '../../data/ferplus_rgb/Valid224/', '../../data/ferplus_rgb/Test224/']

files_gray = os.listdir(gray_folders[2])
images_gray = [file for file in files_gray if 'img' in file]

print('Load pretrained model for grayscale to rgb conversion...')
with open("../../pretrained_models/colorize.tfmodel", mode='rb') as f:
    fileContent = f.read()
print('Pretrained Model Loaded!')

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
grayscale = tf.placeholder("float", [1, 224, 224, 1])
tf.import_graph_def(graph_def, input_map={ "grayscale": grayscale }, name='')

warnings = []

# Convert grayscale to RGB
with tf.Session() as sess:
    inferred_rgb = sess.graph.get_tensor_by_name("inferred_rgb:0")
    for image in tqdm(images_gray):
        try:
            img_gray = load_image(gray_folders[2] + image).reshape(1, 224, 224, 1)
            inferred_batch = sess.run(inferred_rgb, feed_dict={ grayscale: img_gray })
            imsave(rgb_folders[2] + image, inferred_batch[0])
        except:
            print('Warning')
            print(image)
            warnings.append(image)

# print images that are not converted 
print('Total warnings')
print(warnings)
