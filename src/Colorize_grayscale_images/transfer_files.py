import os
from shutil import copyfile

# input folders
gray_folder_train = 'Train32/'
gray_folder_valid = 'Valid32/'
gray_folder_test = 'Test32/'

# output folders
rgb_folder_train = 'Train32_rgb/'
rgb_folder_valid = 'Valid32_rgb/'
rgb_folder_test = 'Test32_rgb/'

files_gray_train = os.listdir(gray_folder_train)
labels_gray_train = [file for file in files_gray_train if 'lab' in file]

files_gray_valid = os.listdir(gray_folder_valid)
labels_gray_valid = [file for file in files_gray_valid if 'lab' in file]

files_gray_test = os.listdir(gray_folder_test)
labels_gray_test = [file for file in files_gray_test if 'lab' in file]

# Transfer files betwwen folders
for lab in labels_gray_train:
	src = gray_folder_train + lab
	dst = rgb_folder_train + lab
	copyfile(src, dst)

for lab in labels_gray_valid:
	src = gray_folder_valid + lab
	dst = rgb_folder_valid + lab
	copyfile(src, dst)

for lab in labels_gray_test:
	src = gray_folder_test + lab
	dst = rgb_folder_test + lab
	copyfile(src, dst)
