import os
from shutil import copyfile
from tqdm import tqdm

in_folder_train = '../data/Train48/'
in_folder_valid = '../data/Valid48/'
in_folder_test = '../data/Test48/'

out_folder_train = '../data/Train48_zca/'
out_folder_valid = '../data/Valid48_zca/'
out_folder_test = '../data/Test48_zca/'

files_gray_train = os.listdir(in_folder_train)
labels_gray_train = [file for file in files_gray_train if 'lab' in file]
print(len(labels_gray_train))

files_gray_valid = os.listdir(in_folder_valid)
labels_gray_valid = [file for file in files_gray_valid if 'lab' in file]
print(len(labels_gray_valid))

files_gray_test = os.listdir(in_folder_test)
labels_gray_test = [file for file in files_gray_test if 'lab' in file]
print(len(labels_gray_test))

for lab in tqdm(labels_gray_train):
	src = in_folder_train + lab
	dst = out_folder_train + lab
	copyfile(src, dst)

for lab in tqdm(labels_gray_valid):
	src = in_folder_valid + lab
	dst = out_folder_valid + lab
	copyfile(src, dst)

for lab in tqdm(labels_gray_test):
	src = in_folder_test + lab
	dst = out_folder_test + lab
	copyfile(src, dst)

