from PIL import Image
import os
import shutil
import sys

data_dir = "../data/torch_data/"

garbage_data_dir = "../data/garbage_data/"

def clean_images(dir_name):
    for file in os.listdir(dir_name):
        try:
            file_path = dir_name+"/"+file
            im = Image.open(file_path)
            width,height = im.size
            im.close()
            if(width!=350 or height!=350):
                shutil.move(file_path,garbage_data_dir+"/"+file)
        except:
            print("Error while processing file %s"%(file_path))
            print("Unexpected error:", sys.exc_info()[0])

for dir_name in os.listdir(data_dir):
    dir_path = data_dir+dir_name
    clean_images(dir_path)