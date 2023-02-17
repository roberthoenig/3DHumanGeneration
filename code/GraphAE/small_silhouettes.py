from os import listdir
from os.path import isfile, join
import natsort
from tqdm import tqdm
import shutil


silhouettes_path = "/home/robert/g/3DHumanGeneration/code/GraphAE/silhouettes/"
small_silhouettes_path = "/home/robert/g/3DHumanGeneration/code/GraphAE/small_silhouettes/"
img_filenames = [f for f in listdir(silhouettes_path) if isfile(join(silhouettes_path, f))]
img_filenames = natsort.natsorted(img_filenames,reverse=False)
arrays = []

for ctr, img_filename in tqdm(enumerate(img_filenames)):
    if ctr % 10 == 0:
        shutil.copyfile(silhouettes_path+img_filename, small_silhouettes_path+img_filename)