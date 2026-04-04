from PIL import Image
import torch
import clip
import numpy as np
import os
from tqdm import tqdm
import cv2
import pickle

from helper import image_visu,finalsearch

with open("features.pkl",'rb') as f:
    embedding = pickle.load(f)
 
with open("files_path.pkl",'rb') as f:
    files_paths = pickle.load(f)
 

image = r"C:\Users\sahil\Downloads\TK_Photo.jpg"

# score_paths = [paths for paths , scores  in search(image)] for clip
score_paths = [paths for paths , scores  in finalsearch(image,embedding)] # for arcface

    
for i in score_paths:
    image_visu(i,window=i)