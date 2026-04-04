# from PIL import Image
# import torch
# import clip
# import numpy as np
# import os
# from tqdm import tqdm
# import cv2
# import pickle

# from helper import image_visu,finalpreprocess

# # model, preprocess = clip.load("ViT-B/32")


# #File Path Making
# files_path=[]
# base_path = ['D:\pc_images','D:\mobile_img']
# for j in base_path:
#     for i in tqdm(os.listdir(j)):
#         filese = os.path.join(j,i)
#         print(filese)
#         files_path.append(filese)
  
  
# #Embeding Storing
# img_db = []
# for path in tqdm(files_path):
#   embd = finalpreprocess(path)
#   img_db.append((path,embd))
  
  
# pickle.dump(files_path,open('files_path.pkl','wb'))
# pickle.dump(img_db,open('features.pkl','wb'))

# print("done✅✅✅")

# CLAUDE CODE LETS TRY THESE

from PIL import Image
import torch
import numpy as np
import os
from tqdm import tqdm
import pickle
import gc

from helper import finalpreprocess, load_db

# ─── File Paths Collect karo ──────────────────────────────────────────────────
files_path = []
base_path = [r'D:\pc_images', r'D:\mobile_img']

for folder in base_path:
    if not os.path.exists(folder):
        print(f"Folder nahi mila: {folder}")
        continue
    for filename in tqdm(os.listdir(folder), desc=f"Scanning {folder}"):
        full_path = os.path.join(folder, filename)
        # Sirf image files lo
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
            files_path.append(full_path)

print(f"\nTotal images found: {len(files_path)}")

# ─── Embedding Building ───────────────────────────────────────────────────────
img_db = []
failed = []

for i, path in enumerate(tqdm(files_path, desc="Processing images")):
    embd = finalpreprocess(path)
    
    if embd is not None:
        img_db.append((path, embd))
    else:
        failed.append(path)

    # Har 200 images pe memory clear karo
    if i % 200 == 0 and i > 0:
        gc.collect()
        torch.cuda.empty_cache()
        # Progress save karo (crash hone pe bhi data safe rahe)
        pickle.dump(img_db, open('features.pkl', 'wb'))
        pickle.dump(files_path, open('files_path.pkl', 'wb'))
        print(f"\n💾 Checkpoint saved at {i}/{len(files_path)}")

# ─── Final Save ───────────────────────────────────────────────────────────────
pickle.dump(files_path, open('files_path.pkl', 'wb'))
pickle.dump(img_db, open('features.pkl', 'wb'))

print(f"\n✅ Done!")
print(f"   Successful : {len(img_db)}")
print(f"   Failed     : {len(failed)}")
if failed:
    print(f"   Failed files: {failed[:5]}{'...' if len(failed)>5 else ''}")




