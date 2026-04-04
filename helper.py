# from PIL import Image
# import torch
# import clip
# from deepface import DeepFace
# import numpy as np
# import os
# from tqdm import tqdm
# import cv2
# import pickle


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# with open("features.pkl",'rb') as f:
#     embedding = pickle.load(f)
 
# with open("files_path.pkl",'rb') as f:
#     files_paths = pickle.load(f)



# #MOdel Loading
# model, preprocess = clip.load("ViT-B/32",device=device) #Clip Model

# # model = ArcFace.ArcFace()
# # emb1 = face_rec.calc_emb("~/Downloads/test.jpg")


# #Image Visualize
# def image_visu(path,window):
#     img = cv2.imread(path)

#     cv2.imshow(window,cv2.resize(img,(512,512)))
#     cv2.waitKey(0)

# # image Preprocess clip
# def prep(path):

#   image_input = preprocess(Image.open(path)).unsqueeze(0).to(device)
#   with torch.no_grad():
#     img_features = (model.encode_image(image_input))[0] #item changed

#   return img_features / img_features.norm(dim=-1, keepdim=True)

# # image prerocess arcface
# def arcfacepreprocess(path):
#   embedding_result = DeepFace.represent(path, model_name="ArcFace",enforce_detection=False)
#   if embedding_result:
#     embedding = np.array(embedding_result[0]['embedding'])
#     return embedding / np.linalg.norm(embedding)
#   else:
#       print(f"Warning: No face detected in {path}. Skipping.")
#       return None

# #final preprocess

# def finalpreprocess(path):

#   try:
#     arc = arcfacepreprocess(path)
#     arc = np.array(arc)
#     print("Arc FAce used")
#     # img_db.append((path,arc))
#     return arc
    
#   except Exception as e:

#     clips =prep(path)
#     clips = np.array(clips)
#     print("clip used")
#     # img_db.append((path,clips))
#     return clips
#   else:
#     print("something went wrong")
#     return None




# #Image Search CLIP
# # def search(query_path):
# #     q_emb = prep(query_path)

# #     scores = []
# #     for path, emb in embedding:
# #         sim = (q_emb @ emb.T).item()  # cosine similarity
# #         scores.append((path, sim))

# #     scores.sort(key=lambda x: x[1], reverse=True)
# #     return scores[:10]

# #Image Search final
# def finalsearch(query_path,img_db):
#     q_emb = finalpreprocess(query_path)

#     scores = []
#     for path, emb in img_db:
#         sim = (q_emb @ emb.T).item()  # cosine similarity
#         scores.append((path, sim))

#     scores.sort(key=lambda x: x[1], reverse=True)
#     return scores[:10]



#CLAUDE code lets try


from PIL import Image
import torch
import clip
from deepface import DeepFace
from deepface.commons import weight_utils
from deepface.modules import modeling as deepface_modeling
import numpy as np
import os
import cv2
import pickle
import gc

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── Models — ek baar load karo, global ───────────────────────────────────────
print("Loading CLIP model...")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

print("Loading ArcFace model...")
arcface_model = deepface_modeling.build_model(task="facial_recognition", model_name="ArcFace")
print("All models loaded ✅")

# ─── DB Load (safe) ───────────────────────────────────────────────────────────
def load_db():
    embedding, files_paths = [], []
    if os.path.exists("features.pkl"):
        with open("features.pkl", "rb") as f:
            embedding = pickle.load(f)
    if os.path.exists("files_path.pkl"):
        with open("files_path.pkl", "rb") as f:
            files_paths = pickle.load(f)
    return embedding, files_paths

# ─── Image Visualize ──────────────────────────────────────────────────────────
def image_visu(path, window):
    img = cv2.imread(path)
    cv2.imshow(window, cv2.resize(img, (512, 512)))
    cv2.waitKey(0)

# ─── CLIP Preprocess ──────────────────────────────────────────────────────────
def prep(path):
    image_input = preprocess(Image.open(path)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_features = clip_model.encode_image(image_input)[0]
    feat = img_features / img_features.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()  # numpy mein return karo

# ─── ArcFace Preprocess ───────────────────────────────────────────────────────
def arcfacepreprocess(path):
    embedding_result = DeepFace.represent(
        path,
        model_name="ArcFace",
        enforce_detection=False,
        detector_backend="opencv",   # fast + lightweight
        align=True,
    )
    if embedding_result:
        embedding = np.array(embedding_result[0]["embedding"])
        return embedding / np.linalg.norm(embedding)
    return None

# ─── Final Preprocess ─────────────────────────────────────────────────────────
def finalpreprocess(path):
    # ArcFace try karo pehle (face images ke liye better)
    try:
        arc = arcfacepreprocess(path)
        if arc is not None:
            print(f"ArcFace ✅ {os.path.basename(path)}")
            return arc
    except Exception as e:
        pass  # face nahi mila, CLIP pe fallback

    # CLIP fallback (non-face / general images)
    try:
        clips = prep(path)
        print(f"CLIP ✅ {os.path.basename(path)}")
        return clips
    except Exception as e:
        print(f"❌ Failed: {os.path.basename(path)} — {e}")
        return None

# ─── Search ───────────────────────────────────────────────────────────────────
def finalsearch(query_path, img_db):
    q_emb = finalpreprocess(query_path)
    if q_emb is None:
        print("Query image process nahi hui!")
        return []

    scores = []
    for path, emb in img_db:
        if emb is None:
            continue
        try:
            sim = float(np.dot(q_emb, emb))  # cosine sim (already normalized)
            scores.append((path, sim))
        except Exception:
            continue

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:10]