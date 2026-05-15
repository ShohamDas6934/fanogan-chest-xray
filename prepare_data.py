import pandas as pd
import shutil
import os
import random
from tqdm import tqdm

random.seed(42)

# ── paths ──────────────────────────────────────────────────────────────────
DATA_CSV   = r"D:\chestxray_data\Data_Entry_2017.csv"
IMG_DIRS   = [rf"D:\chestxray_data\images_{str(i).zfill(3)}\images"
              for i in range(1, 13)]

TRAIN_NORMAL  = r"D:\fanogan_project\data\train_normal"
TEST_NORMAL   = r"D:\fanogan_project\data\test_normal"
TEST_DISEASE  = r"D:\fanogan_project\data\test_disease"

# ── build a filename → full path lookup ────────────────────────────────────
print("Building image index...")
img_index = {}
for folder in IMG_DIRS:
    for fname in os.listdir(folder):
        img_index[fname] = os.path.join(folder, fname)
print(f"Total images found: {len(img_index)}")

# ── load labels ────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_CSV)

# ── split normals ──────────────────────────────────────────────────────────
normals = df[df["Finding Labels"] == "No Finding"]["Image Index"].tolist()
random.shuffle(normals)
normals = normals[:20000]          # cap at 20k for speed

split     = int(len(normals) * 0.85)
train_ids = normals[:split]        # ~17,000 for WGAN training
test_ids  = normals[split:]        # ~3,000 as true negatives

# ── pick disease images (true positives for evaluation) ───────────────────
disease_labels = ["Pneumonia", "Effusion", "Nodule", "Infiltration"]
disease_ids = []
for label in disease_labels:
    subset = df[df["Finding Labels"].str.contains(label)]["Image Index"].tolist()
    random.shuffle(subset)
    disease_ids += subset[:300]    # 300 per disease = 1200 total

# ── copy function ──────────────────────────────────────────────────────────
def copy_images(id_list, dest_folder, label):
    missing = 0
    for fname in tqdm(id_list, desc=f"Copying {label}"):
        if fname in img_index:
            shutil.copy2(img_index[fname], os.path.join(dest_folder, fname))
        else:
            missing += 1
    if missing:
        print(f"  {missing} files not found for {label}")

# ── run copies ─────────────────────────────────────────────────────────────
copy_images(train_ids,   TRAIN_NORMAL, "train_normal")
copy_images(test_ids,    TEST_NORMAL,  "test_normal")
copy_images(disease_ids, TEST_DISEASE, "test_disease")

print("\nDone!")
print(f"  train_normal : {len(os.listdir(TRAIN_NORMAL))} images")
print(f"  test_normal  : {len(os.listdir(TEST_NORMAL))}  images")
print(f"  test_disease : {len(os.listdir(TEST_DISEASE))} images")