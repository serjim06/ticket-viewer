from datasets import load_dataset
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

dataset = load_dataset("jsdnrs/ICDAR2019-SROIE")

output_dir = "./dataset_crops"
os.makedirs(output_dir, exist_ok=True)

data_records = []
crop_count = 0

train_data = dataset["train"]


for item in tqdm(train_data):
    pil_image = item["image"]
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    words = item.get("words", [])
    boxes = item.get("bboxes", [])
    
    for box, text in zip(boxes, words):
        text_str = str(text).strip()
        if not text_str:
            continue
        
        try:
            x_min, y_min, x_max, y_max = map(int, box[:4])
            
            crop = image[y_min:y_max, x_min:x_max]
            
            if crop.shape[0] < 5 or crop.shape[1] < 5:
                continue
            
            crop_filename = f"crop_{crop_count:06d}.png"
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, crop)
            
            data_records.append({
                "image_path": crop_path,
                "label": text_str
            })
            
            crop_count += 1
            
        except Exception:
            print("ERROR")
            continue
        
df = pd.DataFrame(data_records)
df.to_csv("labels.csv", index=False)

print("Finished!")

print(df.head())

