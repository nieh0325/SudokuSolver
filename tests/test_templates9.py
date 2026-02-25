import cv2, os, numpy as np
from src.extractor import extract_templates_from_buttons

n = '截圖 2026-02-24 09.37.16.png'
image = cv2.imread(os.path.join('題目', n))
temps = extract_templates_from_buttons(image)
keys = sorted(temps.keys())
print(f'Found tempaltes: {keys}')
for k, v in temps.items():
    print(f'  {k}: shape={v.shape} unique={np.unique(v)}')
