import cv2, os, numpy as np
n = '截圖 2026-02-24 09.37.16.png'
image = cv2.imread(os.path.join('题目', n))
if image is None: image = cv2.imread(os.path.join('題目', n))
h, w = image.shape[:2]
roi_top, roi_bot = int(h*0.75), int(h*0.95)
roi_bgr = image[roi_top:roi_bot, :]
roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

th = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
num, lab, stats, cents = cv2.connectedComponentsWithStats(th)
print(f'Total components: {num}')
count = 0
for i in range(1, num):
    l, t, bw, bh, area = stats[i]
    if area > 10:
        print(f'Component {i}: w={bw}, h={bh}, area={area}, expected h range: {h*0.03:.1f}-{h*0.07:.1f}, expected w range: {w*0.01:.1f}-{w*0.08:.1f}')
