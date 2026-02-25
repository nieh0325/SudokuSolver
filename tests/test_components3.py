import cv2, os, numpy as np
n = '截圖 2026-02-24 09.37.16.png'
image = cv2.imread(os.path.join('題目', n))
h, w = image.shape[:2]
roi_top, roi_bot = int(h*0.75), int(h*0.95)
roi_bgr = image[roi_top:roi_bot, :]
roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
block_size = max(31, (h // 100) * 2 + 1)
if block_size % 2 == 0: block_size += 1

# Try both regular and inverted thresholding
th_inv = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 10)
th_reg = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 10)

for name, th in [('INV', th_inv), ('REG', th_reg)]:
    num, lab, stats, cents = cv2.connectedComponentsWithStats(th)
    count = 0
    for i in range(1, num):
        l, t, bw, bh, area = stats[i]
        if (h*0.02 < bh < h*0.09) and (w*0.01 < bw < w*0.07):
            ratio = bw / float(bh)
            if 0.2 < ratio < 0.95 and area > (bh * bw * 0.2) and area > 50:
                count += 1
    print(f'{n} [{name}]: matched components = {count}')
