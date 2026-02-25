import cv2, os, numpy as np

n = '截圖 2026-02-24 09.37.16.png'
image = cv2.imread(os.path.join('題目', n))
h, w = image.shape[:2]
roi_top, roi_bot = int(h*0.75), int(h*0.95)
roi_bgr = image[roi_top:roi_bot, :]
roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
block_size = max(31, (h // 100) * 2 + 1)
if block_size % 2 == 0: block_size += 1

th_inv = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 10)
th_norm = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 10)

for th in [th_inv, th_norm]:
    num, lab, stats, cents = cv2.connectedComponentsWithStats(th)
    for i in range(1, num):
        l, t, bw, bh, area = stats[i]
        
        # Height and width limit: 0.02*h ~ 0.09*h -> for 2200h, 44~198. (My bad: 09.37.16 button h is 54).
        # And width: 0.01*w ~ 0.07*w -> for 1170w, 11~81.
        if (h*0.015 < bh < h*0.09) and (w*0.01 < bw < w*0.07):
            ratio = bw / float(bh)
            if 0.15 < ratio < 0.95 and area > 50:
                y_center = t + bh/2
                roi_h = roi_bot - roi_top
                if roi_h * 0.1 < y_center < roi_h * 0.9:
                    # Let's verify what values we have here
                    print(f'Match: w={bw}, h={bh}, ratio={ratio:.2f}, area={area}, y={y_center}')

