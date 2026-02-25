import cv2, os, numpy as np
n = '截圖 2026-02-24 09.37.16.png'
image = cv2.imread(os.path.join('題目', n))
h, w = image.shape[:2]
roi_top, roi_bot = int(h*0.75), int(h*0.95)
roi_bgr = image[roi_top:roi_bot, :]
roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
block_size = max(31, (h // 100) * 2 + 1)
if block_size % 2 == 0: block_size += 1

th = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 10)
num, lab, stats, cents = cv2.connectedComponentsWithStats(th)

res = []
for i in range(1, num):
    l, t, bw, bh, area = stats[i]
    if (h*0.02 < bh < h*0.09) and (w*0.01 < bw < w*0.07):
        ratio = bw / float(bh)
        if 0.2 < ratio < 0.95 and area > (bh * bw * 0.2) and area > 50:
            res.append((l, t, bw, bh, i, lab))

print(f'Total candidates: {len(res)}')
slot_w = w / 9.0
for slot in range(9):
    slot_l, slot_r = slot*slot_w, (slot+1)*slot_w
    best = None
    matched = []
    for c in res:
        cx = c[0] + c[2]/2
        if slot_l < cx < slot_r:
            matched.append(c)
    print(f'Slot {slot+1} (x: {slot_l:.1f}-{slot_r:.1f}): found {len(matched)} candidates')
    for m in matched:
        print(f'  -> cx={m[0] + m[2]/2}')
