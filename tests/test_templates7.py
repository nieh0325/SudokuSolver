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

def get_digits(timg):
    cnts, _ = cv2.findContours(timg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    for i, c in enumerate(cnts):
        l, t, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if (h*0.015 < bh < h*0.09) and (w*0.01 < bw < w*0.07):
            ratio = bw / float(bh)
            if 0.15 < ratio < 0.95 and area > 20: 
                y_center = t + bh/2
                roi_h = roi_bot - roi_top
                if roi_h * 0.1 < y_center < roi_h * 0.9:
                    res.append((l, t, bw, bh, i, timg, area))
    return res

all_cands = get_digits(th_inv) + get_digits(th_norm)
print(f"Total cands: {len(all_cands)}")
for c in all_cands:
    print(f"Cand: x={c[0]}, y={c[1]}, w={c[2]}, h={c[3]}, area={c[6]}")

slot_w = w / 9.0
for slot in range(9):
    slot_l, slot_r = slot*slot_w, (slot+1)*slot_w
    best = None
    for c in all_cands:
        cx = c[0] + c[2]/2
        if slot_l < cx < slot_r:
            score = c[3] * c[2]
            if best is None or score > best[3]*best[2]: best = c
    if best:
        print(f"Slot {slot+1}: FOUND w={best[2]} h={best[3]}")
    else:
        print(f"Slot {slot+1}: MISSING")
