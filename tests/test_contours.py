import cv2, os, numpy as np

for n in ['截圖 2026-02-24 09.37.16.png', '截圖 2026-02-24 11.07.34.png', '截圖 2026-02-24 11.21.56.png']:
    image = cv2.imread(os.path.join('題目', n))
    h, w = image.shape[:2]
    roi_top, roi_bot = int(h*0.75), int(h*0.95)
    roi_bgr = image[roi_top:roi_bot, :]
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    block_size = max(31, (h // 100) * 2 + 1)
    if block_size % 2 == 0: block_size += 1

    th_inv = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 10)
    th_norm = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 10)

    res = []
    # Try finding contours on both mappings
    for th in [th_inv, th_norm]:
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if (h*0.015 < bh < h*0.09) and (w*0.01 < bw < w*0.07):
                ratio = bw / float(bh)
                # Area of contour vs bounding box
                if 0.15 < ratio < 0.95 and area > 20: 
                    y_center = y + bh/2
                    roi_h = roi_bot - roi_top
                    if roi_h * 0.1 < y_center < roi_h * 0.9:
                        res.append((x, y, bw, bh, area, th))

    print(f'{n}: Contours {len(res)}')
    slot_w = w / 9.0
    for slot in range(9):
        slot_l, slot_r = slot*slot_w, (slot+1)*slot_w
        best = None
        for c in res:
            cx = c[0] + c[2]/2
            if slot_l < cx < slot_r:
                score = c[3] * c[2] # bh * bw
                if best is None or score > best[3]*best[2]:
                    best = c
        if best:
            print(f'  Slot {slot+1}: w={best[2]} h={best[3]} area={best[4]}')
        else:
            print(f'  Slot {slot+1}: MISSING')
