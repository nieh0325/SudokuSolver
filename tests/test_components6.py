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
    print(n)
    for idx, th in enumerate([th_inv, th_norm]):
        name = 'INV' if idx == 0 else 'NORM'
        num, lab, stats, cents = cv2.connectedComponentsWithStats(th)
        for i in range(1, num):
            l, t, bw, bh, area = stats[i]
            if (h*0.02 < bh < h*0.09) and (w*0.01 < bw < w*0.07):
                ratio = bw / float(bh)
                # 修改 ratio 條件: 數字 1 的 ratio 可以到 0.2，數字 4 或 0 約 0.8
                if 0.15 < ratio < 0.95 and area > (bh * bw * 0.2) and area > 50:
                    y_center = t + bh/2
                    # 避免抓到最下方邊界以外的東西 (預設按鈕通常在中間)
                    roi_h = roi_bot - roi_top
                    if y_center < roi_h * 0.9:
                        roi = th[t:t+bh, l:l+bw]
                        # check solidness
                        solidness = np.sum(roi == 255) / (bw * bh * 255.0)
                        if solidness > 0.15:
                            res.append((l, t, bw, bh, i, lab, solidness, name))

    slot_w = w / 9.0
    for slot in range(9):
        slot_l, slot_r = slot*slot_w, (slot+1)*slot_w
        best = None
        for c in res:
            cx = c[0] + c[2]/2
            if slot_l < cx < slot_r:
                score = c[3] * c[2] * c[6] # size * solidness
                if best is None or score > best[3]*best[2]*best[6]:
                    best = c
        if best:
            print(f'  Slot {slot+1}: w={best[2]} h={best[3]} type={best[7]}')
        else:
            print(f'  Slot {slot+1}: MISSING')
