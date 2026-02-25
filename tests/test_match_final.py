import cv2, os, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, extract_templates_from_buttons

for n in ['截圖 2026-02-24 10.43.16.png', '截圖 2026-02-24 11.21.56.png', '截圖 2026-02-24 11.07.34.png', '截圖 2026-02-24 09.37.16.png']:
    image = cv2.imread(os.path.join('題目', n))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hl, vl = detect_all_lines(gray)
    templates = extract_templates_from_buttons(image)
    
    print(n)
    for y in range(9):
        for x in range(9):
            if ((n == '截圖 2026-02-24 10.43.16.png' and y==0 and x==2) or (n == '截圖 2026-02-24 11.21.56.png' and y==0 and x==6) or (n == '截圖 2026-02-24 10.43.16.png' and y==1 and x==8) or (n == '截圖 2026-02-24 11.07.34.png' and y==1 and x==6)):
                sy, ey, sx, ex = hl[y], hl[y+1], vl[x], vl[x+1]
                my, mx = int((ey-sy)*0.13), int((ex-sx)*0.13)
                cell_img = gray[sy+my:ey-my, sx+mx:ex-mx]
                _, dr = get_clean_digit(cell_img)
                if dr is not None:
                    # Let's run the exact same block
                    scores = {}
                    _, dr_bin = cv2.threshold(dr, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    points = cv2.findNonZero(dr_bin)
                    if points is not None:
                        x_p, y_p, w_p, h_p = cv2.boundingRect(points)
                        dr_clean = dr_bin[y_p:y_p+h_p, x_p:x_p+w_p]
                    else:
                        dr_clean = dr_bin
                    for d, t in templates.items():
                        t_points = cv2.findNonZero(t)
                        if t_points is not None:
                            tx, ty, tw, th = cv2.boundingRect(t_points)
                            t_cropped = t[ty:ty+th, tx:tx+tw]
                        else:
                            t_cropped = t
                            th, tw = t.shape[:2]
                        dr_aspect = dr_clean.shape[1] / float(dr_clean.shape[0])
                        t_aspect = tw / float(th)
                        aspect_diff = abs(dr_aspect - t_aspect)
                        if aspect_diff > 0.35: continue
                        
                        resized_dr = cv2.resize(dr_clean, (tw, th))
                        score_m = cv2.matchTemplate(resized_dr.astype(np.float32), t_cropped.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0][0]
                        intersection = np.logical_and(resized_dr == 255, t_cropped == 255).sum()
                        union = np.logical_or(resized_dr == 255, t_cropped == 255).sum()
                        score_iou = intersection / union if union > 0 else 0
                        score = (score_m * 0.5) + (score_iou * 0.5)
                        
                        # Apply custom penalties
                        if aspect_diff > 0.15: score -= 0.1
                        
                        dr_solid = np.sum(dr_clean == 255) / float(dr_clean.size)
                        t_solid = np.sum(t_cropped == 255) / float(t_cropped.size)
                        solid_diff = abs(dr_solid - t_solid)
                        if solid_diff > 0.15: score -= 0.1
                        
                        scores[d] = score
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    print(f"  ({y},{x}) -> dr=({dr.shape[1]},{dr.shape[0]}) best=({sorted_scores[0][0]}, score={sorted_scores[0][1]:.3f})")
