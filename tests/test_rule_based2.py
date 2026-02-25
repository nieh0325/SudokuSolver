import cv2, os, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, extract_templates_from_buttons

def match_digit_rule(dr, templates):
    best_score = float('-inf')
    best_digit = 0
    
    # Binarize
    _, dr_bin = cv2.threshold(dr, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    points = cv2.findNonZero(dr_bin)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        dr_bin = dr_bin[y:y+h, x:x+w]
        
    for d, t in templates.items():
        t_points = cv2.findNonZero(t)
        if t_points is not None:
            tx, ty, tw, th = cv2.boundingRect(t_points)
            t_cropped = t[ty:ty+th, tx:tx+tw]
        else:
            t_cropped = t
            th, tw = t.shape[:2]
            
        dr_aspect = dr_bin.shape[1] / float(dr_bin.shape[0])
        t_aspect = tw / float(th)
        aspect_diff = abs(dr_aspect - t_aspect)
        if aspect_diff > 0.35:
            continue
            
        r_dr = cv2.resize(dr_bin, (tw, th))
        # Base Match Template Score
        score = cv2.matchTemplate(r_dr.astype(np.float32), t_cropped.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0][0]
        
        # Add aspect penalty
        if aspect_diff > 0.15: score -= 0.15
        
        # Intersection over Union
        inter = np.logical_and(r_dr == 255, t_cropped == 255).sum()
        union = np.logical_or(r_dr == 255, t_cropped == 255).sum()
        score_iou = inter / union if union > 0 else 0
        score = score * 0.6 + score_iou * 0.4
        
        if d in [2, 3]:
            # For 2 vs 3, the bottom left is the key difference (2 has pixels, 3 doesn't)
            # Define bottom left quadrant
            h_half, w_half = th // 2, tw // 2
            dr_bl = np.sum(r_dr[h_half:, :w_half] == 255) / float(h_half * w_half)
            t_bl = np.sum(t_cropped[h_half:, :w_half] == 255) / float(h_half * w_half)
            if abs(dr_bl - t_bl) > 0.15:
                score -= 0.2
                
        if d in [8, 3]:
            # For 8 vs 3, the top left and bottom left are key differences (8 has pixels, 3 doesn't)
            dr_l = np.sum(r_dr[:, :tw//3] == 255) / float(th * (tw//3))
            t_l = np.sum(t_cropped[:, :tw//3] == 255) / float(th * (tw//3))
            if abs(dr_l - t_l) > 0.15:
                score -= 0.2

        if score > best_score:
            best_score = score
            best_digit = d
            
    return best_digit, best_score

for n in ['截圖 2026-02-24 10.43.16.png', '截圖 2026-02-24 11.21.56.png']:
    image = cv2.imread(os.path.join('題目', n))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hl, vl = detect_all_lines(gray)
    templates = extract_templates_from_buttons(image)
    
    print(n)
    for y in range(9):
        for x in range(9):
            if ((n == '截圖 2026-02-24 10.43.16.png' and y==0 and x==2) or (n == '截圖 2026-02-24 11.21.56.png' and y==0 and x==6) or (n == '截圖 2026-02-24 10.43.16.png' and y==1 and x==8)):
                sy, ey, sx, ex = hl[y], hl[y+1], vl[x], vl[x+1]
                my, mx = int((ey-sy)*0.13), int((ex-sx)*0.13)
                cell_img = gray[sy+my:ey-my, sx+mx:ex-mx]
                _, dr = get_clean_digit(cell_img)
                if dr is not None:
                    d, s = match_digit_rule(dr, templates)
                    print(f"  ({y},{x}) -> dr=({dr.shape[1]},{dr.shape[0]}) best=({d}, score={s:.3f})")
