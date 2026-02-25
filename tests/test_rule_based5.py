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
        if d == 1 and aspect_diff > 0.2: score -= 0.3
        
        # Intersection over Union
        inter = np.logical_and(r_dr == 255, t_cropped == 255).sum()
        union = np.logical_or(r_dr == 255, t_cropped == 255).sum()
        score_iou = inter / union if union > 0 else 0
        score = score * 0.6 + score_iou * 0.4
        
        # Structural 2 vs 3 (2 needs bottom left, 3 is empty)
        dr_bl = np.sum(r_dr[-th//3:, :tw//2] == 255) / float(th//3 * tw//2)
        t_bl = np.sum(t_cropped[-th//3:, :tw//2] == 255) / float(th//3 * tw//2)
        
        # Structural 8 vs 3 (8 needs top left and bottom left, 3 is mostly empty)
        dr_l = np.sum(r_dr[:, :tw//3] == 255) / float(th * (tw//3))
        t_l = np.sum(t_cropped[:, :tw//3] == 255) / float(th * (tw//3))
        
        # Custom rules
        if d == 3:
            # If digit is actually a 2, dr_bl will be much higher than t_bl (which is low for 3)
            if dr_bl - t_bl > 0.15: score -= 0.3
            # If digit is actually an 8, dr_l will be much higher than t_l
            if dr_l - t_l > 0.2: score -= 0.3
        
        if d == 2:
            # If digit is actually a 3, dr_bl will be lower than t_bl (which is high for 2)
            if t_bl - dr_bl > 0.15: score -= 0.3
            
        if d == 8:
            # If digit is actually a 3, dr_l will be lower than t_l
            if t_l - dr_l > 0.2: score -= 0.3

        if score > best_score:
            best_score = score
            best_digit = d
            
    return best_digit, best_score

for n in ['截圖 2026-02-24 10.43.16.png', '截圖 2026-02-24 11.21.56.png', '截圖 2026-02-24 11.07.34.png']:
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
                    d, s = match_digit_rule(dr, templates)
                    print(f"  ({y},{x}) -> dr=({dr.shape[1]},{dr.shape[0]}) best=({d}, score={s:.3f})")
