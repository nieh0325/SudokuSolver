import cv2, os, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, extract_templates_from_buttons

def match_digit_advanced(dr, templates):
    best_score = float('-inf')
    best_digit = 0
    
    # 確保 dr 也是 binary
    _, dr_bin = cv2.threshold(dr, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 取 dr 的輪廓
    dr_cnts, _ = cv2.findContours(dr_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not dr_cnts: return 0, 0
    dr_cnt = max(dr_cnts, key=cv2.contourArea)

    for d, t in templates.items():
        th, tw = t.shape[:2]
        
        # 1. Aspect ratio penalty
        t_aspect = tw / float(th)
        dr_aspect = dr.shape[1] / float(dr.shape[0])
        aspect_diff = abs(t_aspect - dr_aspect)
        if aspect_diff > 0.35:
            continue
            
        r_dr = cv2.resize(dr_bin, (tw, th), interpolation=cv2.INTER_NEAREST)
        
        # 2. Template matching score
        score_match = cv2.matchTemplate(cv2.resize(dr, (tw, th)), t, cv2.TM_CCOEFF_NORMED)[0][0]
        
        # 3. Shape matching score (lower is better, 0 is perfect match)
        # Template contours
        t_cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not t_cnts: continue
        t_cnt = max(t_cnts, key=cv2.contourArea)
        
        shape_dist = cv2.matchShapes(dr_cnt, t_cnt, cv2.CONTOURS_MATCH_I2, 0)
        # Convert shape distance to a score 0-1 (1 is perfect)
        score_shape = max(0, 1.0 - shape_dist * 2.5)
        
        # 4. Combine score
        # Give matchTemplate more weight but penalize heavily if shape doesn't match
        score = (score_match * 0.6) + (score_shape * 0.4)
        if aspect_diff > 0.15: score -= 0.1
        
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
                    d, s = match_digit_advanced(dr, templates)
                    print(f"  ({y},{x}) -> dr=({dr.shape[1]},{dr.shape[0]}) best=({d},{s:.3f})")
