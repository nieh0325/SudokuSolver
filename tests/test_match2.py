import cv2, os, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, extract_templates_from_buttons

def match_digit(roi_img, templates):
    best_score = float('-inf')
    best_digit = 0
    # Process roi_img
    _, dr = cv2.threshold(roi_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Crop to bounding box
    points = cv2.findNonZero(dr)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        dr = dr[y:y+h, x:x+w]
        
    for d, t in templates.items():
        th, tw = t.shape[:2]
        # Same aspect ratio resize
        t_aspect = tw / float(th)
        dr_aspect = dr.shape[1] / float(dr.shape[0])
        
        # Penalize different aspect ratio heavily
        aspect_diff = abs(t_aspect - dr_aspect)
        if aspect_diff > 0.35:
            continue
            
        r_dr = cv2.resize(dr, (tw, th))
        t_cropped = t
        t_points = cv2.findNonZero(t)
        if t_points is not None:
            tx, ty, tw, th = cv2.boundingRect(t_points)
            t_cropped = t[ty:ty+th, tx:tx+tw]
            t_aspect = tw / float(th)
            r_dr = cv2.resize(dr, (tw, th))
            
        # Compare via Structural Similarity or simple XOR
        intersection = np.logical_and(r_dr == 255, t_cropped == 255).sum()
        union = np.logical_or(r_dr == 255, t_cropped == 255).sum()
        iou = intersection / union if union > 0 else 0
        
        # Distance map
        dist_t = cv2.distanceTransform(cv2.bitwise_not(t_cropped), cv2.DIST_L2, 3)
        # Average distance of the pixels in dr from t
        score_dt = np.mean(dist_t[r_dr == 255]) if np.any(r_dr == 255) else 100
        
        # matchTemplate score for shape
        score_m = cv2.matchTemplate(r_dr.astype(np.float32), t_cropped.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0][0]
        
        # We want high IOU, low DT, high matchTemplate
        score = iou * 1.5 - (score_dt / 5.0) + score_m * 1.5
        score -= aspect_diff * 1.5
        
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
                    d, s = match_digit(dr, templates)
                    print(f"  ({y},{x}) -> dr=({dr.shape[1]},{dr.shape[0]}) best=({d}, {s:.3f})")

