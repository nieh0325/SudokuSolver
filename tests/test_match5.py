import cv2, os, math, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, extract_templates_from_buttons

def get_hu_features(img):
    m = cv2.moments(img)
    hu = cv2.HuMoments(m)
    features = []
    for i in range(7):
        if hu[i][0] != 0:
            features.append(-1 * math.copysign(1.0, hu[i][0]) * math.log10(abs(hu[i][0])))
        else:
            features.append(0)
    return np.array(features)

def match_digit(dr, templates):
    best_score = float('inf')
    best_digit = 0
    std_w, std_h = 30, 45
    
    _, dr_bin = cv2.threshold(dr, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    points = cv2.findNonZero(dr_bin)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        dr_bin = dr_bin[y:y+h, x:x+w]
        
    r_dr = cv2.resize(dr_bin, (std_w, std_h))
    feat_dr = get_hu_features(r_dr)
    
    for d, t in templates.items():
        t_points = cv2.findNonZero(t)
        if t_points is not None:
            tx, ty, tw, th = cv2.boundingRect(t_points)
            t_cropped = t[ty:ty+th, tx:tx+tw]
        else:
            t_cropped = t
            
        r_t = cv2.resize(t_cropped, (std_w, std_h))
        feat_t = get_hu_features(r_t)
        
        # L2 norm for Hu moments (first 3 moments are most stable)
        dist_hu = np.linalg.norm(feat_dr[:3] - feat_t[:3])
        
        # Absolute diff for pixels
        dist_pixel = np.sum(np.abs(r_dr.astype(int) - r_t.astype(int))) / 255.0 / (std_w * std_h)
        
        score = dist_hu * 10.0 + dist_pixel * 100.0
        
        if score < best_score:
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
                    print(f"  ({y},{x}) -> dr=({dr.shape[1]},{dr.shape[0]}) best=({d}, score={s:.3f})")

