import cv2, os, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, extract_templates_from_buttons

def get_h_v_profiles(img_bin):
    h_prof = np.sum(img_bin, axis=1) / 255.0
    v_prof = np.sum(img_bin, axis=0) / 255.0
    return h_prof, v_prof

def match_digit(dr, templates):
    best_score = float('inf')
    best_digit = 0
    
    _, dr_bin = cv2.threshold(dr, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    points = cv2.findNonZero(dr_bin)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        dr_bin = dr_bin[y:y+h, x:x+w]
        
    # Resize to standard size (20x30)
    std_w, std_h = 20, 30
    r_dr = cv2.resize(dr_bin, (std_w, std_h))
    h_dr, v_dr = get_h_v_profiles(r_dr)
    
    for d, t in templates.items():
        t_points = cv2.findNonZero(t)
        if t_points is not None:
            tx, ty, tw, th = cv2.boundingRect(t_points)
            t_cropped = t[ty:ty+th, tx:tx+tw]
        else:
            t_cropped = t
            
        r_t = cv2.resize(t_cropped, (std_w, std_h))
        h_t, v_t = get_h_v_profiles(r_t)
        
        # Profile difference
        h_diff = np.sum(np.abs(h_dr - h_t))
        v_diff = np.sum(np.abs(v_dr - v_t))
        
        # Pixel difference (XOR)
        xor_diff = np.logical_xor(r_dr > 128, r_t > 128).sum()
        
        score = h_diff + v_diff + xor_diff
        
        # print(f"    Cand {d} Dist: {score:.2f} (H:{h_diff:.1f}, V:{v_diff:.1f}, XOR:{xor_diff})")
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
                    print(f"  ({y},{x}) -> dr=({dr.shape[1]},{dr.shape[0]}) best=({d}, dist={s:.1f})")

