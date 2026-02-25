import cv2, os, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, extract_templates_from_buttons

def get_h_v_profiles(img_bin):
    h_prof = np.sum(img_bin, axis=1) / 255.0
    v_prof = np.sum(img_bin, axis=0) / 255.0
    return h_prof, v_prof

def match_digit_advanced(dr, templates):
    best_score = float('inf')
    best_digit = 0
    std_w, std_h = 30, 45
    
    _, dr_bin = cv2.threshold(dr, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    points = cv2.findNonZero(dr_bin)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        dr_bin = dr_bin[y:y+h, x:x+w]
        
    r_dr = cv2.resize(dr_bin, (std_w, std_h))
    h_dr, v_dr = get_h_v_profiles(r_dr)
    
    # Check "holes" using connected components on inverse
    num_holes_dr, _, _, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(r_dr))
    num_holes_dr -= 1 # minus background
    
    for d, t in templates.items():
        t_points = cv2.findNonZero(t)
        if t_points is not None:
            tx, ty, tw, th = cv2.boundingRect(t_points)
            t_cropped = t[ty:ty+th, tx:tx+tw]
        else:
            t_cropped = t
            
        r_t = cv2.resize(t_cropped, (std_w, std_h))
        num_holes_t, _, _, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(r_t))
        num_holes_t -= 1
        
        h_t, v_t = get_h_v_profiles(r_t)
        
        # Absolute diff for pixels
        dist_pixel = np.sum(np.abs(r_dr.astype(int) - r_t.astype(int))) / 255.0 / (std_w * std_h)
        
        # Profile difference
        h_diff = np.sum(np.abs(h_dr - h_t)) / float(std_h * std_w)
        v_diff = np.sum(np.abs(v_dr - v_t)) / float(std_h * std_w)
        
        score = dist_pixel * 100.0 + (h_diff + v_diff) * 50.0
        
        # Hole penalty (8 has 2 holes, 3 has 0 holes theoretically inside the bounding box depending on font, but background connects them)
        if num_holes_dr != num_holes_t:
            score += 20.0
        
        # Left-profile difference (very important for 3 vs 8, 2 vs 3)
        # Find first non-zero pixel from left for each row
        dr_left = np.argmax(r_dr > 128, axis=1)
        t_left = np.argmax(r_t > 128, axis=1)
        left_prof_diff = np.sum(np.abs(dr_left - t_left)) / float(std_h * std_w)
        
        score += left_prof_diff * 100.0
        
        if d in [2, 3]:
            # print(f"    cand {d} -> score {score:.2f} (pix:{dist_pixel*100:.1f}, prof:{(h_diff+v_diff)*50:.1f}, holes:{num_holes_dr}/{num_holes_t}, left:{left_prof_diff*100:.1f})")
            pass
            
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
                    d, s = match_digit_advanced(dr, templates)
                    print(f"  ({y},{x}) -> dr=({dr.shape[1]},{dr.shape[0]}) best=({d}, score={s:.3f})")
