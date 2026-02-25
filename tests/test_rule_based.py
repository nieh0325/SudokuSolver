import cv2, os, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, extract_templates_from_buttons

def match_digit_rule(dr, templates):
    best_score = float('-inf')
    best_digit = 0
    std_w, std_h = 24, 36
    
    _, dr_bin = cv2.threshold(dr, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    points = cv2.findNonZero(dr_bin)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        dr_bin = dr_bin[y:y+h, x:x+w]
        
    dr_resize = cv2.resize(dr_bin, (std_w, std_h))
    
    for d, t in templates.items():
        t_points = cv2.findNonZero(t)
        if t_points is not None:
            tx, ty, tw, th = cv2.boundingRect(t_points)
            t_cropped = t[ty:ty+th, tx:tx+tw]
        else:
            t_cropped = t
            
        t_resize = cv2.resize(t_cropped, (std_w, std_h))
        
        # Base score is just matchTemplate
        score = cv2.matchTemplate(dr_resize.astype(np.float32), t_resize.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0][0]
        
        # Structural difference
        diff = cv2.absdiff(dr_resize, t_resize)
        diff_score = np.sum(diff == 255) / float(std_w * std_h)
        score -= diff_score * 0.5
        
        # Rule based penalty for 2/3 and 3/8
        # 3 vs 8: 3 is open on the left, 8 is closed
        if d in [3, 8]:
            dr_left_sum = np.sum(dr_resize[:, :std_w//3] == 255)
            t_left_sum = np.sum(t_resize[:, :std_w//3] == 255)
            # If template is 3 (less pixels on left) and dr is 8 (more pixels on left), dr_left_sum is bigger
            if abs(dr_left_sum - t_left_sum) > (std_h * std_w//3) * 0.15:
                score -= 0.3
                
        # 2 vs 3: 2 has bottom-left pixels and top-right pixels, 3 has top-left open but bottom-left also open
        if d in [2, 3]:
            dr_bl = np.sum(dr_resize[std_h*2//3:, :std_w//2] == 255)
            t_bl = np.sum(t_resize[std_h*2//3:, :std_w//2] == 255)
            if abs(dr_bl - t_bl) > (std_h//3 * std_w//2) * 0.2:
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
