import cv2, os, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, extract_templates_from_buttons

for n in ['截圖 2026-02-24 11.21.56.png', '截圖 2026-02-24 10.43.16.png']:
    image = cv2.imread(os.path.join('題目', n))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hl, vl = detect_all_lines(gray)
    templates = extract_templates_from_buttons(image)
    
    print(n)
    for y in range(9):
        for x in range(9):
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
                    
                std_w, std_h = min(24, dr_clean.shape[1]), min(36, dr_clean.shape[0])
                if std_w == 0 or std_h == 0: continue
                r_dr = cv2.resize(dr_clean, (std_w, std_h))
                
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
                    
                    r_t = cv2.resize(t_cropped, (std_w, std_h))
                    
                    score_m = cv2.matchTemplate(r_dr.astype(np.float32), r_t.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0][0]
                    intersection = np.logical_and(r_dr == 255, r_t == 255).sum()
                    union = np.logical_or(r_dr == 255, r_t == 255).sum()
                    score_iou = intersection / union if union > 0 else 0
                    score = (score_m * 0.6) + (score_iou * 0.4)
                    
                    # Apply custom penalties
                    if aspect_diff > 0.15: score -= 0.15
                    
                    dr_solid = np.sum(dr_clean == 255) / float(dr_clean.size)
                    t_solid = np.sum(t_cropped == 255) / float(t_cropped.size)
                    solid_diff = abs(dr_solid - t_solid)
                    if solid_diff > 0.15: score -= 0.15
                    
                    # Left heavy? Right heavy? Top? Bottom?
                    # This helps with 2 vs 3, 3 vs 8
                    dr_zones = [
                        np.sum(r_dr[:std_h//2, :std_w//2] == 255),
                        np.sum(r_dr[:std_h//2, std_w//2:] == 255),
                        np.sum(r_dr[std_h//2:, :std_w//2] == 255),
                        np.sum(r_dr[std_h//2:, std_w//2:] == 255)
                    ]
                    t_zones = [
                        np.sum(r_t[:std_h//2, :std_w//2] == 255),
                        np.sum(r_t[:std_h//2, std_w//2:] == 255),
                        np.sum(r_t[std_h//2:, :std_w//2] == 255),
                        np.sum(r_t[std_h//2:, std_w//2:] == 255)
                    ]
                    
                    zone_diff = sum([abs(dz - tz) for dz, tz in zip(dr_zones, t_zones)]) / float(std_w * std_h)
                    
                    # Apply zone penalty
                    if zone_diff > 0.15: score -= zone_diff * 0.8
                    
                    scores[d] = score
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                if n == '截圖 2026-02-24 11.21.56.png' and y in [3, 4, 5, 7]:
                    print(f"  ({y},{x}) -> best=({sorted_scores[0][0]}, {sorted_scores[0][1]:.3f}) second=({sorted_scores[1][0] if len(sorted_scores)>1 else 'N/A'}, {sorted_scores[1][1] if len(sorted_scores)>1 else 0:.3f})")
