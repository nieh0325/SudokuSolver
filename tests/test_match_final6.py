import cv2, os, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, extract_templates_from_buttons, train_knn

for n in ['截圖 2026-02-24 11.21.56.png', '截圖 2026-02-24 10.43.16.png']:
    image = cv2.imread(os.path.join('題目', n))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hl, vl = detect_all_lines(gray)
    templates = extract_templates_from_buttons(image)
    knn_model = train_knn(templates)
    
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
                std_w, std_h = min(24, dr.shape[1]), min(36, dr.shape[0])
                if std_w == 0 or std_h == 0: continue
                
                _, dr_bin = cv2.threshold(dr, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                points = cv2.findNonZero(dr_bin)
                if points is not None:
                    x_p, y_p, w_p, h_p = cv2.boundingRect(points)
                    dr_clean = dr_bin[y_p:y_p+h_p, x_p:x_p+w_p]
                else:
                    dr_clean = dr_bin
                    
                knn_pred = None
                if knn_model is not None and dr_clean.size > 0:
                    r_dr_knn = cv2.resize(dr_clean, (24, 36))
                    feat = r_dr_knn.flatten().astype(np.float32).reshape(1, -1)
                    ret, results, neighbours, dist = knn_model.findNearest(feat, k=1)
                    knn_pred = int(results[0][0])
                
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
                    
                    if zone_diff > 0.15: score -= zone_diff * 0.8
                    
                    scores[d] = score
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                best_d, best_s = sorted_scores[0]
                
                # KNN Boost Logic
                if knn_pred is not None:
                    # If highest score is low, trust KNN directly
                    if best_s < 0.4:
                        best_d = knn_pred
                        best_s = 0.5 # Give it just enough to pass threshold (0.45 usually)
                    # If scores are close, use KNN as tie breaker
                    elif best_s < 0.75 and len(sorted_scores) > 1:
                        second_best_d, second_best_s = sorted_scores[1]
                        if (best_s - second_best_s) < 0.15:
                            if knn_pred == best_d:
                                best_s += 0.2
                            elif knn_pred == second_best_d:
                                best_d = second_best_d
                                best_s = second_best_s + 0.2
                                
                if n == '截圖 2026-02-24 11.21.56.png' and y in [4, 5, 7]:
                    print(f"  ({y},{x}) -> dr=({dr.shape[1]},{dr.shape[0]}) best=({best_d}, {best_s:.3f})")
                elif best_s < 0.4:
                    print(f"  ({y},{x}) -> FAIL best=({best_d}, {best_s:.3f})")
