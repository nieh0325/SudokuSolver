import cv2, os, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, extract_templates_from_buttons, train_knn

for n in ['截圖 2026-02-24 10.43.16.png', '截圖 2026-02-24 11.21.56.png']:
    image = cv2.imread(os.path.join('題目', n))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hl, vl = detect_all_lines(gray)
    templates = extract_templates_from_buttons(image)
    knn_model = train_knn(templates)
    
    print(n)
    for y in range(9):
        for x in range(9):
            if ((n == '截圖 2026-02-24 10.43.16.png' and y==0 and x==2) or (n == '截圖 2026-02-24 11.21.56.png' and y==0 and x==6) or (n == '截圖 2026-02-24 10.43.16.png' and y==1 and x==8)):
                sy, ey, sx, ex = hl[y], hl[y+1], vl[x], vl[x+1]
                my, mx = int((ey-sy)*0.13), int((ex-sx)*0.13)
                cell_img = gray[sy+my:ey-my, sx+mx:ex-mx]
                _, dr = get_clean_digit(cell_img)
                if dr is not None:
                    std_w, std_h = 24, 36
                    _, dr_bin = cv2.threshold(dr, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    points = cv2.findNonZero(dr_bin)
                    if points is not None:
                        x_p, y_p, w_p, h_p = cv2.boundingRect(points)
                        dr_clean = dr_bin[y_p:y_p+h_p, x_p:x_p+w_p]
                    else:
                        dr_clean = dr_bin
                        
                    knn_pred = None
                    if knn_model is not None and dr_clean.size > 0:
                        r_dr_knn = cv2.resize(dr_clean, (std_w, std_h))
                        feat = r_dr_knn.flatten().astype(np.float32).reshape(1, -1)
                        ret, results, neighbours, dist = knn_model.findNearest(feat, k=3)
                        # Predict based on majority or just closest
                        knn_pred = int(results[0][0])
                        print(f"  [{y},{x}] KNN pred: {knn_pred}, neigh: {neighbours[0]}")
                        
                    scores = {}
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
                        
                        resized_dr = cv2.resize(dr_clean, (tw, th))
                        score = cv2.matchTemplate(resized_dr.astype(np.float32), t_cropped.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0][0]
                        
                        intersection = np.logical_and(resized_dr == 255, t_cropped == 255).sum()
                        union = np.logical_or(resized_dr == 255, t_cropped == 255).sum()
                        score_iou = intersection / union if union > 0 else 0
                        
                        score = (score * 0.5) + (score_iou * 0.5)
                        
                        if aspect_diff > 0.15: score -= 0.15
                        
                        if d in [8, 3]:
                            num_holes_dr, _, _, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(resized_dr))
                            num_holes_t, _, _, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(t_cropped))
                            if num_holes_dr != num_holes_t: score -= 0.2
                        
                        if d in [2, 3]:
                            left_dr = np.sum(resized_dr[:, :tw//3] == 255)
                            left_t = np.sum(t_cropped[:, :tw//3] == 255)
                            diff_left = abs(left_dr - left_t) / float(tw//3 * th)
                            if diff_left > 0.1: score -= 0.2
                            
                        # If KNN strongly suggests a digit, give it a big boost
                        if knn_pred == d:
                            score += 0.3
                            
                        scores[d] = score
                        
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    print(f"  ({y},{x}) -> dr=({dr.shape[1]},{dr.shape[0]}) best=({sorted_scores[0][0]}, score={sorted_scores[0][1]:.3f})")
