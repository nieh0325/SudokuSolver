import cv2, os, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, extract_templates_from_buttons, ocr_digit_fallback

for n in ['截圖 2026-02-24 10.43.16.png', '截圖 2026-02-24 11.21.56.png']:
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
            res, dr = get_clean_digit(cell_img)
            if dr is not None:
                scores = {}
                # Convert dr to binary to match templates (which are 0/255)
                _, dr_bin = cv2.threshold(dr, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                
                for d, t in templates.items():
                    th, tw = t.shape[:2]
                    resized_dr = cv2.resize(dr_bin, (tw, th), interpolation=cv2.INTER_NEAREST)
                    
                    # Score 1: matchTemplate
                    score_match = cv2.matchTemplate(cv2.resize(dr, (tw, th)), t, cv2.TM_CCOEFF_NORMED)[0][0]
                    
                    # Score 2: IOU (Intersection over Union) of active pixels
                    intersection = np.logical_and(resized_dr == 255, t == 255).sum()
                    union = np.logical_or(resized_dr == 255, t == 255).sum()
                    score_iou = intersection / union if union > 0 else 0
                    
                    # Combine
                    score = (score_match + score_iou * 1.5) / 2.5
                    
                    ratio_diff = abs((res[2]/float(res[3])) - (tw/float(th)))
                    if ratio_diff > 0.2: score -= 0.15
                    scores[d] = score
                    
                if scores:
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    best_d, best_s = sorted_scores[0]
                    # We are tracking known failures: (10.43.16 -> 2 is 0,2; 11.21.56 -> 3 is 0,6)
                    target = 2 if n == '截圖 2026-02-24 10.43.16.png' else 3
                    if best_d != target and ((n == '截圖 2026-02-24 10.43.16.png' and y==0 and x==2) or (n == '截圖 2026-02-24 11.21.56.png' and y==0 and x==6)):
                        print(f"  ({y},{x}) -> best=({best_d},{best_s:.2f}) [Scores for {target}: {scores.get(target, 0):.2f}] Top3: {sorted_scores[:3]}")
                    elif ((n == '截圖 2026-02-24 10.43.16.png' and y==0 and x==2) or (n == '截圖 2026-02-24 11.21.56.png' and y==0 and x==6)):
                        print(f"  ({y},{x}) -> FIX SUCCESSFUL!! best=({best_d},{best_s:.2f})")
