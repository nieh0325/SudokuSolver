import cv2, os, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, extract_templates_from_buttons

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
            res, dr = get_clean_digit(gray[sy+my:ey-my, sx+mx:ex-mx])
            if dr is not None:
                scores = {}
                for d, t in templates.items():
                    th, tw = t.shape[:2]
                    resized_dr = cv2.resize(dr, (tw, th))
                    score = cv2.matchTemplate(resized_dr, t, cv2.TM_CCOEFF_NORMED)[0][0]
                    ratio_diff = abs((res[2]/float(res[3])) - (tw/float(th)))
                    if ratio_diff > 0.15: score -= 0.2
                    scores[d] = score
                if scores:
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    if sorted_scores[0][0] == 3 and sorted_scores[0][1] < 0.6: 
                        print(f"  ({y},{x}) -> best={sorted_scores[0]}, 2nd={sorted_scores[1]}")
