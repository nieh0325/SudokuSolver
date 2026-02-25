import cv2, os, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, extract_templates_from_buttons

def extract_grid_features(img, grid_size=(4, 4)):
    """Divide image into grid and compute pixel density for each cell"""
    if img is None or img.size == 0:
        return np.zeros(grid_size[0] * grid_size[1])
        
    h, w = img.shape
    features = []
    
    cell_h = h / grid_size[0]
    cell_w = w / grid_size[1]
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y1, y2 = int(i * cell_h), int((i + 1) * cell_h)
            x1, x2 = int(j * cell_w), int((j + 1) * cell_w)
            
            # 確保邊界
            y2 = min(y2, h)
            x2 = min(x2, w)
            
            cell = img[y1:y2, x1:x2]
            if cell.size == 0:
                density = 0
            else:
                density = np.sum(cell > 128) / (cell.size)
            features.append(density)
            
    return np.array(features)

def match_digit_grid(dr, templates):
    best_score = float('inf')  # Lower distance is better
    best_digit = 0
    
    # 二值化 dr，讓它跟 template 一樣
    _, dr_bin = cv2.threshold(dr, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Crop to bounding box 避免周圍有不必要的黑邊影響特徵
    points = cv2.findNonZero(dr_bin)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        dr_bin = dr_bin[y:y+h, x:x+w]
        
    dr_features = extract_grid_features(dr_bin, (5, 4))
    
    for d, t in templates.items():
        # 對 template 也做 crop，雖然通常 template 已經很乾淨
        t_points = cv2.findNonZero(t)
        if t_points is not None:
            tx, ty, tw, th = cv2.boundingRect(t_points)
            t_cropped = t[ty:ty+th, tx:tx+tw]
        else:
            t_cropped = t
            
        t_features = extract_grid_features(t_cropped, (5, 4))
        
        # Calculate Euclidean distance between feature vectors
        dist = np.linalg.norm(dr_features - t_features)
        
        # Bonus: Add aspect ratio penalty
        t_ratio = t_cropped.shape[1] / float(t_cropped.shape[0])
        dr_ratio = dr_bin.shape[1] / float(dr_bin.shape[0])
        ratio_diff = abs(t_ratio - dr_ratio)
        
        dist += ratio_diff * 0.5
        
        if dist < best_score:
            best_score = dist
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
                    d, dist = match_digit_grid(dr, templates)
                    print(f"  ({y},{x}) -> dr=({dr.shape[1]},{dr.shape[0]}) best=({d}, dist={dist:.3f})")
