import cv2, os, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, extract_templates_from_buttons

# Extract all templates from all images to build a training set
training_data = []
training_labels = []
std_w, std_h = 24, 36

for n in [f for f in os.listdir('題目') if f.endswith('.png')]:
    try:
        img = cv2.imread(os.path.join('題目', n))
        temps = extract_templates_from_buttons(img)
        for d, t in temps.items():
            t_points = cv2.findNonZero(t)
            if t_points is not None:
                tx, ty, tw, th = cv2.boundingRect(t_points)
                t_cropped = t[ty:ty+th, tx:tx+tw]
                r_t = cv2.resize(t_cropped, (std_w, std_h))
                # Flatten the image to use as feature vector
                training_data.append(r_t.flatten().astype(np.float32))
                training_labels.append(d)
    except Exception as e:
        print(f"Error loading {n}: {e}")

training_data = np.array(training_data)
training_labels = np.array(training_labels)

print(f"Loaded {len(training_labels)} training samples.")

# Train KNN Model
knn = cv2.ml.KNearest_create()
knn.train(training_data, cv2.ml.ROW_SAMPLE, training_labels)

# Test on problematic grids
for n in ['截圖 2026-02-24 10.43.16.png', '截圖 2026-02-24 11.21.56.png', '截圖 2026-02-24 11.07.34.png']:
    image = cv2.imread(os.path.join('題目', n))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hl, vl = detect_all_lines(gray)
    
    print(n)
    for y in range(9):
        for x in range(9):
            if ((n == '截圖 2026-02-24 10.43.16.png' and y==0 and x==2) or (n == '截圖 2026-02-24 11.21.56.png' and y==0 and x==6) or (n == '截圖 2026-02-24 10.43.16.png' and y==1 and x==8) or (n == '截圖 2026-02-24 11.07.34.png' and y==1 and x==6)):
                sy, ey, sx, ex = hl[y], hl[y+1], vl[x], vl[x+1]
                my, mx = int((ey-sy)*0.13), int((ex-sx)*0.13)
                cell_img = gray[sy+my:ey-my, sx+mx:ex-mx]
                _, dr = get_clean_digit(cell_img)
                if dr is not None:
                    _, dr_bin = cv2.threshold(dr, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    points = cv2.findNonZero(dr_bin)
                    if points is not None:
                        x_p, y_p, w_p, h_p = cv2.boundingRect(points)
                        dr_bin = dr_bin[y_p:y_p+h_p, x_p:x_p+w_p]
                        
                    r_dr = cv2.resize(dr_bin, (std_w, std_h))
                    feat = r_dr.flatten().astype(np.float32).reshape(1, -1)
                    
                    ret, results, neighbours, dist = knn.findNearest(feat, k=3)
                    print(f"  ({y},{x}) -> best={int(results[0][0])}, neighbours={neighbours[0]}, dist={dist[0]}")
