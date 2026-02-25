import cv2, os, numpy as np

names = ['截圖 2026-02-24 09.37.16.png', '截圖 2026-02-24 11.07.34.png', '截圖 2026-02-24 11.21.56.png']

for n in names:
    print(f'=== {n} ===')
    for i in range(1, 10):
        p = f'debug_{n}/{i}.png'
        if os.path.exists(p):
            img = cv2.imread(p, 0)
            if img is not None:
                print(f'Digit {i}: shape={img.shape}')
                img_r = cv2.resize(img, (10, 10))
                for r in img_r:
                    print(''.join(['#' if x > 128 else '.' for x in r]))
            else:
                print(f'Digit {i}: Failed to load image')
        else:
            print(f'Digit {i}: File not found ({p})')
