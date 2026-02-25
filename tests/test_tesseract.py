import cv2, os, pytesseract, numpy as np
from src.extractor import get_clean_digit, detect_all_lines, ocr_digit_fallback

for n in ['截圖 2026-02-24 10.43.16.png', '截圖 2026-02-24 11.21.56.png']:
    image = cv2.imread(os.path.join('題目', n))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hl, vl = detect_all_lines(gray)
    
    print(n)
    for y in range(9):
        for x in range(9):
            if ((n == '截圖 2026-02-24 10.43.16.png' and y==0 and x==2) or (n == '截圖 2026-02-24 11.21.56.png' and y==0 and x==6)):
                sy, ey, sx, ex = hl[y], hl[y+1], vl[x], vl[x+1]
                my, mx = int((ey-sy)*0.13), int((ex-sx)*0.13)
                cell_img = gray[sy+my:ey-my, sx+mx:ex-mx]
                res, dr = get_clean_digit(cell_img)
                
                # Print clean digit
                if dr is not None:
                    print(f"  ({y},{x}) -> clean digit shape: {dr.shape}")
                    # Try ocr_digit_fallback directly
                    ans = ocr_digit_fallback(cell_img)
                    print(f"  ({y},{x}) fallback returned: {ans}")
                    
                    # Try Tesseract on clean digit directly
                    try:
                        _, th = cv2.threshold(dr, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        th = cv2.copyMakeBorder(th, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0,0,0])
                        th_inv = cv2.bitwise_not(th)
                        
                        # Pad with white is better for tesseract
                        config = '--oem 3 --psm 10 -c tessedit_char_whitelist=123456789'
                        txt = pytesseract.image_to_string(th_inv, config=config).strip()
                        print(f"  ({y},{x}) custom tesseract: {txt}")
                    except Exception as e:
                        print(f"  Error: {e}")
