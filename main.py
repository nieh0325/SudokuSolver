import os
import cv2
import numpy as np
import sys
import copy
from src.extractor import process_full_image, validate_initial_grid
from src.solver import solve
from src.renderer import render_solved_grid

def process_image(input_path, output_path):
    print(f"讀取圖片：{input_path}...")
    image = cv2.imread(input_path)
    if image is None:
        print(f"錯誤：無法讀取圖片 {input_path}")
        return False

    print("萃取網格中...")
    try:
        # returns grid, top, bottom, left, right
        original_grid, top, bottom, left, right = process_full_image(image)
        warped = image[top:bottom, left:right]
        maxDim = warped.shape[0]
        M = np.float32([[1, 0, -left], [0, 1, -top], [0, 0, 1]])
    except Exception as e:
        print(f"網格萃取與辨識錯誤：{e}")
        import traceback
        traceback.print_exc()
        return False

    print("檢查辨識結果...")
    conflicts = validate_initial_grid(original_grid)
    if conflicts:
        print("警告：辨識出的網格存在衝突，可能識別有誤：")
        for c in conflicts:
            print(f"  - {c}")
        # We can still try to solve it, or stop here. Let's stop to avoid nonsensical attempts.
        # However, for debugging we print the grid.
        print("辨識出的網格：")
        for row in original_grid:
            print(" ".join(str(x) if x != 0 else "." for x in row))
        return False

    print("解題中...")
    puzzle_to_solve = copy.deepcopy(original_grid)
    if solve(puzzle_to_solve):
        print("數獨解題成功！")
        final_image = render_solved_grid(image, warped, M, maxDim, original_grid, puzzle_to_solve)
        cv2.imwrite(output_path, final_image)
        print(f"答案已儲存至：{output_path}")
        return True
    else:
        print("錯誤：無法解開該數獨。可能是 OCR 數字辨識錯誤或題目有誤。")
        # Print the grid for debugging
        print("辨識出的網格：")
        for row in original_grid:
            print(" ".join(str(x) if x != 0 else "." for x in row))
        return False

def main():
    input_dir = "題目"
    output_dir = "答案"

    # 若目錄不存在則自動建立
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 尋找支援的圖片格式
    valid_extensions = ('.png', '.jpg', '.jpeg')
    if not os.path.exists(input_dir):
        print(f"找不到 '{input_dir}' 目錄。")
        return
        
    files_to_process = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not files_to_process:
        print(f"在 '{input_dir}' 目錄中找不到圖片檔（.png, .jpg, .jpeg）。")
        print(f"請將您的數獨截圖放到 '{input_dir}' 目錄內再執行一次。")
        return

    for filename in files_to_process:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"\n--- 正在處理: {filename} ---")
        process_image(input_path, output_path)

if __name__ == "__main__":
    main()
