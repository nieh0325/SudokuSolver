import cv2
import numpy as np

def render_solved_grid(original_image, warped_image, M, maxDim, original_grid, solved_grid):
    cell_size = int(maxDim / 9)
    solved_overlay = np.zeros_like(warped_image)
    
    for y in range(9):
        for x in range(9):
            if original_grid[y][x] == 0 and solved_grid[y][x] != 0:
                text = str(solved_grid[y][x])
                
                # Calculate text size and coordinates for centering
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = cell_size / 40.0
                font_thickness = max(1, int(cell_size / 20.0))
                
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                
                startX = x * cell_size
                startY = y * cell_size
                
                textX = startX + (cell_size - text_width) // 2
                textY = startY + (cell_size + text_height) // 2
                
                cv2.putText(solved_overlay, text, (int(textX), int(textY)), font, font_scale, (0, 0, 255), font_thickness)

    # Transform overlay back to original perspective
    # invert the perspective matrix
    M_inv = np.linalg.inv(M)
    overlay_original_perspective = cv2.warpPerspective(solved_overlay, M_inv, (original_image.shape[1], original_image.shape[0]))

    # Combine images
    # Create mask for overlay
    gray_overlay = cv2.cvtColor(overlay_original_perspective, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    img1_bg = cv2.bitwise_and(original_image, original_image, mask=mask_inv)
    img2_fg = cv2.bitwise_and(overlay_original_perspective, overlay_original_perspective, mask=mask)
    
    final_output = cv2.add(img1_bg, img2_fg)
    return final_output
