import cv2
import numpy as np
from pathlib import Path

img_path = Path("data/images/your_image.png")
mask_path = Path("data/masks/your_image.png")

# Load grayscale SAR image
img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
mask = np.zeros_like(img, np.uint8)

drawing = False
radius = 5  # brush size

def draw_circle(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(mask, (x, y), radius, 255, -1)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(mask, (x, y), radius, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mask, (x, y), radius, 255, -1)

cv2.namedWindow("Draw Mask")
cv2.setMouseCallback("Draw Mask", draw_circle)

while True:
    overlay = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
    cv2.imshow("Draw Mask", overlay)
    k = cv2.waitKey(1)
    if k == 27:  # ESC to exit
        break

cv2.imwrite(str(mask_path), mask)
cv2.destroyAllWindows()
print(f"Saved mask to {mask_path}")
