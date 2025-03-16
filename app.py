import cv2
import numpy as np
from rembg import remove
from PIL import Image

# Global variables
persons = []
person_positions = []
selected_person = None
offset_x, offset_y = 0, 0
background = None
original_image = None

def load_image(image_path):
    """Load image, detect people, remove background, and store individual persons."""
    global persons, person_positions, background, original_image

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Image not found at: {image_path}")

    background_path = "images-1.jpeg"  # **REPLACE with YOUR background path**
    print(f"Trying to load background from: {background_path}") # Debugging
    background = cv2.imread(background_path)
    if background is None:
        raise ValueError(f"Background image not found at: {background_path}")

    # Resize background if needed
    if background.shape[0] < original_image.shape[0] or background.shape[1] < original_image.shape[1]:
        background = cv2.resize(background, (original_image.shape[1], original_image.shape[0]))

    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    removed_bg = remove(pil_image)
    removed_np = np.array(removed_bg)
    removed_cv = cv2.cvtColor(removed_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(removed_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 1000:  # Adjust this threshold if needed
            person = removed_cv[y:y + h, x:x + w]
            persons.append(person)
            person_positions.append((x, y))

    update_display()

def update_display():
    """Merge all persons onto the background."""
    global background

    final_image = background.copy()

    for i, person in enumerate(persons):
        x, y = person_positions[i]
        h, w = person.shape[:2]

        x = max(0, min(x, final_image.shape[1] - w))
        y = max(0, min(y, final_image.shape[0] - h))

        if person.size > 0:
            gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            roi = final_image[y:y + h, x:x + w]

            mask = cv2.resize(mask, (roi.shape[1], roi.shape[0])).astype(np.uint8)
            mask_inv = cv2.bitwise_not(mask)

            if roi.shape[2] == 3:  # If ROI is color, make mask 3-channel
                mask = np.stack([mask, mask, mask], axis=-1)
                mask_inv = np.stack([mask_inv, mask_inv, mask_inv], axis=-1)

            bg_part = cv2.bitwise_and(roi.copy(), roi.copy(), mask=mask_inv)
            fg_part = cv2.bitwise_and(person.copy(), person.copy(), mask=mask)
            final_image[y:y + h, x:x + w] = cv2.add(bg_part, fg_part)

    cv2.imshow("Image Editor", final_image)

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events."""
    global selected_person, offset_x, offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (px, py) in enumerate(person_positions):
            h, w = persons[i].shape[:2]
            if px <= x <= px + w and py <= y <= py + h:
                selected_person = i
                offset_x, offset_y = x - px, y - py
                break

    elif event == cv2.EVENT_MOUSEMOVE and selected_person is not None:
        person_positions[selected_person] = (x - offset_x, y - offset_y)
        update_display()

    elif event == cv2.EVENT_LBUTTONUP:
        selected_person = None

# Load image (REPLACE with your person image path)
image_path = "tinywow_IMG_0831_74286459.jpg"  # Example
try:
    load_image(image_path)
except ValueError as e:
    print(f"Error: {e}")
    exit()

cv2.namedWindow("Image Editor")
cv2.setMouseCallback("Image Editor", mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()