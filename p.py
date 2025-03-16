import cv2
import numpy as np
import rembg

class BackgroundReplacer:
    def __init__(self, foreground_path, background_path, output_path):
        self.foreground_path = foreground_path
        self.background_path = background_path
        self.output_path = output_path
        self.foreground = None
        self.background = None
        self.foreground_masked = None
        self.mask = None
        self.x_offset = 0  # Offset for person movement (global)
        self.y_offset = 0
        self.bg_x_offset = 0
        self.bg_y_offset = 0
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.person_rois = []  # List of ROIs for multiple people
        self.selected_person_index = None
        self.bg_dragging = False
        self.movement_mode = "background"  # "background" or "person"

    def load_images(self):
        self.foreground = cv2.imread(self.foreground_path)
        self.background = cv2.imread(self.background_path)

        if self.foreground is None or self.background is None:
            raise ValueError("Error: Could not load one or both images.")

        self.background = cv2.resize(self.background, (self.foreground.shape[1], self.foreground.shape[0]))

        model = rembg.remove
        foreground_without_bg = model(self.foreground)
        foreground_without_bg = np.array(foreground_without_bg)

        alpha_channel = foreground_without_bg[:, :, 3]
        foreground_rgb = foreground_without_bg[:, :, :3]
        _, self.mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)

        self.foreground_masked = cv2.bitwise_and(foreground_rgb, foreground_rgb, mask=self.mask)

        self.person_rois = [(0, 0, self.foreground.shape[1], self.foreground.shape[0])]  # Initial ROI (entire image)
        self.selected_person_index = 0

    def update_background(self):
        new_background = self.background.copy()

        M = np.float32([[1, 0, self.bg_x_offset], [0, 1, self.bg_y_offset]])
        rows, cols = new_background.shape[:2]
        new_background = cv2.warpAffine(new_background, M, (cols, rows))

        for x, y, w, h in self.person_rois:
            # Boundary checks for ROI (essential!)
            x = max(0, x)
            y = max(0, y)
            w = min(w, self.foreground.shape[1] - x) #prevent w from being too large
            h = min(h, self.foreground.shape[0] - y) #prevent h from being too large

            cv2.rectangle(new_background, (x + self.x_offset, y + self.y_offset), (x + w + self.x_offset, y + h + self.y_offset), (0, 255, 0), 2)  # Draw ROI

            y_start = y + self.y_offset
            x_start = x + self.x_offset
            y_end = y_start + h
            x_end = x_start + w

            y_start = max(0, y_start)
            y_end = min(new_background.shape[0], y_end)
            x_start = max(0, x_start)
            x_end = min(new_background.shape[1], x_end)


            foreground_to_paste = self.foreground_masked[y:y+h, x:x+w] #slicing according to ROI
            mask_to_paste = self.mask[y:y+h, x:x+w]

            new_background[y_start:y_end, x_start:x_end] = np.where(mask_to_paste[:,:,np.newaxis] > 0, foreground_to_paste, new_background[y_start:y_end, x_start:x_end])

        return new_background
    def mouse_callback(self, event, x, y, flags, param):
        if self.movement_mode == "background":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.bg_dragging = True
                self.drag_start_x = x
                self.drag_start_y = y
            elif event == cv2.EVENT_LBUTTONUP:
                self.bg_dragging = False
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.bg_dragging:
                    dx = x - self.drag_start_x
                    dy = y - self.drag_start_y
                    self.bg_x_offset += dx
                    self.bg_y_offset += dy
                    self.drag_start_x = x
                    self.drag_start_y = y
        elif self.movement_mode == "person":
            if event == cv2.EVENT_LBUTTONDOWN:
                for i, (x_start, y_start, width, height) in enumerate(self.person_rois):
                    if x_start + self.x_offset < x < x_start + width + self.x_offset and y_start + self.y_offset < y < y_start + height + self.y_offset:
                        self.selected_person_index = i
                        self.person_selected = True
                        self.dragging = True
                        self.drag_start_x = x
                        self.drag_start_y = y
                        break  # Important: Exit loop after finding the selected person

            elif event == cv2.EVENT_LBUTTONUP:
                self.dragging = False
                self.person_selected = False
                self.selected_person_index = None

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.dragging and self.person_selected and self.selected_person_index is not None:
                    dx = x - self.drag_start_x
                    dy = y - self.drag_start_y
                    x_start, y_start, width, height = self.person_rois[self.selected_person_index]
                    self.person_rois[self.selected_person_index] = (x_start + dx, y_start + dy, width, height) #move the ROI
                    self.drag_start_x = x
                    self.drag_start_y = y

            elif event == cv2.EVENT_RBUTTONDOWN:
                for i, (x_start, y_start, width, height) in enumerate(self.person_rois):
                    if x_start + self.x_offset < x < x_start + width + self.x_offset and y_start + self.y_offset < y < y_start + height + self.y_offset:
                        self.selected_person_index = i
                        self.person_selected = True
                        self.dragging = False
                        self.roi_start_x = x
                        self.roi_start_y = y
                        break

            elif event == cv2.EVENT_RBUTTONUP:
                if self.person_selected and self.selected_person_index is not None:
                    x_end = x
                    y_end = y
                    x_start, y_start, width, height = self.person_rois[self.selected_person_index]
                    self.person_rois[self.selected_person_index] = (min(x_start + self.x_offset, x_end), min(y_start + self.y_offset, y_end), abs(x_end - x_start), abs(y_end - y_start)) #change ROI
                    self.person_selected = False
                    self.dragging = False
                    self.selected_person_index = None

            elif event == cv2.EVENT_MBUTTONDOWN:  # Add new person ROI
                self.person_rois.append((x - self.x_offset, y - self.y_offset, 50, 50))  # Small default ROI

    def run(self):
        self.load_images()
        cv2.namedWindow("Background Replacement")
        cv2.setMouseCallback("Background Replacement", self.mouse_callback)

        def on_trackbar(val):
            if val == 0:
                self.movement_mode = "background"
            elif val == 1:
                self.movement_mode = "person"

        cv2.createTrackbar('Movement Mode', 'Background Replacement', 0, 1, on_trackbar)

        while True:
            new_background = self.update_background()
            cv2.imshow("Background Replacement", new_background)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                cv2.imwrite(self.output_path, new_background)
                print(f"Image saved to {self.output_path}")
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

# Example usage:
foreground_path = "Screenshot 2025-03-16 at 9.14.56 AM.png"  # Replace with your foreground image path
background_path = "Unknown.jpeg"  # Replace with your background image path
output_path = "output_image1.jpg"  # Replace with desired output path
replacer = BackgroundReplacer(foreground_path, background_path, output_path)
replacer.run()