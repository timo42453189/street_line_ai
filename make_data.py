from data_storing.store_data import StoreData
import cv2
import sys
s = StoreData()
import time
class DrawLineWidget(object):
    def __init__(self, image_name):
        self.count = 0
        self.img, self.pot = s.read(image_name)
        self.median_blur = cv2.medianBlur(self.img, 5)
        self.downsampled = cv2.resize(self.median_blur, (self.img.shape[1] // 2, self.img.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
        self.original_image = self.downsampled[110:190, :]
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.count == 0:
                self.image_coordinates = [(x,y)]
            else:
                self.image_coordinates.append((x,y))

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            self.count += 1
            print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))

            # Draw line
            cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (255,255,255), 2)
            cv2.imshow("image", self.clone) 
            if self.count == 2:
                s.store_automatic(self.original_image,[[self.image_coordinates[0], self.image_coordinates[1]], [self.image_coordinates[2], self.image_coordinates[3]]])
                exit()

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone



draw_line_widget = DrawLineWidget(str(sys.argv[1]))
cv2.imshow('image', draw_line_widget.show_image())
cv2.moveWindow('image', 800, 300)
cv2.waitKey(0)