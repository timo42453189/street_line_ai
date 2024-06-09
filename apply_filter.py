import cv2
import os
from data_storing.store_data import StoreData

s = StoreData()
y=0
images = os.listdir("data_storing/train_images")
for i in images:
    img, pot = s.read(i)
    ############################################# Normal Image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    equalized = cv2.equalizeHist(blurred)
    s.store_automatic(equalized, pot) 
    ############################################# Bright Image
    bright_image = cv2.convertScaleAbs(img, alpha=1.4, beta=0)
    gray = cv2.cvtColor(bright_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    equalized = cv2.equalizeHist(blurred)
    s.store_automatic(equalized, pot) 
    ############################################# Dark Image
    low_image = cv2.convertScaleAbs(img, alpha=0.6, beta=0)
    gray = cv2.cvtColor(low_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    equalized = cv2.equalizeHist(blurred)
    s.store_automatic(equalized, pot) 
    ############################################# Flipped Image
    flipped_image = cv2.flip(equalized, 1)
    height, width = flipped_image.shape
    new_lines = []
    for line in pot:
        (x1, y1), (x2, y2) = line
        new_x1 = width - x1
        new_x2 = width - x2
        new_lines.append([[new_x1, y1], [new_x2, y2]])
    s.store_automatic(flipped_image, new_lines)
    print(y)
    y+=1