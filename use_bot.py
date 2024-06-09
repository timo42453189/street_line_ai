from camera.camera import Cam
import numpy as np
import serial
import cv2

c = Cam(index=[0])
#ser = serial.Serial("COMX", 9600)

while True:
    image = c.get_frame()
    median_blur = cv2.medianBlur(image, 5)
    downsampled = cv2.resize(median_blur, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    downsampled = downsampled[110:190, :]
    _, thresholded = cv2.threshold(downsampled, 128, 255, cv2.THRESH_BINARY)
    height, width, _ = thresholded.shape
    middle = width // 2

    left_half = thresholded[:, :middle]
    right_half = thresholded[:, middle:]
    gray_image_left = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
    gray_image_right = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)

    print("gray_image_left: ", np.mean(gray_image_left))
    print("gray_image_right: ", np.mean(gray_image_right))
    if np.mean(gray_image_left) > np.mean(gray_image_right):
        #ser.write(str(int(13)).encode())
        print(1)
    else:
        #ser.write(str(int(7)).encode())
        
        print(0)
