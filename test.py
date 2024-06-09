import cv2
import numpy as np
from data_storing.store_data import StoreData
from camera.camera import Cam
import os
from camera.camera import Cam

c = Cam(index=[])
s = StoreData()


image, pot = s.read("0.h5")
print(pot)
c.show_image(image)