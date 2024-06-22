import numpy as np
from data_storing.store_data import StoreData
from camera.camera import Cam
import os
s = StoreData()


for i in os.listdir("data_storing/labled_images"):
    image, lines = s.read(i)
    s.store_automatic(image, lines)
    print(i)