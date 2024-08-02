from data_storing.store_data import StoreData
from camera.camera import Cam


s = StoreData()
c = Cam(index=[])

image, pot = s.read("22.h5")
c.show_image(pot)