import os
import subprocess

images = os.listdir("data_storing/database")
start_file = "491.h5"
index = images.index(start_file)
images = images[index:]
images_left = len(images)
for i in images:
    print(i)
    print("Images left: ", images_left)
    images_left -= 1
    subprocess.call(["python", "make_data.py", i])