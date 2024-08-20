import matplotlib.pyplot as plt
import numpy as np
import cv2

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from data_storing.store_data import StoreData

def is_street_mask(mask):
    """
    Funktion zur Filterung von Straßenmasken anhand von heuristischen Merkmalen wie der Fläche und dem Verhältnis.
    Hier wird angenommen, dass Straßen tendenziell längere und rechteckigere Flächen sind.
    """
    segmentation = mask['segmentation']
    area = mask['area']
    bbox = mask['bbox']  # Bounding box im XYWH-Format (X, Y, Breite, Höhe)

    # Verhältnis von Breite zu Höhe berechnen
    aspect_ratio = bbox[2] / (bbox[3] + 1e-5)  # Vermeide Division durch Null

    # Heuristiken für Straßen:
    # 1. Breite zu Höhe Verhältnis größer als ein bestimmter Wert (z.B. 2.0)
    # 2. Mindestgröße der Fläche
    print(area)
    if aspect_ratio > 2.0 and area > 10000:  # Diese Werte können angepasst werden
        return True
    return False

def show_street_mask(image, masks):
    street_masks = [mask for mask in masks if is_street_mask(mask)]

    if len(street_masks) == 0:
        print("No street-like masks were found.")
        return

    sorted_street_masks = sorted(street_masks, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((image.shape[0], image.shape[1], 4))
    img[:, :, 3] = 0  # Set transparency

    for ann in sorted_street_masks:
        m = ann['segmentation']
        color_mask = np.array([1, 0, 0, 0.5])  # Rote Maske mit Transparenz 0.5 für Straßen
        img[m] = color_mask

    ax.imshow(img)

# Dein bestehender Code
s = StoreData()
image, _ = s.read2("390.h5")

print("LOADING MODEL")
sam = sam_model_registry["vit_h"](checkpoint="segment_model.pth")
print("SUCCESS")
mask_generator = SamAutomaticMaskGenerator(sam)
print("GENERATING MASKS")
masks = mask_generator.generate(image)
print("DONE")

if len(masks) > 0:
    print(f"{len(masks)} masks were found.")

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_street_mask(image, masks)  # Zeigt nur die Straßenmasken an
    plt.axis('off')
    plt.show()
else:
    print("No masks were found.")
