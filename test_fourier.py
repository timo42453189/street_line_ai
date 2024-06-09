import numpy as np
import matplotlib.pyplot as plt
from data_storing.store_data import StoreData
from camera.camera import Cam

c = Cam(index=[])
s = StoreData()

def preprocess_image(image):
    # Normalisieren Sie das Bild auf Werte zwischen 0 und 1
    image = image / 255.0
    # Wenden Sie die Fourier-Transformation an
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    return fshift

def inverse_fourier_transform(frequency_domain_image):
    f_ishift = np.fft.ifftshift(frequency_domain_image)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

# Beispielbild laden
image, pot = s.read("1.h5")

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Input Image')

prep_image = preprocess_image(image)

# Die Amplitudenspektrum zur Visualisierung der Fourier-Transformation
magnitude_spectrum = 20 * np.log(np.abs(prep_image) + 1)

plt.subplot(1, 3, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Fourier Transformed Image')

# Inverse Fourier-Transformation anwenden
inverse_image = inverse_fourier_transform(prep_image)

plt.subplot(1, 3, 3)
plt.imshow(inverse_image, cmap='gray')
plt.title('Fourier Inverse Image')
plt.show()