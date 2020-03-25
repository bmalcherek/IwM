import matplotlib.pyplot as plt
import pydicom
import numpy as np
import math
import cv2
from skimage.draw import line
from pydicom.pixel_data_handlers import gdcm_handler, pillow_handler

def radon(img, num_detectors, angular_step, theta, plt):
    height, width = img.shape
    radius = width / 2 if width < height else height / 2
    
    x_offset = math.floor(width / 2)
    y_offset = math.floor(height / 2)
    
    num_steps = math.floor((math.pi) / angular_step)

    sinogram = np.zeros([num_steps, num_detectors], dtype=np.uint8)
    padded_img = cv2.copyMakeBorder(img, 0, 1, 0, 1, cv2.BORDER_CONSTANT, 0)

    for iteration in range(0, num_steps):
        emitter_angle = iteration * angular_step
        emitter = (math.floor(radius * math.cos(emitter_angle) + x_offset), math.floor(radius * math.sin(emitter_angle) + y_offset))
        # plt.plot(*emitter, 'go')
        for i in range(0, num_detectors):
            detector_angle = emitter_angle + math.pi - theta / 2 + i * (theta/(num_detectors - 1))
            detector = (math.floor(radius * math.cos(detector_angle) + x_offset), math.floor(radius * math.sin(detector_angle) + y_offset))
            # plt.plot(*detector, 'ro')
            rr, cc = line(emitter[1], emitter[0], detector[1], detector[0])
            # plt.plot(rr, cc, '-')
            # mean_brightness = math.floor(sum(padded_img[rr, cc]) / len(rr))
            mean_brightness = math.floor(np.array(padded_img[rr, cc]).mean())
            print(mean_brightness)
            sinogram[iteration][i] = mean_brightness

    return sinogram

def radon_inverse(sinogram, width, height, theta):
    radius = width / 2 if width < height else height / 2

    x_offset = math.floor(width / 2)
    y_offset = math.floor(height / 2)

    img = np.zeros([width + 1, height + 1], dtype=np.uint8)
    num_steps, num_detectors = sinogram.shape
    angular_step = (math.pi) / num_steps

    print(num_steps, num_detectors, angular_step)

    for i in range(0, num_steps):
        emitter_angle = i * angular_step
        emitter = (math.floor(radius * math.cos(emitter_angle) + x_offset), math.floor(radius * math.sin(emitter_angle) + y_offset))
        for j in range(0, num_detectors):
            detector_angle = emitter_angle + math.pi - theta / 2 + j * (theta/(num_detectors - 1))
            detector = (math.floor(radius * math.cos(detector_angle) + x_offset), math.floor(radius * math.sin(detector_angle) + y_offset))

            rr, cc = line(emitter[1], emitter[0], detector[1], detector[0])

            for y, x in zip(rr, cc):
                img[y, x] += sinogram[i][j]
    
    print(img)
    return img

def load_image(path):
    if path.endswith('.dcm'):
        dcm_source = pydicom.dcmread(path)
        print(dcm_source.pixel_array[0])
        return dcm_source.pixel_array[0]
    return cv2.imread(path)


def main():
    # path = input('What image do you want to import: ')
    image = cv2.imread('Kwadraty2.jpg', 0)
    r = radon(image, 180, math.pi/180, math.pi, plt)
    inv = radon_inverse(r, 512, 512, math.pi)
    plt.imshow(inv, cmap=plt.cm.bone)
    plt.show()
    # pydicom.config.image_handlers = [gdcm_handler, pillow_handler]
    # image = pydicom.dcmread('0002.dcm')
    # plt.imshow(image.pixel_array, cmap=plt.cm.bone)
    # plt.show()


if __name__ == '__main__':
    main()
