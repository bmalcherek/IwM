import matplotlib.pyplot as plt
import pydicom
import numpy as np
import math
import cv2
from skimage.draw import line
from pydicom.pixel_data_handlers import gdcm_handler, pillow_handler

def radon(img, num_detectors, angular_step, theta, plt):
    height, width = img.shape
    height = height
    width = width
    radius = width / 2 if width < height else height / 2
    
    x_offset = math.floor(width / 2)
    y_offset = math.floor(height / 2)
    
    num_steps = math.floor((2 * math.pi) / angular_step)

    sinogram = np.empty([num_steps, num_detectors])
    padded_img = cv2.copyMakeBorder(img, 0, 1, 0, 1, cv2.BORDER_CONSTANT, 0)

    for iteration in range(0, num_steps):
        emitter_angle = iteration * angular_step
        emitter = (math.floor(radius * math.cos(emitter_angle) + x_offset), math.floor(radius * math.sin(emitter_angle) + y_offset))
        # plt.plot(*emitter, 'go')
        for i in range(0, num_detectors):
            detector_angle = emitter_angle + math.pi - theta / 2 + i * (theta/(num_detectors - 1))
            detector = (math.floor(radius * math.cos(detector_angle + x_offset)), math.floor(radius * math.sin(detector_angle) + y_offset))
            # plt.plot(*detector, 'ro')
            rr, cc = line(*emitter, *detector)
            # plt.plot(rr, cc, '-')
            mean_brightness = sum(padded_img[rr, cc]) / len(rr)
            sinogram[iteration, i] = mean_brightness

    return sinogram


def load_image(path):
    if path.endswith('.dcm'):
        dcm_source = pydicom.dcmread(path)
        print(dcm_source.pixel_array[0])
        return dcm_source.pixel_array[0]
    return cv2.imread(path)


def main():
    path = input('What image do you want to import: ')
    image = load_image(path)
    r = radon(image, 180, math.pi/90, math.pi, plt)
    plt.imshow(r, cmap=plt.cm.bone)
    plt.show()
    # pydicom.config.image_handlers = [gdcm_handler, pillow_handler]
    # image = pydicom.dcmread('0002.dcm')
    # plt.imshow(image.pixel_array, cmap=plt.cm.bone)
    # plt.show()


if __name__ == '__main__':
    main()
