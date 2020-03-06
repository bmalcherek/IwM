import matplotlib.pyplot as plt
import pydicom
import numpy as np
import cv2
from pydicom.pixel_data_handlers import gdcm_handler, pillow_handler


def load_image(path):
    if path.endswith('.dcm'):
        dcm_source = pydicom.dcmread(path)
        print(dcm_source.pixel_array[0])
        return dcm_source.pixel_array[0]
    return cv2.imread(path)


def main():
    path = input('What image do you want to import: ')
    image = load_image(path)
    plt.imshow(image, cmap=plt.cm.bone)
    plt.show()
    # pydicom.config.image_handlers = [gdcm_handler, pillow_handler]
    # image = pydicom.dcmread('0002.dcm')
    # plt.imshow(image.pixel_array, cmap=plt.cm.bone)
    # plt.show()


if __name__ == '__main__':
    main()
