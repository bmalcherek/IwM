"""Tomograph simulator for IwM Class in PUT"""

import math
import os
import tkinter as tk
from PIL import Image, ImageTk

import pydicom
import numpy as np
import cv2

from skimage.draw import line

def radon(img, num_detectors, angular_step, theta, plt):
    height, width = img.shape
    height = height
    width = width
    radius = width / 2 if width < height else height / 2

    x_offset = math.floor(width / 2)
    y_offset = math.floor(height / 2)

    num_steps = math.floor((2 * math.pi) / angular_step)

    sinogram = np.empty([num_steps, num_detectors])
    padrop_menued_img = cv2.copyMakeBorder(img, 0, 1, 0, 1, cv2.BORDER_CONSTANT, 0)

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
            mean_brightness = sum(padrop_menued_img[rr, cc]) / len(rr)
            sinogram[iteration, i] = mean_brightness

    return sinogram


def load_image(path, root):
    '''Load .dcm or other image from path and displays it'''

    try:
        img = None
        if path.endswith('.dcm'):
            dcm_source = pydicom.dcmread(path)
            img = Image.fromarray(np.uint8(dcm_source.pixel_array[0] * 255))
        else:
            img = Image.open(path)

        img = img.resize((200, 200))
        width, height = img.size
        img = ImageTk.PhotoImage(img)

        img_label = tk.Label(root, text="Original image")
        img_label.place(relx=0.02, rely=0.10)
        img_widget = tk.Label(root, image=img)
        img_widget.place(relx=0.25, rely=0.10, width=width, height=height)

        print('Image Loaded')
        root.mainloop()
    except Exception as e:
        print(e)


def dropdown(root):
    '''Setup image loading widgets'''

    files_list = [file for file in os.listdir() if file.endswith(('.dcm', '.jpg', '.png'))]
    file = tk.StringVar(root)
    file.set(files_list[0])

    image_label = tk.Label(root, text="Image path:")
    image_label.place(relx=0.02, rely=0.035)

    file = tk.StringVar(root)
    file.set(files_list[0])

    drop_menu = tk.OptionMenu(root, file, *files_list)
    # drop_menu.pack()
    drop_menu.place(relx=0.25, rely=0.03)
    drop_menu.config(width=30)

    load_btn = tk.Button(
        root,
        text="Load Image",
        command=lambda: load_image(file.get(), root)
        )
    load_btn.place(relx=0.8, rely=0.03)

    return drop_menu


def window_setup():
    '''Root setup and call other widgets initialization functions'''
    root = tk.Tk()
    root.title('TOMOGRAPH')
    root.geometry('1000x1000')

    dropdown(root)

    root.mainloop()

    return root


def main():
    # path = input('What image do you want to import: ')
    # image = load_image(path)
    # r = radon(image, 180, math.pi/90, math.pi, plt)
    # plt.imshow(r, cmap=plt.cm.bone)
    # plt.show()
    window_setup()


if __name__ == '__main__':
    main()
