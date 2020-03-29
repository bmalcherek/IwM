"""Tomograph simulator for IwM Class in PUT"""

import os
import tkinter as tk
import math
from PIL import Image, ImageTk

import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sinogram import Sinogram


def load_image(root, path):
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

        sinogram_settings(root, path)

        root.mainloop()
    except Exception as e:
        print(e)


def dropdown(root):
    '''Setup image loading widgets'''

    files_list = [file for file in os.listdir('./images/') if file.endswith(('.dcm', '.jpg', '.png'))]
    file = tk.StringVar(root)
    file.set(files_list[0])

    image_label = tk.Label(root, text="Image path:")
    image_label.place(relx=0.02, rely=0.035)

    file = tk.StringVar(root)
    file.set(files_list[0])

    drop_menu = tk.OptionMenu(root, file, *files_list)
    drop_menu.place(relx=0.25, rely=0.03)
    drop_menu.config(width=30)
    load_btn = tk.Button(
        root,
        text="Load Image",
        command=lambda: load_image(root, f'./images/{file.get()}')
        )
    load_btn.place(relx=0.8, rely=0.03)

    return f'./images/{file.get()}'


def only_numbers(char):
    #TODO get it to work
    print(char)
    return char.isdigit()


def sinogram_settings(root, img_path):
    radon_filter = tk.BooleanVar(root)
    filter_check = tk.Checkbutton(root, text="Filter", variable=radon_filter)
    filter_check.select()
    filter_check.place(relx=0.03, rely=0.37)

    radon_gauss = tk.BooleanVar(root)
    gauss_check = tk.Checkbutton(root, text="Gauss", variable=radon_gauss)
    gauss_check.select()
    gauss_check.place(relx=0.15, rely=0.37)

    validation = root.register(only_numbers)

    radon_steps_label = tk.Label(root, text="Steps:")
    radon_steps_label.place(relx=0.25, rely=0.37)
    radon_steps = tk.StringVar(root, value=180)
    radon_steps_input = tk.Entry(
        root,
        textvariable=radon_steps,
        width=10,
        validate='key',
        validatecommand=(validation, '%S')
    )
    radon_steps_input.place(relx=0.29, rely=0.37)

    radon_detectors_label = tk.Label(root, text='Detectors:')
    radon_detectors_label.place(relx=0.385, rely=0.37)
    radon_detectors = tk.StringVar(root, value=180)
    radon_detectors_input = tk.Entry(
        root,
        textvariable=radon_detectors,
        width=10
    )
    radon_detectors_input.place(relx=0.45, rely=0.37)

    radon_theta_label = tk.Label(root, text="Theta:")
    radon_theta_label.place(relx=0.54, rely=0.37)
    radon_theta = tk.StringVar(root, value=180)
    radon_theta_input = tk.Entry(
        root,
        textvariable=radon_theta,
        width=10
    )
    radon_theta_input.place(relx=0.582, rely=0.37)

    start_btn = tk.Button(
        root,
        text="Start",
        command=lambda: sinogram(
            root,
            img_path,
            radon_filter,
            radon_gauss,
            radon_steps,
            radon_detectors,
            radon_theta
            )
    )
    start_btn.place(relx=0.705, rely=0.368)

    root.mainloop()


def update_iradon_image(iradon_img, idx, iradon_all):
    iradon_img.configure(image=iradon_all[idx.get()])
    iradon_img.image = iradon_all[idx.get()]


def sinogram(root, img_path, filter_, gauss, steps, detectors, theta):
    img = cv2.imread(img_path, 0)

    sin = Sinogram(
        image=img,
        num_steps=int(steps.get()),
        num_detectors=int(detectors.get()),
        theta=math.radians(float(theta.get())),
        filter=filter_.get(),
        gaussian=gauss.get()
    )

    sin_label = tk.Label(root, text='Sinogram')
    sin_label.place(relx=0.03, rely=0.54)
    sin_im = Image.fromarray(sin.get_sinogram()).resize((200, 200))
    sin_imgtk = ImageTk.PhotoImage(image=sin_im)
    sin_img = tk.Label(root, image=sin_imgtk)
    sin_img.place(relx=0.25, rely=0.42)

    iradon_label = tk.Label(root, text='Inverse Radon')
    iradon_label.place(relx=0.03, rely=0.82)
    iradon_all = [ImageTk.PhotoImage(image=Image.fromarray(irad_part).resize((200, 200)))
                  for irad_part in sin.get_backprojection_frames()]
    iradon_im = Image.fromarray(sin.get_backprojection()).resize((200, 200))
    iradon_imgtk = ImageTk.PhotoImage(image=iradon_im)
    iradon_img = tk.Label(root, image=iradon_imgtk)
    iradon_img.place(relx=0.25, rely=0.7)

    idx_var = tk.IntVar(root)
    idx_var.set(len(iradon_all) - 1)
    slider = tk.Scale(
        from_=0,
        to=len(iradon_all)-1,
        orient=tk.HORIZONTAL,
        variable=idx_var,
        command=lambda x: update_iradon_image(iradon_img, idx_var, iradon_all)
        )
    slider.place(relx=0.5, rely=0.79)

    root.mainloop()


def window_setup():
    '''Root setup and call other widgets initialization functions'''
    root = tk.Tk()
    root.resizable(width=False, height=False)
    root.title('TOMOGRAPH')
    root.geometry('1000x800')

    dropdown(root)
    # sinogram_settings(root, image_path)

    root.mainloop()

    return root


def main():
    '''main'''
    window_setup()


if __name__ == '__main__':
    main()
