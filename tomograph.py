"""Tomograph simulator for IwM Class in PUT"""

import os
import tkinter as tk
import math
from PIL import Image, ImageTk

import pydicom
from pydicom.data import get_testdata_files
from pydicom.tag import Tag
import numpy as np
import cv2

from sinogram import Sinogram


def load_image(root, path):
    '''Load .dcm or other image from path and displays it'''

    try:
        patient_data = patient_data_widgets(root)

        img = None
        if path.endswith('.dcm'):
            dcm_source = pydicom.dcmread(path)
            print(dcm_source.StudyDescription)
            patient_data['name'].set(dcm_source.PatientName)
            patient_data['date'].set(dcm_source.StudyDate)
            patient_data['description'].insert(tk.END, dcm_source.StudyDescription)
            img = Image.fromarray(np.uint8(dcm_source.pixel_array * 255))           
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = Image.fromarray(img)

        img = img.resize((200, 200))
        width, height = img.size
        imgtk = ImageTk.PhotoImage(img)
        img = np.array(img)

        img_label = tk.Label(root, text="Original image")
        img_label.place(relx=0.02, rely=0.10)
        img_widget = tk.Label(root, image=imgtk)
        img_widget.place(relx=0.25, rely=0.10, width=width, height=height)

        sinogram_settings(root, img, path, patient_data)

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
    load_btn.place(relx=0.6, rely=0.03)

    return f'./images/{file.get()}'


def patient_data_widgets(root):
    name = tk.StringVar(root)
    name_label = tk.Label(root, text='Name:')
    name_label.place(relx=0.55, rely=0.1)
    name_entry = tk.Entry(root, textvariable=name)
    name_entry.place(relx=0.63, rely=0.1)

    date = tk.StringVar(root)
    date_label = tk.Label(root, text='Date:')
    date_label.place(relx=0.55, rely=0.14)
    date_entry = tk.Entry(root, textvariable=date)
    date_entry.place(relx=0.63, rely=0.14)

    description_label = tk.Label(root, text='Description:')
    description_label.place(relx=0.55, rely=0.18)
    description_entry = tk.Text(root, width=40, height=6)
    description_entry.place(relx=0.63, rely=0.18)

    patient_data = {
        'name': name,
        'date': date,
        'description': description_entry
    }

    return patient_data

def sinogram_settings(root, img_path, path, patient_data):
    radon_filter = tk.BooleanVar(root)
    filter_check = tk.Checkbutton(root, text="Filter", variable=radon_filter)
    filter_check.select()
    filter_check.place(relx=0.03, rely=0.37)

    radon_gauss = tk.BooleanVar(root)
    gauss_check = tk.Checkbutton(root, text="Gauss", variable=radon_gauss)
    gauss_check.select()
    gauss_check.place(relx=0.15, rely=0.37)

    radon_steps_label = tk.Label(root, text="Steps:")
    radon_steps_label.place(relx=0.25, rely=0.37)
    radon_steps = tk.StringVar(root, value=180)
    radon_steps_input = tk.Entry(
        root,
        textvariable=radon_steps,
        width=10
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
            radon_theta,
            path,
            patient_data
            )
    )
    start_btn.place(relx=0.705, rely=0.368)

    root.mainloop()


def update_iradon_image(iradon_img, idx, iradon_all):
    iradon_img.configure(image=iradon_all[idx.get()])
    iradon_img.image = iradon_all[idx.get()]


def sinogram(root, img, filter_, gauss, steps, detectors, theta, path, patient_data):
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

    result = sin.get_backprojection()
    save_to_dicom_widgets(root, path, result, patient_data)

    root.mainloop()


def save_to_dicom(path, result, patient_data):
    path = f"./results/{path.split('/')[-1].split('.')[0]}.dcm"

    filename = get_testdata_files('CT_small.dcm')[0]
    template_file = pydicom.dcmread(filename)
    template_file.PatientName = patient_data['name'].get()
    template_file.StudyDate = patient_data['date'].get()
    template_file.StudyDescription = patient_data['description'].get("1.0", tk.END)
    template_file.Rows = result.shape[0]
    template_file.Columns = result.shape[1]
    result_dcm = result * 1024
    np.round(result_dcm, decimals=0, out=result_dcm)
    result_dcm = result_dcm.astype('int16')
    template_file.PixelData = result_dcm.tobytes()
    template_file.save_as(path)


def save_to_dicom_widgets(root, path, result, patient_data):
    save_btn = tk.Button(
        root,
        text='SAVE',
        command=lambda: save_to_dicom(path, result, patient_data),
        width=20,
        height=6
    )
    save_btn.place(relx=0.75, rely=0.75)


def window_setup():
    '''Root setup and call other widgets initialization functions'''
    root = tk.Tk()
    root.resizable(width=False, height=False)
    root.title('TOMOGRAPH')
    root.geometry('1000x800')

    dropdown(root)

    root.mainloop()

    return root


def main():
    '''main'''
    window_setup()


if __name__ == '__main__':
    main()
