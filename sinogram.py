import cv2
from skimage.draw import line as bresenham
import math
import matplotlib.pyplot as plt
import numpy as np

class Sinogram:
    def __init__(self, img, num_detectors, steps, theta):
        self.img = img
        self.num_detectors = num_detectors
        self.steps = steps
        self.theta = theta

        self.width = img.shape[0]
        self.radius = self.width / 2 - 1

        self.offset = int(self.width / 2)
        
        self.gen_lines()

    
    def get_coords(self, alpha):
        x = int(math.cos(alpha) * self.radius) + self.offset
        y = -int(math.sin(alpha) * self.radius) + self.offset
        return x, y

    
    def gen_lines(self):
        self.scans = list()

        # plt.imshow(self.img)

        angular_step = math.pi * 2 / self.steps
        for step in range(self.steps):
            alpha = step * angular_step

            step_lines = list()
            for detector in range(self.num_detectors):
                detector_angle = alpha + math.pi - self.theta / 2 + detector * (self.theta / (self.num_detectors - 1))
                detector_x, detector_y = self.get_coords(detector_angle)

                emitter_angle = alpha + self.theta / 2 - detector * (self.theta / (self.num_detectors - 1))
                emitter_x, emitter_y = self.get_coords(emitter_angle)

                # plt.plot([emitter_x, detector_x], [emitter_y, detector_y], 'g-')

                rr, cc = bresenham(emitter_y, emitter_x, detector_y, detector_x)
                step_lines.append((rr, cc))
            

            self.scans.append(step_lines)

        # plt.show()

    def calc_filter(self, number_freq):
        filtering_array = 2 * np.arange(number_freq + 1) / np.float32(2 * number_freq)
        # w = 2 * np.pi * np.arange(number_freq + 1) / np.float32(2 * number_freq)
        # filtering_array[1:] *= (1.0 + np.cos(w[1:])/2.0)
        filtering_array = np.concatenate((filtering_array, filtering_array[number_freq - 1:0:-1]), axis=0)

        return filtering_array


    def filter_projection(self, sinogram):

        number_angles, number_offsets = sinogram.shape
        number_freq = 2 * int(2**(int(np.ceil(np.log2(number_offsets)))))

        filter_array = self.calc_filter(number_freq)

        # print(sinogram)

        padded_sinogram = np.concatenate((sinogram, np.zeros((number_angles, 2 * number_freq - number_offsets))), axis=1)

        for i in range(number_angles):
            padded_sinogram[i, :] = np.real(np.fft.ifft(np.fft.fft(padded_sinogram[i, :]) * filter_array))

        # sinogram[:, :] = padded_sinogram[:, :number_offsets]

        return padded_sinogram[:, :number_offsets]


    def gen_sinogram(self, iterations=None):

        self.sinogram = np.zeros((self.steps, self.num_detectors), dtype=np.float64)

        iterations = self.steps if iterations is None else iterations

        for iteration, scan in enumerate(self.scans):

            if iteration == iterations - 1:
                break

            for detector, line in enumerate(scan):
                rr, cc = line
                # mean_brightness = np.array(self.img[rr, cc]).mean()
                mean_brightness = np.array(self.img[rr, cc]).sum() / (255 * len(rr)) if len(rr) > 0 else 0
                mean_brightness = np.exp(-mean_brightness)
                self.sinogram[iteration, detector] = mean_brightness
        

        #normalize or sth
        max_value = np.max(self.sinogram)
        min_value = np.min(self.sinogram)
        factor = 256 / (max_value - min_value)
        self.sinogram -= min_value
        self.sinogram *= factor
        self.sinogram = np.full(np.shape(self.sinogram), 255) - self.sinogram

        plt.imshow(self.sinogram, cmap=plt.cm.bone)
        # plt.show()

    
    def gen_backprojection(self):
        
        output_image = np.zeros((self.width, self.width), dtype=np.float64)
        filtered_sinogram = self.filter_projection(self.sinogram)
        for iteration, scan in enumerate(self.scans):
            for detector, line in enumerate(scan):
                rr, cc = line
                output_image[rr, cc] += filtered_sinogram[iteration, detector]


        # output_image = (output_image - np.min(output_image)) / (np.max(output_image) - np.min(output_image)) 
        # print(np.min(output_image), np.max(output_image))
        plt.imshow(output_image, cmap=plt.cm.bone)
        plt.show()




sin = Sinogram(cv2.imread('logan.png'), 720, 360, math.pi)
sin.gen_sinogram()
sin.gen_backprojection()
