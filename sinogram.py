import cv2
from skimage.draw import line as bresenham
import math
import matplotlib.pyplot as plt

class Sinogram:
    def __init__(self, img, num_detectors, steps, theta):
        self.img = img
        self.num_detectors = num_detectors
        self.steps = steps
        self.theta = theta

        self.width = img.shape[0]
        self.radius = self.width / 2 - 1

        self.offset = int(self.width / 2)

    
    def get_coords(self, alpha):
        x = int(math.cos(alpha) * self.radius) + self.offset
        y = int(math.sin(alpha) * self.radius) + self.offset
        return x, y

    
    def gen_lines(self):
        self.lines = list()

        plt.imshow(self.img)

        angular_step = math.pi * 2 / self.steps
        for step in range(20, 21):
            alpha = step * angular_step
            emitter_x, emitter_y = self.get_coords(alpha)

            step_lines = list()
            for detector in range(self.num_detectors):
                detector_angle = alpha + math.pi - self.theta / 2 + detector * (self.theta / (self.num_detectors - 1))
                detector_x, detector_y = self.get_coords(detector_angle)
                
                plt.plot([emitter_x, detector_x], [emitter_y, detector_y], 'g-')

                rr, cc = bresenham(emitter_y, emitter_x, detector_y, detector_x)
                step_lines.append((rr, cc))
            

            self.lines.append(step_lines)

        plt.show()



sin = Sinogram(cv2.imread('logan.jpg'), 180, 180, math.pi)
sin.gen_lines()
