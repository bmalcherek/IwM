import cv2
from skimage.draw import line as bresenham
import math
import numpy as np

class Sinogram:

    def __init__(self, image, *, num_detectors, num_steps, theta, filter=True, gaussian=True):
        self.image = image
        self.num_detectors = num_detectors
        self.num_steps = num_steps
        self.theta = theta
        self.filter = filter
        self.gaussian = gaussian

        self.width = int(image.shape[0])
        self.offset = int(self.width / 2)
        self.radius = int(self.width / 2) - 1
        
        self.sinogram = np.zeros((self.num_steps, self.num_detectors), dtype=np.uint8)
        self.backprojections = []
        self._generate()

    def _get_coords(self, alpha):
        x = int(math.cos(alpha) * self.radius) + self.offset
        y = -int(math.sin(alpha) * self.radius) + self.offset
        return x, y

    
    def _generate_filter(self, length):
        filter = [(-4/(np.pi**2))/(i**2) if i%2 != 0 else 0 for i in range(-int(length/2),int((length/2))+1)]
        filter[int(length/2)] = 1  
        return filter

    def _normalize_image(self, img):
        img = np.clip(img, 0, None)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return np.array(img, dtype=np.uint8)

    def _generate(self):
        backprojection = np.zeros((self.width, self.width))
        sinogram = np.zeros((self.num_steps, self.num_detectors))

        filter = self._generate_filter(self.num_detectors)
        angular_step = math.pi * 2 / self.num_steps

        for step in range(self.num_steps):
            alpha = step * angular_step

            step_lines = []
            sinogram_row = np.zeros((self.num_detectors), dtype=np.int32)
            for detector in range(self.num_detectors):
                detector_angle = alpha + math.pi - self.theta / 2 + detector * (self.theta / (self.num_detectors - 1))
                detector_x, detector_y = self._get_coords(detector_angle)

                emitter_angle = alpha + self.theta / 2 - detector * (self.theta / (self.num_detectors - 1))
                emitter_x, emitter_y = self._get_coords(emitter_angle)

                rr, cc = bresenham(emitter_y, emitter_x, detector_y, detector_x)
                sinogram_row[detector] = np.array(self.image[rr, cc]).sum()
                step_lines.append((rr, cc))
            
            sinogram[step] = sinogram_row

            if self.filter:
                sinogram_row = np.array(np.convolve(sinogram_row, filter, 'same'), dtype=np.int32)
            

            for detector in range(self.num_detectors):
                rr, cc = step_lines[detector]
                backprojection[rr, cc] += sinogram_row[detector]

            if(self.gaussian):
                self.backprojections.append(self._normalize_image(cv2.GaussianBlur(backprojection,(3,3),3)))
            else:
                self.backprojections.append(self._normalize_image(backprojection))

        self.sinogram = cv2.normalize(sinogram, None, 0, 255, cv2.NORM_MINMAX)


    def get_backprojection_frames(self):
        return self.backprojections

    def get_backprojection(self):
        return self.backprojections[-1]

    def get_sinogram(self):
        return self.sinogram

    def get_rmse(self):
        return np.sqrt(np.mean((self.image-self.backprojections[-1])**2))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img = cv2.imread('images/shepp.jpg', 0)
    sin = Sinogram(image=img, num_steps=180, num_detectors=180, theta=math.pi)
    print(sin.get_rmse())
    b = sin.get_backprojection()
    plt.imshow(b, cmap=plt.cm.bone)
    plt.show()