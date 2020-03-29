from sinogram import Sinogram
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import math
import matplotlib.pyplot as plt

img = cv2.imread('images/shepp.jpg', 0)
# sin = Sinogram(image=img, num_steps=180, num_detectors=180, theta=math.radians(270), gaussian=True, filter=True)
# ax1 = plt.subplot('131')
# ax1.imshow(img, cmap=plt.cm.bone)
# ax2 = plt.subplot('132')
# ax2.imshow(sin.get_sinogram(), cmap=plt.cm.bone)
# ax2 = plt.subplot('133')
# ax2.imshow(sin.get_backprojection(), cmap=plt.cm.bone)
# cv2.imwrite('output/out.jpg', sin.get_backprojection())

# plt.show()

for num_detectors in range(90, 721, 90):
    sin = Sinogram(image=img, num_steps=180, num_detectors=num_detectors, theta=math.pi, gaussian=True)
    print(f'{num_detectors};{sin.get_rmse()}')

print('---------------')
for num_steps in range(90, 721, 90):
    sin = Sinogram(image=img, num_steps=num_steps, num_detectors=180, theta=math.pi, gaussian=True)
    print(f'{num_steps};{sin.get_rmse()}')

print('---------------')
for theta in range(45, 271, 45):
    sin = Sinogram(image=img, num_steps=180, num_detectors=180, theta=math.radians(theta), gaussian=True)
    print(f'{theta};{sin.get_rmse()}')


img = cv2.imread('images/saddle_large.jpg', 0)
sin = Sinogram(image=img, num_steps=360, num_detectors=360, theta=math.radians(270), filter=False ,gaussian=False)
print(f'{sin.get_rmse()}')
sin = Sinogram(image=img, num_steps=360, num_detectors=360, theta=math.radians(270), filter=True ,gaussian=True)
print(f'{sin.get_rmse()}')

img = cv2.imread('images/saddle_large.jpg', 0)
sin = Sinogram(image=img, num_steps=360, num_detectors=360, theta=math.radians(270), filter=False ,gaussian=False)
cv2.imwrite('output/saddle_large_nofilter.jpg', sin.get_backprojection())
print(sin.get_rmse())
sin = Sinogram(image=img, num_steps=360, num_detectors=360, theta=math.radians(270), filter=True ,gaussian=True)
cv2.imwrite('output/saddle_large_filter.jpg', sin.get_backprojection())
print(sin.get_rmse())
