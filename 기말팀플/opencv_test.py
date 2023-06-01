import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import numpy as np

img = imread('img_yn4.png')
plt.figure(figsize=(15,12))
plt.imshow(img)
plt.show()
