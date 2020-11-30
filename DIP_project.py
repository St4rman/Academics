import cv2
import numpy as np
from matplotlib import pyplot as plt

# reading image
img = cv2.imread('charlie.jpg')

# printing original image
cv2.imshow('original', img)
cv2.waitKey(0)  # waits for an input before carrying on with rest of code

# converting original image to its components
blue, green, red = cv2.split(img)
img_gs = cv2.imread('charlie.jpg', cv2.IMREAD_GRAYSCALE)

# introducing noise (through salt and pepper) to the greyscale image
def salt_pepper(prob):
      row, col = img_gs.shape

      sp_ratio = 0.5
      output = np.copy(img_gs)

      num_salt = np.ceil(prob * img_gs.size * sp_ratio)
      coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in img_gs.shape]
      output[coords] = 1

      num_pepper = np.ceil(prob * img_gs.size * (1. - sp_ratio))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in img_gs.shape]
      output[coords] = 0
      cv2.imshow('noise',output)
      cv2.waitKey(0)

      return output

#saving noisy image
sp_noise_05 = salt_pepper(2)
cv2.imwrite('noisy_image.jpg', sp_noise_05)

#reducing noise
color_image = cv2.imread('noisy_image.jpg')
gray_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
sliding_window_size_x = 5
sliding_window_size_y = 5
mean_filter_kernel = np.ones((sliding_window_size_x,sliding_window_size_y),np.float32)/(sliding_window_size_x*sliding_window_size_y)

#2d filtering image
filtered_image = cv2.filter2D(gray_img,-1,mean_filter_kernel)
#bilateral filtering the image
bilateral_image = cv2.bilateralFilter(gray_img,9,75,75)
#gaussian filtering the image
gaus_image = cv2.GaussianBlur(gray_img, (3,3), 0)
#blur filter
blur_image = cv2.blur(gray_img, (3,3))
#median blurring
median_image = cv2.medianBlur(gray_img, 3)

#printing results
plt.subplot(2, 3, 1),plt.imshow(gray_img),plt.title('Original noisy image')
plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(filtered_image),plt.title('2D Filtered Image')
plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(bilateral_image),plt.title('bilateral filtered Image')
plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(gaus_image),plt.title('Gaussian blurred Image')
plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(blur_image),plt.title('Blurred Image')
plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(filtered_image),plt.title('Median blurred Image')
plt.xticks([]), plt.yticks([])
plt.show()