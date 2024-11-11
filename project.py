from google.colab import drive
import math
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

left_image = cv2.imread('/content/drive/MyDrive/resized-images/left9.jpg')
right_image = cv2.imread('/content/drive/MyDrive/resized-images/right9.jpg')

left_ycrcb = cv2.cvtColor(left_image, cv2.COLOR_BGR2YCrCb)
right_ycrcb = cv2.cvtColor(right_image, cv2.COLOR_BGR2YCrCb)

# Apply histogram equalization on the Y (luminance) channel for both images
left_ycrcb[:, :, 0] = cv2.equalizeHist(left_ycrcb[:, :, 0])
right_ycrcb[:, :, 0] = cv2.equalizeHist(right_ycrcb[:, :, 0])

# Convert back to BGR color space
equalized_left_image = cv2.cvtColor(left_ycrcb, cv2.COLOR_YCrCb2BGR)
equalized_right_image = cv2.cvtColor(right_ycrcb, cv2.COLOR_YCrCb2BGR)

# Displaying the original and equalized images for comparison
plt.figure(figsize=(12, 6))

# Original Left and Right Images
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
plt.title('Original Left Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
plt.title('Original Right Image')
plt.axis('off')

# Equalized Left and Right Images
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(equalized_left_image, cv2.COLOR_BGR2RGB))
plt.title('Equalized Left Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(equalized_right_image, cv2.COLOR_BGR2RGB))
plt.title('Equalized Right Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# Apply Gaussian blur
left_image = cv2.GaussianBlur(equalized_left_image, (5, 5), 0)
right_image = cv2.GaussianBlur(equalized_right_image, (5, 5), 0)

# Initialize SIFT detector
sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.04, edgeThreshold=10)

gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

keypoints_left, descriptors_left = sift.detectAndCompute(gray_left, None)
keypoints_right, descriptors_right = sift.detectAndCompute(gray_right, None)

index_params = dict(algorithm=1, trees=10)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)

# Apply ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.71 * n.distance: # value towards 0.7 indicated more accuracy.
        good_matches.append(m)

good_matches.sort(key=lambda x: x.distance)

top_n_matches = good_matches[:50]

# Draw matches
match_image = cv2.drawMatches(gray_left, keypoints_left, gray_right, keypoints_right, top_n_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2_imshow(match_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
