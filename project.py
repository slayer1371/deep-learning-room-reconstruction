from PIL import Image
from google.colab import drive
import math
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import time

class ImageProcessor:
    """
    A class for processing images using the Pillow and Numpy libraries.
    """

    def __init__(self, image_path):
        """
        Initialize the ImageProcessor with an image file.

        Args:
            image_path (str): Path to the input image.
        """
        self.image = Image.open(image_path).convert('L')

    def fetch_info(self):
        """
        Retrieve the dimensions and mode of the image.

        Returns:
            tuple: A tuple containing the image's size and mode.
        """
        dimensions = self.image.size
        mode = self.image.mode
        print(f"Dimensions: {dimensions}")
        print(f"Mode: {mode}")
        return dimensions, mode

    def to_array(self):
        """
        Convert the image to a Numpy array.

        Returns:
            numpy.ndarray: Array representation of the image.
        """
        return np.array(self.image, dtype=np.int64)

    def resize(self, new_width, new_height):
        """
        Resize the image to the specified dimensions.

        Args:
            new_width (int): Desired width.
            new_height (int): Desired height.
        """
        self.image = self.image.resize((new_width, new_height))

    def extract_slice(self, start_x, start_y, region_size):
        """
        Extract a square region from the image array.

        Args:
            start_x (int): Starting x-coordinate of the region.
            start_y (int): Starting y-coordinate of the region.
            region_size (int): Size of the square region.

        Returns:
            numpy.ndarray: Extracted region as a Numpy array.
        """
        image_array = self.to_array()
        return image_array[start_x:start_x + region_size, start_y:start_y + region_size]

    def display(self):
        """
        Display the image.
        """
        self.image.show()

def preprocess_image(image_path, target_height, target_width):
    """
    Preprocess an image by resizing it to specified dimensions and converting it to a grayscale numpy array.

    Args:
        image_path (str): Path to the input image file.
        target_height (int): Desired height of the resized image.
        target_width (int): Desired width of the resized image.

    Returns:
        numpy.ndarray: Grayscale image as a numpy array with the specified dimensions.
    """
    processor = ImageProcessor(image_path)  # Instantiate the ImageProcessor class
    processor.resize(target_width, target_height)  # Resize the image
    image_array = processor.to_array()  # Convert the image to a numpy array

    print(f"Image at {image_path} has been successfully preprocessed.")
    print("Details:")
    processor.fetch_info()  # Print image information

    return image_array


def compute_disparity(left_image_array, right_image_array, window_size, max_disparity):
    """
    Compute a disparity map using two image arrays with the Sum of Absolute Differences (SAD) method.

    Args:
        left_image_array (numpy.ndarray): Grayscale array of the left image.
        right_image_array (numpy.ndarray): Grayscale array of the right image.
        window_size (int): Size of the square window for matching regions.
        max_disparity (int): Maximum search range for matching in the negative direction.

    Returns:
        numpy.ndarray: Disparity map as a numpy array with adjusted dimensions based on window size.
    """
    start_time = time.time()
    disparity_map = []

    rows, cols = left_image_array.shape

    # Iterate over the rows, leaving space for the window size
    for row in range(rows - window_size):
        if row % 10 == 0:
            print(f"Processing row {row} for disparity...")

        row_disparity = []

        # Iterate over the columns, leaving space for the window size
        for col_left in range(cols - window_size):
            # Extract the matching window from the left image
            left_window = left_image_array[row:row + window_size, col_left:col_left + window_size].flatten()

            # Determine the range of column indices to search in the right image
            search_start = max(0, col_left - max_disparity)
            search_end = col_left + 1

            sad_values = []

            # Calculate the SAD for each candidate window in the search range
            for col_right in range(search_end - 1, search_start - 1, -1):
                right_window = right_image_array[row:row + window_size, col_right:col_right + window_size].flatten()
                sad = np.sum(np.abs(left_window - right_window))
                sad_values.append(sad)

            # Find the disparity as the column shift with the minimum SAD
            disparity = np.argmin(sad_values)
            row_disparity.append(disparity)

        disparity_map.append(row_disparity)

    disparity_map = np.array(disparity_map)
    end_time = time.time()

    print("Disparity computation completed.")
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")

    return disparity_map

def refine_disparity_map(disparity_map):
    """
    Post-process a disparity map to refine its values using mean filtering, mode adjustment, and thresholding.

    Args:
        disparity_map (numpy.ndarray): Input disparity map.

    Returns:
        numpy.ndarray: Refined disparity map.
    """
    refined_map = np.copy(disparity_map)
    rows, cols = refined_map.shape

    for x in range(cols):
        for y in range(rows):
            # Mean filtering
            y_start, y_end = max(0, y - 7), min(rows, y + 8)
            x_start, x_end = max(0, x - 7), min(cols, x + 8)
            local_mean = np.mean(refined_map[y_start:y_end, x_start:x_end])

            if abs(refined_map[y, x] - local_mean) > 5:
                refined_map[y, x] = local_mean

            # Mode adjustment for larger regions
            if 12 < x < cols - 12 and 12 < y < rows - 12:
                if refined_map[y, x] > 25:
                    region = refined_map[y - 12:y + 13, x - 12:x + 13].flatten()
                    mode_result = stats.mode(region, keepdims=True)
                    refined_map[y, x] = mode_result.mode[0] if isinstance(mode_result.mode, np.ndarray) else mode_result.mode

            # Thresholding
            if refined_map[y, x] > 30:
                refined_map[y, x] = 25

    print(f"Post-processing completed for disparity map with shape {disparity_map.shape}.")
    return refined_map


def generate_point_cloud(image_path, disparity_map, output_file):
    """
    Generate a point cloud file (CSV format) from a raw image and its disparity map.

    Args:
        image_path (str): Path to the raw image.
        disparity_map (numpy.ndarray): Disparity map as a numpy array.
        output_file (str): Path to save the generated point cloud file.

    Returns:
        pandas.DataFrame: Dataframe containing the point cloud data with x, y, z, r, g, b columns.
    """
    # Match image dimensions with disparity map
    height, width = disparity_map.shape
    image = Image.open(image_path).resize((width, height))
    rgb_array = np.array(image)

    # Prepare the point cloud data
    point_cloud = []
    scale_factor = 6  # Scale factor for z-values

    for x in range(width):
        for y in range(height):
            z = disparity_map[y, x] * scale_factor
            r, g, b = rgb_array[y, x]
            point_cloud.append([x, y, z, r, g, b])

    # Convert to DataFrame and save as CSV
    columns = ['x', 'y', 'z', 'r', 'g', 'b']
    point_cloud_df = pd.DataFrame(point_cloud, columns=columns)
    point_cloud_df.to_csv(output_file, index=False)

    print(f"Point cloud successfully saved to {output_file}.")
    return point_cloud_df

def main():
    # Define input paths and parameters
    image_path_left = '/content/data/left3.jpg'  #change image pair here
    image_path_right = '/content/data/right3.jpg' #change image pair here

    target_height = 250
    target_width = 357

    window_size = 11
    max_disparity = 44

    output_txt_path = f"/content/data/output/point_cloud_{target_height}.txt"
    output_image_path = f"/content/data/output/{target_height}.png"

    # Pre-process images
    array_left = preprocess_image(image_path=image_path_left,
                                  target_height=target_height,
                                  target_width=target_width)
    array_right = preprocess_image(image_path=image_path_right,
                                   target_height=target_height,
                                   target_width=target_width)

    # Compute disparity map
    disparity_map = compute_disparity(left_image_array=array_left,
                                      right_image_array=array_right,
                                      window_size=window_size,
                                      max_disparity=max_disparity)

    # Post-process the disparity map
    refined_disparity_map = disparity_map
    for _ in range(5):  # Apply refinement multiple times
        refined_disparity_map = refine_disparity_map(refined_disparity_map)

    # Generate the point cloud
    point_cloud_df = generate_point_cloud(image_path=image_path_left,
                                          disparity_map=refined_disparity_map,
                                          output_file=output_txt_path)

    # Save the processed disparity map as an image
    plt.imshow(refined_disparity_map, cmap='gray')
    plt.colorbar()
    plt.savefig(output_image_path)

    print(f"Disparity map image saved at {output_image_path}")
    print(f"Point cloud data saved at {output_txt_path}")


if __name__ == "__main__":
    main()


# from google.colab import drive
# import math
# import numpy as np
# import cv2
# from google.colab.patches import cv2_imshow
# import matplotlib.pyplot as plt

# left_image = cv2.imread('/content/drive/MyDrive/resized-images/left9.jpg')
# right_image = cv2.imread('/content/drive/MyDrive/resized-images/right9.jpg')

# left_ycrcb = cv2.cvtColor(left_image, cv2.COLOR_BGR2YCrCb)
# right_ycrcb = cv2.cvtColor(right_image, cv2.COLOR_BGR2YCrCb)

# # Apply histogram equalization on the Y (luminance) channel for both images
# left_ycrcb[:, :, 0] = cv2.equalizeHist(left_ycrcb[:, :, 0])
# right_ycrcb[:, :, 0] = cv2.equalizeHist(right_ycrcb[:, :, 0])

# # Convert back to BGR color space
# equalized_left_image = cv2.cvtColor(left_ycrcb, cv2.COLOR_YCrCb2BGR)
# equalized_right_image = cv2.cvtColor(right_ycrcb, cv2.COLOR_YCrCb2BGR)

# # Displaying the original and equalized images for comparison
# plt.figure(figsize=(12, 6))

# # Original Left and Right Images
# plt.subplot(2, 2, 1)
# plt.imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
# plt.title('Original Left Image')
# plt.axis('off')

# plt.subplot(2, 2, 2)
# plt.imshow(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
# plt.title('Original Right Image')
# plt.axis('off')

# # Equalized Left and Right Images
# plt.subplot(2, 2, 3)
# plt.imshow(cv2.cvtColor(equalized_left_image, cv2.COLOR_BGR2RGB))
# plt.title('Equalized Left Image')
# plt.axis('off')

# plt.subplot(2, 2, 4)
# plt.imshow(cv2.cvtColor(equalized_right_image, cv2.COLOR_BGR2RGB))
# plt.title('Equalized Right Image')
# plt.axis('off')

# plt.tight_layout()
# plt.show()

# # Apply Gaussian blur
# left_image = cv2.GaussianBlur(equalized_left_image, (5, 5), 0)
# right_image = cv2.GaussianBlur(equalized_right_image, (5, 5), 0)

# # Initialize SIFT detector
# sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.04, edgeThreshold=10)

# gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
# gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# keypoints_left, descriptors_left = sift.detectAndCompute(gray_left, None)
# keypoints_right, descriptors_right = sift.detectAndCompute(gray_right, None)

# index_params = dict(algorithm=1, trees=10)
# search_params = dict(checks=50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)

# matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)

# # Apply ratio test to filter good matches
# good_matches = []
# for m, n in matches:
#     if m.distance < 0.71 * n.distance: # value towards 0.7 indicated more accuracy.
#         good_matches.append(m)

# good_matches.sort(key=lambda x: x.distance)

# top_n_matches = good_matches[:50]

# # Draw matches
# match_image = cv2.drawMatches(gray_left, keypoints_left, gray_right, keypoints_right, top_n_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# cv2_imshow(match_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
