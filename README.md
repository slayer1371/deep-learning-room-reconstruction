# Semester Project - 3D Room Reconstruction using Stereo Vision
The goal of the project is to develop a classical stereo vision solution for reconstructing a room's 3D structure from a pair of 2D images. The system should be able to identify walls, floors, ceilings, and furniture, and accurately render the geometry of the room in a 3D space. <br />
The proposed solution relies on stereo vision method, to process the input images, extract meaningful spatial information, and estimate the 3D structure. <br />
The features extracted from the 2D images would may be something like edges of the room, texture and maybe objects kept in the room. Acc to ChatGPT, there exists pre-trained models, such as ResNet or VGG, which can be modeled to recognize architectural features and room components.

# Part 1 - Conceptual Design
The conceptual design of 3D room reconstruction using stereo vision involves capturing stereo image pairs, calculating depth maps, generating 3D point clouds, and optionally creating a mesh for a solid model. With careful calibration and image processing, this approach can produce accurate 3D representations of indoor environments.

# Part 2 - Data Acquisition
Traditional Stereo Vision Approach : In a classical stereo vision project, training data in the ML sense may not be necessary, but I will need a dataset for testing and evaluation. 
Data - Stereo images of different rooms as my "testing" dataset to evaluate the performance of your 3D reconstruction algorithm.
For stereo vision, this will include left and right images from multiple rooms.
I collected 2 pairs of pictures (left_image, right_image) of my living room and my kitchen(with the phone on the same plane, just dislocated horizontally by a short distance). 

# UPDATE
During the conversation we had in class, you mentioned that 2 pairs of images were not enough for good grades, and advised me to collect more images in different lighting conditions. Also, when I asked that what should I be working on for the Semester Project update  - your advice was to solve the image correspondence problem. 
So I bought a tripod for the accuracy of the image, and a multicolored lamp for more variations in the images. I collected 22 pairs of images using this setup, and resized them to the dimensions of 768x1024 pixels, in a seperate "resized-images" folder on github. 

# Part 3 - 
Methods applied for data pre-processing and feature extraction 
- Histogram equalization 
- Gaussian Blurring
- SIFT
- FLANN ( for feature matching )

Justification for deciding to use these Algorithms - 
- Histogram equalization - 
    - Better Contrast: The contrast may be less than ideal while taking pictures in various lighting scenarios. By extending the intensity values, histogram equalization aids in improving the visibility of important features and structures in the picture. This facilitates the identification and matching of unique keypoints between images by feature detection techniques.

    - Consistent Lighting circumstances: By normalizing the pixel intensity distributions across both images, histogram equalization can assist in minimizing lighting disparities if the left and right images in the stereo pair have different lighting circumstances. The accuracy of feature matching between stereo pictures is increased as a result of more consistent characteristics.

  - Improved Feature Extraction: Keypoints are identified using local texture and intensity gradients using feature recognition methods like SIFT. Keypoints may be more difficult to see in photographs with low contrast or bad lighting. The robustness of subsequent matching is increased by histogram equalization, which guarantees the detection of additional keypoints.

- Gaussian Blurring
    - Noise Reduction: Images frequently contain noise, particularly those taken in uncontrolled settings. Feature extraction methods like SIFT may discover incorrect or unstable keypoints as a result of this noise, leading to poor matching and untrustworthy correspondences. By lowering high-frequency noise, Gaussian blurring improves the stability and dependability of the keypoints for matching.

    - Improved Keypoint Detection: SIFT and other feature extraction techniques are sensitive to small details and noise. The feature detection method can concentrate on bigger and more important elements by smoothing the image with a Gaussian blur, which lessens the effect of tiny, unimportant changes in pixel values. More reliable keypoint detection results from this.

- SIFT - Scale-Invariant Feature Transform
    - Scale-Invariant - In stereo vision, differing distances or the camera's zoom level may cause the same object or scene to seem at different scales in the two views. Keypoints can be identified regardless of the scale at which they occur thanks to SIFT's scale invariance. For precise feature matching between photos that might not have been captured at the same zoom level, this is essential.

    - Rotation-Invariant - Different camera orientations can cause images in a stereo pair to spin. Additionally, SIFT is rotation invariant, which means that no matter how much the image is rotated, it will still reliably identify keypoints.

    - Local Feature Descriptors - For every identified keypoint, SIFT calculates a descriptor that characterizes the local image structure surrounding that keypoint. Even if the images have different scales, or rotations, these distinguishing features can be utilized to match keypoints across them.

    - Image correspondence problem - The image correspondence problem is about finding corresponding points or regions in two or more images that depict the same scene but possibly from different perspectives, scales, or lighting conditions. 

- FLANN - Fast Library for Approximate Nearest Neighbors
    - Fast Approximate Nearest Neighbor Search - It is designed to match feature descriptors quickly. The process of matching descriptors is computationally costly, particularly when working with big keypoint sets, which is common in stereo vision applications. By employing approximation techniques, FLANN greatly accelerates the procedure, which is adequate for the majority of real-world applications.

    - Handling Large Datasets - When dealing with huge keypoint datasets, as is frequently the case in room reconstruction where several characteristics must be matched across stereo pictures, FLANN is especially helpful. FLANN does approximation searches instead of exact ones, which enables scaled matching of many keypoints.

<img src="https://github.com/slayer1371/deep-learning-room-reconstruction/blob/main/illustrations/illustrat1.png?raw=true" alt="Illustration 1" width="300" /><img src = "https://github.com/slayer1371/deep-learning-room-reconstruction/blob/main/illustrations/illustration1.png?raw=true" width = "500" />

<img src="https://github.com/slayer1371/deep-learning-room-reconstruction/blob/main/illustrations/illustrat3.png?raw=true" alt="Illustration 3" width="300" /><img src = "https://github.com/slayer1371/deep-learning-room-reconstruction/blob/main/illustrations/illustration3.png?raw=true" width = "500" />

It's not 100% optimal right now, with some matches going the wrong way. In the images, where there are reflections caused due to the glass door, the algorithm is making some wrong feature matching decisions, and I'll be working on correcting that.

Colab link, for your convenience, so that you don't have to clone my project folder into your setup. Also, colab allows cv2_imshow, and python allows cv2.imshow, so that may be different if the code is executed locally.
https://colab.research.google.com/drive/1ydoMhAZBp_U2fRK1HCW5UrhA_qAE9g0x?usp=sharing

Instructions to run on colab = Just take any left-right image pairs from the resized-images folder with the same number, for eg, left 10 and right10, or left20 and right20.

# Part 4 - 

Update - Pivoted to Block matching rather than feature matching.
Reason - 
1. Used checkerboard pattern for camera calibaration and got values of K and camera intrinsic parameters.
2. Tried to rectify left and right images using the techniques of stereo rectification, but was largely unsuccessful. Tried capturing images from different mobile device, but rectified left and rectified right never turned out correct.
3. When 4 days passed, and the problem was not solved, switched to block matching using sliding window technique.

Process - 

1. Block Matching for Disparity Calculation
    - Stereo Matching: Works by comparing small windows of pixels (blocks) from the left and right images and calculating the difference between them.
    - Efficiency: This technique is computationally simpler compared to feature-based matching (like SIFT), and works well when matching corresponding regions in relatively uniform or textured images.
    - Fixed Window Size: The sliding window method (with a fixed window size) ensures that the local pixel neighborhoods are compared in both the left and right images, allowing the algorithm to find the pixel shifts (disparities) that correspond to differences in depth.
    
2. Sum of Absolute Differences (SAD) for Matching  
    - Error Metric: The Sum of Absolute Differences (SAD) is used as a matching criterion to measure the similarity (or dissimilarity) between two image windows. It computes the absolute difference between pixel intensities in the left and right images, and the result is summed to quantify how closely the windows match.
    - Efficiency: SAD is simple to compute and works well in block matching for dense stereo images. SAD strikes a good balance between performance and accuracy in this scenario.

3. Post-Processing with Filtering (Mean and Mode)
    - Noise Reduction: The disparity map generated initially may contain noise or inconsistencies due to mismatches in the stereo matching process. Post-processing smooths out these irregularities.
    - Mean Filtering: The mean filter smooths the disparity map by averaging disparity values in a local neighborhood. This reduces small errors and helps refine the depth estimation.
    - Mode Filtering: The mode filter assigns the most frequent disparity value in a neighborhood of pixels. This is useful for removing outliers and ensuring that the disparity map reflects more consistent depth values, especially in areas with large differences in depth.

4. Thresholding of Disparity Values
    - Outlier Removal: Thresholding ensures that extreme disparity values (likely resulting from mismatches or noise) are removed or reduced to more reasonable levels. For example, disparities that exceed a certain threshold are likely erroneous and are clipped to a lower value.
    - Improved Depth Accuracy: By eliminating extreme disparity values, the final depth map becomes more accurate, especially in regions where matching is difficult due to occlusions, textureless surfaces, or noise.

5. 3D Point Cloud Generation
    - Depth Mapping: The disparity values are converted into depth values (z-coordinates) based on the relationship between disparity and depth. The equation typically used for this is:
       <br /> 𝑧 = 𝑓⋅𝐵 / 𝑑 
        - where,
        - f is the focal length, 
        - B is the baseline (distance between cameras), and 
        - d is the disparity. 
        - This allows the creation of a 3D representation of the scene.
    - RGB Mapping: Each point in the point cloud is enriched with color information from the corresponding pixel in the original image. This makes the final point cloud more meaningful for visualization, as it includes both 3D coordinates and color data.

6. Visualization
    - Drag and drop the txt point_cloud file generated into cloud compare software, and check the visualization.
  
EXTRA INFORMATION - 
Cloud Compare Software website - https://cloudcompare-org.danielgm.net/release/
Use image pairs from update-images folder, not from the resized-images folder.
When running the program, create 3 folders in your setup. 
1. data
    1.1. output folder
    and the image pair

This is because the program is configured such that the pointcloud file will be created and stored in output folder for ease of accessibility.

How to change image pairs - Just change filename of the image you want to use in the program in the main() function.

IMPROVEMENTS - 
In some image pairs, the point cloud generated is 
1. inverted and/or 
2. a mirror image. 
Will try to understand the reson behind this, and solve it.

There was a concept that Prof Adam taught in the class, that keypoint matching can also be done by creating a number of small cnns' and using them instead of sift or sliding window technique. I will try to implement that, if possible for part 5.

Updated colab link - https://colab.research.google.com/drive/1ydoMhAZBp_U2fRK1HCW5UrhA_qAE9g0x?usp=sharing

Part 5 - 

Description of the test database you collected or downloaded: I created a test database by capturing image pairs of 2 different rooms with different lighting conditions. 22 image pairs.

This custom project for 3D room reconstruction using stereo vision doesn't directly involve classification metrics because it's not a supervised learning task requiring labeled data for training and testing. Instead, it falls under the domain of geometry-based computer vision, where the focus is on accurate estimation of spatial information rather than classification.

No Training/Testing Data:

In this project, I didn't train a model to classify objects or make predictions. Instead, I relied on mathematical algorithms (like triangulation and disparity computation) to reconstruct 3D scenes.
The inputs are stereo images, and the outputs are point clouds or depth maps, which are evaluated geometrically rather than using classification metrics.

Output Type:

The final output is a 3D point cloud or 3D model, not a classification label or a decision boundary. The evaluation is based on geometric accuracy (e.g., how closely the reconstructed 3D model matches the real-world geometry).

Some illustrations - 
<br />
<img width="638" alt="Screenshot 2024-12-14 at 11 32 55 AM" src="https://github.com/user-attachments/assets/de3960f1-cc7c-4e8d-87dd-19e54f57ea56" />
<img width="657" alt="Screenshot 2024-12-14 at 11 33 28 AM" src="https://github.com/user-attachments/assets/e26f07e4-4254-4ca6-a2da-a5c368fa30cb" />

What went wrong - 
Image cloud generated was always a mirror image through origin, but after compiling function mirror_image_origin(), which takes the initial image and passes a mirror image into the point_clooud_generator, so that the final result is the desired orientation, it works slightly better in the sense that the compiled image cloud is now a mirror image through y-axis, but is not inverted.

Udates to the code - function mirror_image_origin()

INSTRUCTIONS ON RUNNING THE CODE - 

1. CREATE A FOLDER NAMED 'data' IN YOUR CONTENT FOLDER IN COLAB.
2. INSIDE THIS FOLDER, CREATE ANOTHER FOLDER NAMED 'output'.
3. UPLOAD THE IMAGE PAIR (FROM UPDATED-IMAGES) YOU WANT TO TEST INTO THE 'DATA' FOLDER CREATED ABOVE, NOT INTO THE OUTPUT FOLDER.
4. RUN THE CELLS.
5. AFTER SUCCESSFUL EXECUTION, DOWNLOAD THE POINT_CLOUD.TXT FILE GENERATED IN THE 'OUTPUT' FOLDER.
6. DROP THIS FILE IN CLOUDCOMPARE SOFTWARE, AND VOILA, YOU'LL HAVE A CRUDE 3D RECONSTRUCTION OF THE IMAGES. INCREASE THE DEFAULT POINT WIDTH IN THE SOFTWARE FOR BETTER VISUALIZATION.

Cloud Compare Software website - https://cloudcompare-org.danielgm.net/release/

SHORT PRESENTATION - https://docs.google.com/presentation/d/1HMuM7xL8upuATVgcbVRNbgbOg6S_3rDOjN5eOp9E3kU/edit?usp=sharing
