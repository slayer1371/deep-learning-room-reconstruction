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
During our conversation in class, you mentioned that 2 pairs of images were not enough for good grades, and advised me to collect more images in different lighting conditions. Also, when I asked that what should I be working on for the Semester Project update  - your advice was to solve the image correspondence problem. 
So I bought a tripod for the accuracy of the image, and a multicolored lamp for more variations in the images. I collected 22 pairs of images using this setup, and resized them to the dimensions of 768x1024 pixels, in a seperate "resized-images" folder on github. 

# Part 3 - 
Methods applied for data pre-processing and feature extraction 
- Histogram equalization 
- Gaussian Blurring
- SIFT
- FLANN ( for feature matching )

Justification for deciding to use these Algorithms - 
- Histogram equalization - 
a. Better Contrast: The contrast may be less than ideal while taking pictures in various lighting scenarios. By extending the intensity values, histogram equalization aids in improving the visibility of important features and structures in the picture. This facilitates the identification and matching of unique keypoints between images by feature detection techniques.

b. Consistent Lighting circumstances: By normalizing the pixel intensity distributions across both images, histogram equalization can assist in minimizing lighting disparities if the left and right images in the stereo pair have different lighting circumstances. The accuracy of feature matching between stereo pictures is increased as a result of more consistent characteristics.

c. Improved Feature Extraction: Keypoints are identified using local texture and intensity gradients using feature recognition methods like SIFT. Keypoints may be more difficult to see in photographs with low contrast or bad lighting. The robustness of subsequent matching is increased by histogram equalization, which guarantees the detection of additional keypoints.

- Gaussian Blurring
a. Noise Reduction: Images frequently contain noise, particularly those taken in uncontrolled settings. Feature extraction methods like SIFT may discover incorrect or unstable keypoints as a result of this noise, leading to poor matching and untrustworthy correspondences. By lowering high-frequency noise, Gaussian blurring improves the stability and dependability of the keypoints for matching.

b. Improved Keypoint Detection: SIFT and other feature extraction techniques are sensitive to small details and noise. The feature detection method can concentrate on bigger and more important elements by smoothing the image with a Gaussian blur, which lessens the effect of tiny, unimportant changes in pixel values. More reliable keypoint detection results from this.

- SIFT - Scale-Invariant Feature Transform
a. Scale-Invariant - In stereo vision, differing distances or the camera's zoom level may cause the same object or scene to seem at different scales in the two views. Keypoints can be identified regardless of the scale at which they occur thanks to SIFT's scale invariance. For precise feature matching between photos that might not have been captured at the same zoom level, this is essential.

b. Rotation-Invariant - Different camera orientations can cause images in a stereo pair to spin. Additionally, SIFT is rotation invariant, which means that no matter how much the image is rotated, it will still reliably identify keypoints.

c. Local Feature Descriptors - For every identified keypoint, SIFT calculates a descriptor that characterizes the local image structure surrounding that keypoint. Even if the images have different scales, or rotations, these distinguishing features can be utilized to match keypoints across them.

- FLANN - Fast Library for Approximate Nearest Neighbors
a. Fast Approximate Nearest Neighbor Search - It is designed to match feature descriptors quickly. The process of matching descriptors is computationally costly, particularly when working with big keypoint sets, which is common in stereo vision applications. By employing approximation techniques, FLANN greatly accelerates the procedure, which is adequate for the majority of real-world applications.

b. Handling Large Datasets - When dealing with huge keypoint datasets, as is frequently the case in room reconstruction where several characteristics must be matched across stereo pictures, FLANN is especially helpful. FLANN does approximation searches instead of exact ones, which enables scaled matching of many keypoints.

