# Semester Project - 3D Room Reconstruction using Stereo Vision
The goal of the project is to develop a classical stereo vision solution for reconstructing a room's 3D structure from a pair of 2D images. The system should be able to identify walls, floors, ceilings, and furniture, and accurately render the geometry of the room in a 3D space. <br />
The proposed solution relies on stereo vision method, to process the input images, extract meaningful spatial information, and estimate the 3D structure. <br />
The features extracted from the 2D images would may be something like edges of the room, texture and maybe objects kept in the room. Acc to ChatGPT, there exists pre-trained models, such as ResNet or VGG, which can be modeled to recognize architectural features and room components.

# Part 1 - Conceptual Design
The conceptual design of 3D room reconstruction using stereo vision involves capturing stereo image pairs, calculating depth maps, generating 3D point clouds, and optionally creating a mesh for a solid model. With careful calibration and image processing, this approach can produce accurate 3D representations of indoor environments.

# Part 2 - Data Acquisition
Traditional Stereo Vision Approach : In a classical stereo vision project, training data in the ML sense may not be necessary, but you will need a dataset for testing and evaluation. 
Stereo images of different rooms as your "testing" dataset to evaluate the performance of your 3D reconstruction algorithm.
For stereo vision, this will include left and right images from multiple rooms.
I collected pairs of pictures (with the phone on the same plane, just dislocated horizontally by a short distance).
