# Semester Project - Room Reconstruction using Deep Learning
The goal of the project is to develop a deep learning-based solution for reconstructing a room's 3D structure from a set of 2D images. The system should be able to identify walls, floors, ceilings, and furniture, and accurately render the geometry of the room in a 3D space.
The proposed solution relies on deep learning architectures, particularly convolutional neural networks (CNNs) and transformer-based models, to process the input images, extract meaningful spatial information, and estimate the 3D structure.
I need to construct a 3D model from a set of 2D images. The features extracted from the 2D images would be something like edges of the room, texture and maybe objects kept in the room. Acc. to ChatGPT, there exists pre-trained models, such as ResNet or VGG, which can be modeled to recognize architectural features and room components.

# Part 1 - Data Acquisition
I will be clicking photos of various room at different angles through smartphone. These will serve as my training dataset. When the model is trained, I will test the model with some random images of the rooms I have trained the model on,(different from the photos I clicked originally while training the model), and see if it reconstructs the room correctly.

# Part 2 - Feature Extraction
Thinking upon this :)
