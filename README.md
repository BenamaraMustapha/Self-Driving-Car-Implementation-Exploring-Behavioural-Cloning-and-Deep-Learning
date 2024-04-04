<h1>Self-Driving-Car-Implementation-Exploring-Behavioural-Cloning-and-Deep-Learning</h1>
<h2>Introduction</h2>

This article delves into the methodology of self-driving cars utilizing the Udacity Open Sourced Self-Driving Car Simulator. It explores the implementation of self-driving cars and how this can be readily adapted to various scenarios, particularly through Behavioral Cloning. This process entails leveraging Deep Neural Networks, employing Convolution Networks for feature extraction, and conducting continuous regression.

<h2>The Entire Process</h2>

The training involves navigating a car within the simulator's training track, capturing images at each instance of the drive. These images serve as the training dataset, with the label for each image corresponding to the car's steering angle at that instance. Through Convolutional Neural Networks, the system learns to autonomously drive by emulating the behavior of a manual driver. The primary variable the model adjusts is the steering angle, adapting it appropriately based on the encountered situation. Behavioral cloning, showcased in this process, holds significant relevance in real-world self-driving car development.

<h2>Data Collection</h2>

Initiating the behavioral training process begins with downloading the simulator. Initially, we drive the car using keyboard inputs, allowing the Convolutional Neural Network to observe and learn from our controlled vehicle operations. Subsequently, this learned behavior is replicated in autonomous mode through behavioral cloning, where the network mimics our driving actions. The effectiveness of the neural network depends on the accuracy with which it replicates our driving skills. The simulator can be downloaded from the provided link : [Udacity_Self_Driving_Car_Sim](https://github.com/udacity/self-driving-car-sim)


