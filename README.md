<h1>Autonomous Car Implementation-Exploring Behavioural Cloning and Deep Learning</h1>
<h2>Introduction</h2>

This article delves into the methodology of self-driving cars utilizing the Udacity Open Sourced Self-Driving Car Simulator. It explores the implementation of self-driving cars and how this can be readily adapted to various scenarios, particularly through Behavioral Cloning. This process entails leveraging Deep Neural Networks, employing Convolution Networks for feature extraction, and conducting continuous regression.

<h2>The Entire Process</h2>

The training involves navigating a car within the simulator's training track, capturing images at each instance of the drive. These images serve as the training dataset, with the label for each image corresponding to the car's steering angle at that instance. Through Convolutional Neural Networks, the system learns to autonomously drive by emulating the behavior of a manual driver. The primary variable the model adjusts is the steering angle, adapting it appropriately based on the encountered situation. Behavioral cloning, showcased in this process, holds significant relevance in real-world self-driving car development.

<h2>Data Collection</h2>

Initiating the behavioral training process begins with downloading the simulator. Initially, we drive the car using keyboard inputs, allowing the Convolutional Neural Network to observe and learn from our controlled vehicle operations. Subsequently, this learned behavior is replicated in autonomous mode through behavioral cloning, where the network mimics our driving actions. The effectiveness of the neural network depends on the accuracy with which it replicates our driving skills. The simulator can be downloaded from the provided link : [Udacity_Self_Driving_Car_Sim](https://github.com/udacity/self-driving-car-sim)

<img src="https://github.com/BenamaraMustapha/Self-Driving-Car-Implementation-Exploring-Behavioural-Cloning-and-Deep-Learning/assets/119163433/3f9c4ba7-caf1-4f6d-9150-5a1121627b10">

After becoming proficient with controlling the car in the simulator using keyboard inputs, we proceed by initiating the recording process to gather data.

<img src="https://github.com/BenamaraMustapha/Self-Driving-Car-Implementation-Exploring-Behavioural-Cloning-and-Deep-Learning/assets/119163433/b2262241-5f30-44b5-8830-75b7bf01c96e">

The recorded data will be saved into a designated folder for storage and further processing.

<img src="https://github.com/BenamaraMustapha/Self-Driving-Car-Implementation-Exploring-Behavioural-Cloning-and-Deep-Learning/assets/119163433/ff3da5ab-3f2b-4186-8da6-f75345f0557f">

To gather insights into our driving behavior, we will conduct three laps of simulated driving, navigating tracks designed with varying textures, curvatures, layouts, and landscapes. This diversity in track features challenges the neural network to adeptly handle different scenarios. Our aim is to drive the car consistently along the center of the track, employing a regression-based approach.

To ensure robustness, we also perform reverse laps to capture a broader range of data for generalization. During reverse laps, we consciously balance between left and right directions to avoid bias in the dataset.

The development of machine learning algorithms involves iterative experimentation with different datasets until reaching the desired performance metrics. We analyze loss and accuracy plots to ascertain whether the model is overfitting or underfitting, adjusting it accordingly. In this regression-type example, we evaluate mean squared error. High mean squared error in both training and validation sets indicates underfitting, while low training error and high validation error signal overfitting.

Our primary objective in the simulator is to maintain straight driving along the centerline at all times, completing three laps in both forward and reverse directions.

The simulator is equipped with three cameras: left, center, and right. Each camera records footage and collects data on steering angle, brake, and throttle for each image.

<h2>Training Process</h2>
To begin the process of developing a self-driving car, we upload the recorded images from the simulator.

- Access [GitHub](https://github.com/) through a web browser.
- If not already registered, create a new account.
- Create a new repository to host the project
- Name the repository with a specified name and set its visibility to public.
- We will now open a command window to check if Git is installed. If Git is not installed, we will proceed with the installation process [Git](https://git-scm.com/downloads).

Now, we'll navigate to the folder where we have saved the recordings, including both the image and CSV files.

We'll issue the command:

    Git init
Then

    Git add .

and the folder will be replicated accordingly.

<img src="https://github.com/BenamaraMustapha/Self-Driving-Car-Implementation-Exploring-Behavioural-Cloning-and-Deep-Learning/assets/119163433/85b4b01b-1f7f-4e67-9513-accbe6426bbb">

We'll be utilizing Google Colab for the training process. To begin, we'll open a new Python 3 notebook by navigating to [Google Colab](https://colab.research.google.com/). Once the notebook is open, we'll proceed to clone the repository using the following command:

    !git clone https://github.com/BenamaraMustapha/SDCE

Now, we'll import all the necessary libraries required for the training process. We'll utilize TensorFlow as the backend and Keras at the frontend.

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import keras
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    from imgaug import augmenters as iaa
    import cv2
    import pandas as pd
    import ntpath
    import random

We'll designate "datadir" as the name for the folder itself and utilize its parameters. Utilizing the "head" command, we will display the first five values from the CSV file in the desired format.

    datadir = 'SDCE'
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
    pd.set_option('display.max_colwidth', -1)
    data.head()

Since the current implementation picks up the entire path from the local machine, we'll use the "ntpath" function to extract the network path. We'll declare a variable named "path_leaf" and assign the extracted network path accordingly.

    def path_leaf(path):
      head, tail = ntpath.split(path)
      return tail
    data['center'] = data['center'].apply(path_leaf)
    data['left'] = data['left'].apply(path_leaf)
    data['right'] = data['right'].apply(path_leaf)
    data.head()

We'll bin the number of values, setting it to 25 to achieve a center distribution (an odd number for this purpose). Then, we'll generate a histogram using the np.histogram function on the 'steering' data frame, dividing it into the specified number of bins.

<img src="https://github.com/BenamaraMustapha/Self-Driving-Car-Implementation-Exploring-Behavioural-Cloning-and-Deep-Learning/assets/119163433/417e6900-a076-43cd-bb9e-81ab6515ba73">

    num_bins = 25
    samples_per_bin = 400
    hist, bins = np.histogram(data['steering'], num_bins)
    center = (bins[:-1]+ bins[1:]) * 0.5
    plt.bar(center, hist, width=0.05)
    plt.plot((np.min(data['steering']), np.max(data['steering'])), \
    (samples_per_bin, samples_per_bin))

We'll maintain the number of samples at 400 and then draw a line. Upon observation, we note that the data is centered around the middle, which is 0.

<img src="https://github.com/BenamaraMustapha/Self-Driving-Car-Implementation-Exploring-Behavioural-Cloning-and-Deep-Learning/assets/119163433/346ea821-599e-490d-9c40-6ba4aa048ed4">











