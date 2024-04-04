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

We will specify a variable named "remove_list"

<img src="https://github.com/BenamaraMustapha/Self-Driving-Car-Implementation-Exploring-Behavioural-Cloning-and-Deep-Learning/assets/119163433/ff8ca2f2-cbc9-4a60-b416-147b9f8f4353">

        print('total data:', len(data))
        remove_list = []
        for j in range(num_bins):
          list_ = []
          for i in range(len(data['steering'])):
            if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
              list_.append(i)
          list_ = shuffle(list_)
          list_ = list_[samples_per_bin:]
          remove_list.extend(list_)
 
        print('removed:', len(remove_list))
        data.drop(data.index[remove_list], inplace=True)
        print('remaining:', len(data))
 
        hist, _ = np.histogram(data['steering'], (num_bins))
        plt.bar(center, hist, width=0.05)
        plt.plot((np.min(data['steering']), np.max(data['steering'])), \
        (samples_per_bin, samples_per_bin))

Sure, let's define a function called load_img_steering. This function will take two empty lists as arguments: one for the image paths and the other for the corresponding steering angles. We'll loop through the data frame, using the iloc selector to access data based on specific indices. For now, we'll use the "cut" data.

<img src="https://github.com/BenamaraMustapha/Self-Driving-Car-Implementation-Exploring-Behavioural-Cloning-and-Deep-Learning/assets/119163433/4880f2b8-75c7-4b7c-bb6b-c5f960a095c3">
<img src="https://github.com/BenamaraMustapha/Self-Driving-Car-Implementation-Exploring-Behavioural-Cloning-and-Deep-Learning/assets/119163433/6bae34d1-d38d-4fb5-be19-39e1330b7d92">

        print(data.iloc[1])
        def load_img_steering(datadir, df):
          image_path = []
          steering = []
          for i in range(len(data)):
            indexed_data = data.iloc[i]
            center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
            image_path.append(os.path.join(datadir, center.strip()))
            steering.append(float(indexed_data[3]))
            # left image append
            image_path.append(os.path.join(datadir,left.strip()))
            steering.append(float(indexed_data[3])+0.15)
            # right image append
            image_path.append(os.path.join(datadir,right.strip()))
            steering.append(float(indexed_data[3])-0.15)
          image_paths = np.asarray(image_path)
          steerings = np.asarray(steering)
          return image_paths, steerings
 
        image_paths, steerings = load_img_steering(datadir + '/IMG', data)

        X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, \
        test_size=0.2, random_state=6)
        print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))

We'll split the image paths and store the corresponding arrays accordingly within the function.
<img src="https://github.com/BenamaraMustapha/Self-Driving-Car-Implementation-Exploring-Behavioural-Cloning-and-Deep-Learning/assets/119163433/47a0ce15-4e13-4e84-9bef-906d0cb4c18a">
<br>
We will have the histograms now.
<br>
<img src="https://github.com/BenamaraMustapha/Self-Driving-Car-Implementation-Exploring-Behavioural-Cloning-and-Deep-Learning/assets/119163433/3c45d02a-8da5-44e1-8303-cb8840010091">
<br>
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
        axes[0].set_title('Training set')
        axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
        axes[1].set_title('Validation set')

In the upcoming steps, we'll normalize the data. For the NVIDIA model, we'll need to arrange it in a UAV (RGB) pattern and remove unnecessary information. Additionally, we'll preprocess the image.

        def zoom(image):
          zoom = iaa.Affine(scale=(1, 1.3))
          image = zoom.augment_image(image)
          return image
        image = image_paths[random.randint(0, 1000)]
        original_image = mpimg.imread(image)
        zoomed_image = zoom(original_image)
 
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        fig.tight_layout()
 
        axs[0].imshow(original_image)
        axs[0].set_title('Original Image')
 
        axs[1].imshow(zoomed_image)
        axs[1].set_title('Zoomed Image')

        def pan(image):
          pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
          image = pan.augment_image(image)
          return image
        image = image_paths[random.randint(0, 1000)]
        original_image = mpimg.imread(image)
        panned_image = pan(original_image)
 
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        fig.tight_layout()
 
        axs[0].imshow(original_image)
        axs[0].set_title('Original Image')
 
        axs[1].imshow(panned_image)
        axs[1].set_title('Panned Image')
        def img_random_brightness(image):
            brightness = iaa.Multiply((0.2, 1.2))
            image = brightness.augment_image(image)
            return image
        image = image_paths[random.randint(0, 1000)]
        original_image = mpimg.imread(image)
        brightness_altered_image = img_random_brightness(original_image)
 
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        fig.tight_layout()
 
        axs[0].imshow(original_image)
        axs[0].set_title('Original Image')
 
        axs[1].imshow(brightness_altered_image)
        axs[1].set_title('Brightness altered image ')

        def img_random_flip(image, steering_angle):
            image = cv2.flip(image,1)
            steering_angle = -steering_angle
            return image, steering_angle
        random_index = random.randint(0, 1000)
        image = image_paths[random_index]
        steering_angle = steerings[random_index]
 
        original_image = mpimg.imread(image)
        flipped_image, flipped_steering_angle = img_random_flip(original_image, steering_angle)
 
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        fig.tight_layout()
 
        axs[0].imshow(original_image)
        axs[0].set_title('Original Image - ' + 'Steering Angle:' + str(steering_angle))
 
        axs[1].imshow(flipped_image)
        axs[1].set_title('Flipped Image - ' + 'Steering Angle:' + str(flipped_steering_angle))
        def random_augment(image, steering_angle):
            image = mpimg.imread(image)
            if np.random.rand() < 0.5:
              image = pan(image)
            if np.random.rand() < 0.5:
              image = zoom(image)
            if np.random.rand() < 0.5:
              image = img_random_brightness(image)
            if np.random.rand() < 0.5:
              image, steering_angle = img_random_flip(image, steering_angle)
    
            return image, steering_angle
        ncol = 2
        nrow = 10
 
        fig, axs = plt.subplots(nrow, ncol, figsize=(15, 50))
        fig.tight_layout()
 
        for i in range(10):
          randnum = random.randint(0, len(image_paths) - 1)
          random_image = image_paths[randnum]
          random_steering = steerings[randnum]
    
          original_image = mpimg.imread(random_image)
          augmented_image, steering = random_augment(random_image, random_steering)
    
          axs[i][0].imshow(original_image)
          axs[i][0].set_title("Original Image")
  
          axs[i][1].imshow(augmented_image)
          axs[i][1].set_title("Augmented Image")
 
        def img_preprocess(img):
            img = img[60:135,:,:]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img = cv2.GaussianBlur(img,  (3, 3), 0)
            img = cv2.resize(img, (200, 66))
            img = img/255
            return img
        image = image_paths[100]
        original_image = mpimg.imread(image)
        preprocessed_image = img_preprocess(original_image)
 
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        fig.tight_layout()
        axs[0].imshow(original_image)
        axs[0].set_title('Original Image')
        axs[1].imshow(preprocessed_image)
        axs[1].set_title('Preprocessed Image')
        def batch_generator(image_paths, steering_ang, batch_size, istraining):
  
          while True:
            batch_img = []
            batch_steering = []
    
            for i in range(batch_size):
              random_index = random.randint(0, len(image_paths) - 1)
      
              if istraining:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
     
              else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]
      
              im = img_preprocess(im)
              batch_img.append(im)
              batch_steering.append(steering)
            yield (np.asarray(batch_img), np.asarray(batch_steering))


To address the complexity of our dataset, which involves images with dimensions (200, 66) and the need to classify traffic signs, we're transitioning from the Lenet 5 model to the NVIDIA model. Our dataset comprises 3,511 images for training, unlike the MNIST dataset, which contains around 60,000 images.

Since our task involves returning an appropriate steering angle, which is a regression-type example, we require a more advanced model. The NVIDIA model provides such sophistication.

The architecture of the NVIDIA model is as follows:

<img src="https://github.com/BenamaraMustapha/Self-Driving-Car-Implementation-Exploring-Behavioural-Cloning-and-Deep-Learning/assets/119163433/055a4d74-8044-4743-9a16-75d1b00a45f0">

To define the model architecture, we'll start by creating the model object. We'll skip the normalization step since the data is already normalized. Then, we'll add the convolutional layer.

In comparison to the model, we'll organize accordingly. The NVIDIA model utilizes 24 filters in the layer with a kernel size of 5x5. We'll introduce sub-sampling, where the stride length of the kernel is adjusted to process large images. Horizontal movement will be set to 2 pixels at a time, and vertical movement to 2 pixels at a time.

Since this is the first layer, we need to define the input shape of the model as (66, 200, 3), and the activation function will be "elu".

        def nvidia_model():
          model = Sequential()
          model.add(Convolution2D(24, 5, 5, subsample=(2, 2),
          input_shape=(66, 200, 3), activation='elu'))

Upon revisiting the model, we observe that our second layer consists of 36 filters with a kernel size of (5,5). We'll apply the same sub-sampling option with a stride length of (2,2) for this layer as well, and conclude it with the activation function "elu".

        model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))

According to the NVIDIA model, we have three additional layers in the convolutional neural network. The third layer consists of 48 filters, followed by two layers with 64 filters each, both with (3,3) kernel size.

Since the dimensions have been significantly reduced by this point, we'll remove subsampling from the fourth and fifth layers.

        model.add(Convolution2D(64, 3, 3, activation='elu'))
        model.add(Convolution2D(64, 3, 3, activation='elu'))

Next, we add a Flatten layer. This layer will take the output array from the previous convolutional neural network and convert it into a one-dimensional array. This transformation enables the data to be fed into the fully connected layer that follows.

        model.add(Flatten())

Our last convolution layer outputs an array shape of (1,18) by 64.

        model.add(Convolution2D(64, 3, 3, activation='elu'))

We conclude the architecture of the NVIDIA model with a dense layer containing a single output node, which will output the predicted steering angle for our self-driving car.

Now, we'll use model.compile() to compile our architecture. Since this is a regression-type example, the metrics we'll be using are mean squared error, and we'll optimize using Adam. We'll employ a relatively low learning rate to enhance accuracy.

To address overfitting, we'll incorporate a Dropout layer. This layer randomly sets a fraction of input units to 0 during each update, forcing the model to learn from different combinations of nodes.

We'll separate the convolutional layers from the fully connected layers with a Dropout factor of 0.5, ensuring that 50% of the input is set to 0.

We'll define the model by calling the NVIDIA model itself.

Now, we'll move to the model training process. To define training parameters, we'll use model.fit(). We'll import our training data (X_train and y_train). Since we have a smaller dataset, we'll need more epochs to effectively train the model. We'll also utilize validation data and specify the batch size.

        def nvidia_model():
          model = Sequential()
          model.add(Convolution2D(24, 5, 5, subsample=(2, 2),
          input_shape=(66, 200, 3), activation='elu'))
          model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
          model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
          model.add(Convolution2D(64, 3, 3, activation='elu'))
  
          model.add(Convolution2D(64, 3, 3, activation='elu'))
        #   model.add(Dropout(0.5))  
  
          model.add(Flatten())
  
          model.add(Dense(100, activation = 'elu'))
        #   model.add(Dropout(0.5))
  
          model.add(Dense(50, activation = 'elu'))
        #   model.add(Dropout(0.5))
  
          model.add(Dense(10, activation = 'elu'))
        #   model.add(Dropout(0.5))
 
          model.add(Dense(1))
  
          optimizer = Adam(lr=1e-3)
          model.compile(loss='mse', optimizer=optimizer)
          return model

        model = nvidia_model()
        print(model.summary())
        history = model.fit_generator(batch_generator(X_train, y_train, 100, 1),
                                          steps_per_epoch=300, 
                                          epochs=10,
                                          validation_data=batch_generator(X_valid, y_valid, 100, 0),
                                          validation_steps=200,
                                          verbose=1,
                                          shuffle = 1)

<h3>We choose ELU (Exponential Linear Unit) over ReLU (Rectified Linear Unit)</h3> because ELU has a built-in mechanism to prevent dead neurons.

In ReLU, a node in the neural network may "die" and only pass a value of zero to the nodes that follow it. This can lead to the issue of dead ReLUs, particularly in deep networks.

ELU, on the other hand, allows negative values which helps to avoid this problem. It has the ability to recover and fix errors during the learning process, contributing to the overall performance of the model.

After defining our model architecture with ELU activation functions, we'll visualize the model and save it in the HDF5 format for Keras. This format allows us to save both the architecture and weights of the model in a single file, making it easy to reload the model for future use.

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['training', 'validation'])
        plt.title('Loss')
        plt.xlabel('Epoch')

We will save the model then download it

        model.save('model.h5')
        from google.colab import files
        files.download('model.h5')

<h2>The Connection Part</h2>

<img src="https://github.com/BenamaraMustapha/Self-Driving-Car-Implementation-Exploring-Behavioural-Cloning-and-Deep-Learning/assets/119163433/e74975c4-8b68-4506-b970-9a491a55093e">

This step is crucial for executing the model within the simulated car environment.

To implement a web service using Python, we'll need to install Flask. We'll utilize the Anaconda environment for this purpose. Flask is a Python micro-framework designed for building web applications.

We'll use Visual Studio Code for this task.

Before proceeding, we need to open the folder where the saved *.h5 file is located. Then, we'll open a file. However, before that, we'll install some dependencies.

Additionally, we'll create an Anaconda environment for our work.

        F:\SDCE>conda create --name myenviron
        Fetching package metadata ...............
        Solving package specifications:
        Package plan for installation in environment C:\Users\mustapha\Miniconda3\envs\myenviron:

        Proceed ([y]/n)? y

        
        # To activate this environment, use:
        # > activate myenviron
        #
        # To deactivate this environment, use:
        # > deactivate myenviron
        #
        # * for power-users using bash, you must source

We will activate the environment

        F:\SDCE>activate myenviron

        (myenviron) F:\SDCE>

Let's proceed with installing the necessary dependencies for web sockets

        ---------------------------------------------------------------------------------
        (myenviron) F:\SDCE>conda install -c anaconda flask
        Fetching package metadata .................
        Solving package specifications: .
        Warning: 4 possible package resolutions (only showing differing packages):
          - anaconda::jinja2-2.10-py36_0, anaconda::vc-14.1-h21ff451_3
          - anaconda::jinja2-2.10-py36_0, anaconda::vc-14.1-h0510ff6_3
          - anaconda::jinja2-2.10-py36h292fed1_0, anaconda::vc-14.1-h21ff451_3
          - anaconda::jinja2-2.10-py36h292fed1_0, anaconda::vc-14.1-h0510ff6_3

        Package plan for installation in environment C:\Users\mustapha\Miniconda3\envs\myenviron:

        The following NEW packages will be INSTALLED:

            click:          7.0-py36_0           anaconda
            flask:          1.0.2-py36_1         anaconda
            itsdangerous:   1.1.0-py36_0         anaconda
            jinja2:         2.10-py36_0          anaconda
            markupsafe:     1.1.0-py36he774522_0 anaconda
            pip:            18.1-py36_0          anaconda
            python:         3.6.7-h33f27b4_1     anaconda
            setuptools:     27.2.0-py36_1        anaconda
            vc:             14.1-h21ff451_3      anaconda
            vs2015_runtime: 15.5.2-3             anaconda
            werkzeug:       0.14.1-py36_0        anaconda
            wheel:          0.32.3-py36_0        anaconda

        Proceed ([y]/n)? y

        vs2015_runtime 100% |###############################| Time: 0:00:03 646.12 kB/s
        vc-14.1-h21ff4 100% |###############################| Time: 0:00:00 933.96 kB/s
        python-3.6.7-h 100% |###############################| Time: 0:00:14   1.47 MB/s
        click-7.0-py36 100% |###############################| Time: 0:00:00   1.40 MB/s
        itsdangerous-1 100% |###############################| Time: 0:00:00   2.56 MB/s
        markupsafe-1.1 100% |###############################| Time: 0:00:00 430.49 kB/s
        setuptools-27. 100% |###############################| Time: 0:00:01 622.77 kB/s
        werkzeug-0.14. 100% |###############################| Time: 0:00:00 806.47 kB/s
        jinja2-2.10-py 100% |###############################| Time: 0:00:00 813.78 kB/s
        wheel-0.32.3-p 100% |###############################| Time: 0:00:00   1.54 MB/s
        flask-1.0.2-py 100% |###############################| Time: 0:00:00 726.80 kB/s
        pip-18.1-py36_ 100% |###############################| Time: 0:00:01   1.15 MB/s

We'll start by writing a Python file. First, we'll import Flask. Using Flask, we'll initialize our application as 'app' and set it equal to 'Flask(name)', creating an instance for our web app.

We'll declare a special variable called 'name', which will suffice as 'main'. Then, we'll define a function called 'greeting', which will return a string "welcome".

Next, we'll specify a route decorator '@app.route("/home")', which, when accessed via the URL '/home', will invoke the following function and return the appropriate string as shown in the browser.

After running the Python code, we'll have a web app with some content returned by Python.

To enhance functionality, we'll install additional dependencies such as 'socketio' and others. This will allow us to connect the self-driving car's autonomous mode and utilize web sockets to make it work using the trained Keras model file.

        (myenviron) F:\SDCE1>conda install -c conda-forge eventlet
        Fetching package metadata ...............
        Solving package specifications: .

        Package plan for installation in environment C:\Users\mustapha\Miniconda3\envs\myenviron:

        The following NEW packages will be INSTALLED:

            ca-certificates: 2018.11.29-ha4d7672_0    conda-forge
            cffi:            1.11.5-py36hfa6e2cd_1001 conda-forge
            cryptography:    1.7.1-py36_0
            eventlet:        0.23.0-py36_1000         conda-forge
            greenlet:        0.4.13-py36_0            conda-forge
            idna:            2.8-py36_1000            conda-forge
            openssl:         1.0.2p-hfa6e2cd_1001     conda-forge
            pyasn1:          0.4.4-py_1               conda-forge
            pycparser:       2.19-py_0                conda-forge
            pyopenssl:       16.2.0-py36_0            conda-forge
            six:             1.12.0-py36_1000         conda-forge

        Proceed ([y]/n)? y

        ca-certificate 100% |###############################| Time: 0:00:00 313.57 kB/s
        openssl-1.0.2p 100% |###############################| Time: 0:00:05   1.01 MB/s
        greenlet-0.4.1 100% |###############################| Time: 0:00:00 348.11 kB/s
        idna-2.8-py36_ 100% |###############################| Time: 0:00:00 177.91 kB/s
        pyasn1-0.4.4-p 100% |###############################| Time: 0:00:00 770.67 kB/s
        pycparser-2.19 100% |###############################| Time: 0:00:00   1.10 MB/s
        six-1.12.0-py3 100% |###############################| Time: 0:00:00 153.41 kB/s
        cffi-1.11.5-py 100% |###############################| Time: 0:00:00 533.62 kB/s
        pyopenssl-16.2 100% |###############################| Time: 0:00:00 800.90 kB/s
        eventlet-0.23. 100% |###############################| Time: 0:00:00 852.74 kB/s

        (myenviron) F:\SDCE1>

        ------conda package sockets
        (myenviron) F:\SDCE1>conda install -c conda-forge python-socketio
        Fetching package metadata ...............
        Solving package specifications: .

        Package plan for installation in environment C:\Users\mustapha\Miniconda3\envs\myenviron:

        The following NEW packages will be INSTALLED:

            python-engineio: 3.0.0-py_0 conda-forge
            python-socketio: 2.1.2-py_0 conda-forge

        Proceed ([y]/n)? y

        python-enginei 100% |###############################| Time: 0:00:00 177.16 kB/s
        python-socketi 100% |###############################| Time: 0:00:00 346.08 kB/s

In the 'myenviron' environment, we'll execute the 'drive.py' file and connect it to the local server. Then, we'll run the simulator in autonomous mode. Once the connection is established, we'll observe the self-driving car moving within the environment.

        import socketio
        import eventlet
        import numpy as np
        from flask import Flask
        from keras.models import load_model
        import base64
        from io import BytesIO
        from PIL import Image
        import cv2
 
        sio = socketio.Server()
 
        app = Flask(__name__) #'__main__'
        speed_limit = 10
        def img_preprocess(img):
            img = img[60:135,:,:]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img = cv2.GaussianBlur(img,  (3, 3), 0)
            img = cv2.resize(img, (200, 66))
            img = img/255
            return img
 
        @sio.on('telemetry')
        def telemetry(sid, data):
            speed = float(data['speed'])
            image = Image.open(BytesIO(base64.b64decode(data['image'])))
            image = np.asarray(image)
            image = img_preprocess(image)
            image = np.array([image])
            steering_angle = float(model.predict(image))
            throttle = 1.0 - speed/speed_limit
            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)

        @sio.on('connect')
        def connect(sid, environ):
            print('Connected')
            send_control(0, 0)
 
        def send_control(steering_angle, throttle):
            sio.emit('steer', data = {
                'steering_angle': steering_angle.__str__(),
                'throttle': throttle.__str__()
            })
 
        if __name__ == '__main__':
            model = load_model('model.h5')
            app = socketio.Middleware(sio, app)
            eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

When the self-driving car simulator is in running mode:

<img src="https://github.com/BenamaraMustapha/Self-Driving-Car-Implementation-Exploring-Behavioural-Cloning-and-Deep-Learning/assets/119163433/e924d34f-9c0b-4b63-9edc-c7c71df3a050">

<h2>In conclusion</h2>

the entire solution, step by step, is included in the IPython file. You are encouraged to modify it according to your needs. I have also provided all the necessary steps to follow along with the tutorial. If you have any questions or need further assistance, please don't hesitate to ask.














