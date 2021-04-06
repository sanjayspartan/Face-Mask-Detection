# Face-Mask-Detection
Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning
we’ll discuss our two-phase COVID-19 face mask detector, detailing how our computer vision/deep learning pipeline will be implemented.

From there, we’ll review the dataset we’ll be using to train our custom face mask detector.
I’ll then show you how to implement a Python script to train a face mask detector on our dataset using Keras and TensorFlow.
We’ll use this Python script to train a face mask detector and review the results.
Given the trained COVID-19 face mask detector, we’ll proceed to implement two more additional Python scripts used to:
Detect COVID-19 face masks in images
Detect face masks in real-time video streams

# COVID-19: Face Mask Detection using TensorFlow and OpenCV
In these tough COVID-19 times, wouldn’t it be satisfying to do something related to it? I decided to build a very simple and basic Convolutional Neural Network (CNN) model using TensorFlow with Keras library and OpenCV to detect if you are wearing a face mask to protect yourself.

I am going to use these images to build a CNN model using TensorFlow to detect if you are wearing a face mask by using the webcam of your PC.

# Step 1: Data Visualization
In the first step, let us visualize the total number of images in our dataset in both categories. We can see that there are 690 images in the ‘yes’ class and 686 images in the ‘no’ class.
The number of images with facemask labelled 'yes': 690 
The number of images with facemask labelled 'no': 686
# Step 2: Data Augmentation
In the next step, we augment our dataset to include more number of images for our training. In this step of data augmentation, we rotate and flip each of the images in our dataset. We see that, after data augmentation, we have a total of 2751 images with 1380 images in the ‘yes’ class and ‘1371’ images in the ‘no’ class.
Number of examples: 2751 
Percentage of positive examples: 50.163576881134134%, number of pos examples: 1380 
Percentage of negative examples: 49.836423118865866%, number of neg examples: 1371
# Step 3: Splitting the data
In this step, we split our data into the training set which will contain the images on which the CNN model will be trained and the test set with the images on which our model will be tested.
In this, we take split_size =0.8, which means that 80% of the total images will go to the training set and the remaining 20% of the images will go to the test set.
The number of images with facemask in the training set labelled 'yes': 1104
The number of images with facemask in the test set labelled 'yes': 276
The number of images without facemask in the training set labelled 'no': 1096
The number of images without facemask in the test set labelled 'no': 275
After splitting, we see that the desired percentage of images have been distributed to both the training set and the test set as mentioned above.
# Step 4: Building the Model
In the next step, we build our Sequential CNN model with various layers such as Conv2D, MaxPooling2D, Flatten, Dropout and Dense. In the last Dense layer, we use the ‘softmax’ function to output a vector that gives the probability of each of the two classes.
# Step 5: Pre-Training the CNN model
After building our model, let us create the ‘train_generator’ and ‘validation_generator’ to fit them to our model in the next step. We see that there are a total of 2200 images in the training set and 551 images in the test set.
Found 2200 images belonging to 2 classes. 
Found 551 images belonging to 2 classes.
# Step 6: Training the CNN model
This step is the main step where we fit our images in the training set and the test set to our Sequential model we built using keras library. I have trained the model for 30 epochs (iterations). However, we can train for more number of epochs to attain higher accuracy lest there occurs over-fitting.
history = model.fit_generator(train_generator,
                              epochs=30,
                              validation_data=validation_generator,
                              callbacks=[checkpoint])
>>Epoch 30/30
220/220 [==============================] - 231s 1s/step - loss: 0.0368 - acc: 0.9886 - val_loss: 0.1072 - val_acc: 0.9619
We see that after the 30th epoch, our model has an accuracy of 98.86% with the training set and an accuracy of 96.19% with the test set. This implies that it is well trained without any over-fitting.
# Step 7: Labeling the Information
After building the model, we label two probabilities for our results. [‘0’ as ‘without_mask’ and ‘1’ as ‘with_mask’]. I am also setting the boundary rectangle color using the RGB values.[‘RED’ for ‘without_mask’ and ‘GREEN’ for ‘with_mask]
labels_dict={0:'without_mask',1:'with_mask'} 
color_dict={0:(0,0,255),1:(0,255,0)}
# Step 8: Importing the Face detection Program
After this, we intend to use it to detect if we are wearing a face mask using our PC’s webcam. For this, first, we need to implement face detection. In this, I am using the Haar Feature-based Cascade Classifiers for detecting the features of the face.
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
This cascade classifier is designed by OpenCV to detect the frontal face by training thousands of images. The .xml file for the same needs to be downloaded and used in detecting the face. I have uploaded the file in my GitHub repository.
# Step 9: Detecting the Faces with and without Masks
In the last step, we use the OpenCV library to run an infinite loop to use our web camera in which we detect the face using the Cascade Classifier. The code webcam = cv2.VideoCapture(0)denotes the usage of webcam.
The model will predict the possibility of each of the two classes ([without_mask, with_mask]). Based on which probability is higher, the label will be chosen and displayed around our faces.
