#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* src/model.py containing the script to define the model in keras
* src/DataLoadHelper.py containing the script to load csv files, read images, create generator for images with given batch size
* src/behavioral_cloning.py 
* test/drive.py for driving the car in autonomous mode
* results/model.h5 containing a trained convolution neural network 
* docs/writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5 convolution layers, 1 flatten and 5 fully connected layers. 
The data is subsampled by a factor of 2 in the first 2 convolution layers and used 5x5 filters for these 2 layers. For rest 3 conv layers 3x3 filter sizes were used. The depths of these layers are in between 24 and 64. 

The model includes RELU layers to introduce nonlinearity. The data is cropped first using keras cropping layer and then normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
Three test image for center, left and right positions are used to test basic correctness of the network. 

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (driving both clockwise and anticlockwise direction), recovering from the left and right sides of the road. I also collected more data of the cases where it was failing after running the model.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with simple networks and try to known networks to see how it behaves. 

My first step was to use a convolution neural network model similar to the LENET. I thought this model might be appropriate because we are mainly searching for edges in the images as we did with the traffic signs. It didn't work that well. So, I started to make the network more complex by adding more convolution and full connected layers. As the model became bigger I added dropout layers to combat the overfitting (there was high mean squared error on validation set and low mse on train set).

Then I tried NVIDIA's architecture. It worked quite well but I though it can be made simpler as images from simulator are not as complicated as real world. So, I subsampled the images in the first two layers which can also help in speeding up computation with the network.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (especially the one after the brdige). To improve the driving behavior in these cases, I recorded and added more data to train the model what to do in these situations.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. Phew!!!

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture 

| Layer  | Description |
| ------------- | ------------- |
| Input  | 160 x 320 x 3  |
| Cropping2D  | 90 x 320 x 3  |
| Convolution 5x5  | 2x2 subsample, 1x1 stride  |
| RELU  |  |
| Convolution 5x5  | 2x2 subsample, 1x1 stride  |
| RELU  |  |
| Convolution 5x5  | 1x1 stride  |
| RELU  |  |
| Convolution 3x3  | 1x1 stride  |
| RELU  |  |
| Convolution 3x3  | 1x1 stride  |
| RELU  |  |
| FLATTEN  |  |
| Fully Connected  | output 1164  |
| Fully Connected  | output 100  |
| Fully Connected  | output 50  |
| Fully Connected  | output 10  |
| Fully Connected  | output 1  |


####3. Creation of the Training Set & Training Process

[image1]: ./img/center.png "Car in center. steering angle =0"
[image2]: ./img/rightmost.png "Car out of the lane on the right side. High value of steering angle"
[image3]: ./img/rightcenter.png "Car just inside the lane on the right side. Medium steering angle"
[image4]: ./img/rightcenternear.png "Car near to the center but slightly drifter on the right side. Low steering angle"

To capture good driving behavior, I first recorded two laps on track one using center lane driving (one whiaaaaaaaaaale going in clockwise direction and other in the anti-clocwise direction). Here is an example image of center lane driving:

![alt_text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to the center of the track. This is helpful while having sharp turns. These images show what a recovery :

![alt text][image2]
![alt text][image3]
![alt text][image4]

I tried to augment the data but made the behavior of the model more unrealible. In addition, as i collected the data for both clockwise and anti-clockwise - I didn't had to add approximations for the other side by flipping the image.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

After the collection process, I then cropped and normalized the images using the keras pre-processing layers. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by visualizing the mean squared error for training and validation data. I used an adam optimizer so that manually training the learning rate wasn't necessary.
