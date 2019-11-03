
# Behavioral Cloning Project
The target is here to train the weights of our network to minimize the mean squarred error between the steering command output by the network and the command of either the human driver.
To make that, we collect firstly the data with a normal drive with a human driver. We will compare these recording data with the data of the cameras of our autonomous drive. The RMSE will compare for every situation the desired steering command with the network computed steering command (from our CNN). One Error measurement will be measured and we will adapt the weight in the back propagation step to reduce in maximum this error.

--- 

# The goals / steps of this project are the following:
Use the simulator to collect data of good driving behavior
Build, a convolution neural network in Keras that predicts steering angles from images
Train and validate the model with a training and validation set
Test that the model successfully drives around track one without leaving the road
Summarize the results with a written report

---

# Files Submitted & Code Quality
My project includes the following files:
- input for the application "behavior cloning" :
- model.py containing the script to create and train the model
- drive.py for driving the car in autonomous mode
- folder data with the csv file with the "training mode" data recording (features and labels)

# output/results from the application "behavior cloning" :
- model.h5 containing a trained convolution neural network
- folder named run1 with the image after the "autonomous mode" testing
- run1.mp4 with the results of the SW pipeline for behavior cloning
- writeup_report.md to summarize the results

---

#instructions :
model.py generates a model.h5 : this model will be used to test the model in the real simulation window
python drive.py model.h5 run1
ls run
python video.py run1 --fps48
