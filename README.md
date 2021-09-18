![Last Update](https://img.shields.io/badge/last%20change-September%20--%2015%20--%202021%20-yellowgreen)
# A Glimpse into Automotive Computer Vision Using Convolutional Neural Networks
Galvanize Capstone Project: A Glimpse into Automotive Computer Vision Using Convolutional Neural Networks

## Objective
The objective of this project is to introduce the field of computer vision and how it is implemented in smart cars. It won't be long until we all have a vehicle with some type of smart technology in it and it would be nice if everyone has a basic understanding of what goes on in the brain of their car. 

## Understanding the Motivation 
[Computer Vision](https://www.ibm.com/topics/computer-vision) is a field of Artificial Intelligence that enables computers and systems to extract meaningful information from images, videos, and other visual inputs.

![Statista](images/statista.jpeg)

I'm  choosing the automotive industry because over the years, as you can see consumer confidence in self driving cars has slowly picked up. This is probably because of the advancements in sensor technology and the refinement of Data Science techniques. 

This project will showcase one of those techniques, the Convolutional Neural Network, and how it can be used for image classification in relation to the requirements of smart automobiles. 

## Data 
### Datasets Origin
The data is from the [German Traffic Sign Recgonition Benchmark](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) dataset hosted on kaggle.

![German Traffic Dataset](images/german_traffic_sign_dataset.png)

### Data
The structure of the data is a directory of 40 classes of traffic sign image directories, where each directory class contains different images of a particular sign for a total of over 50,000 images. This directory can be described as a single-image, multi-class classification dataset.

Excerpt of the dataframe:

![Labels Dataframe](plots/Label_df.png)

Infographic of all classes and their corresponding sign.

![Traffic Sign Infographic](images/Traffic_Sign_Label_Infographic.jpg)

### Data Distribution
Here is a distribution of the training data after the `train_test_split`.

![Data Distribution Training](plots/DistributionTrainingExamples.png)

As you can see it is extremely imbalanced. 

### Data Augmentation
In a perfect world I would have enough of each class of images (each unique sign) that I wouldn't need to balance the dataset. Since that is not the case, I balanced the dataset by generating new images. I chose to augment the data through resizing, grayscaling, equalization, and normalization. These small changes to copies of the data lets me count them as different images. 

## Convolutional Neural Network

![Code Basics CNN Koala](images/CodeBasicsCNNKoala.png)

## Our Model

![CNN Model](plots/CNN_Model.png)

## Results
The results of the each of the models are shown in the table below:


![Accuracy CNN](plots/AccuracyCNN.png) ![Loss CNN](plots/LossCNN.png)

|  | Loss | Accuracy |
| :--: | :--: | :----: |
| CNN | 0.0514 | 0.9842 |

![Tkinter App](plots/tkinter.png)

## Burdens of This Project and their Solutions
There is one major pitfall to this project which leads to the "misclassification" of some of the signs. Misclassification is in quotes because even though the predicting are incorrect, according to the training data, the model is predicting exactly how is should.

The problem originates in the data augmentation step.

## Conclusions / Future-Steps

If a record label was using this Random Forest Model, knowing if a song will be a on Billboard'ss The Hot 100 Chart 70% (accuracy) of the time can be extremely lucrative. 

However, the precision metric score in my opinion is more important than the accuracy. 
The translation for precision in relation to our objective is:
* Out of all the tracks that the model predicted made it to the Hot 100, how many actually made it. Our model judged a little under 90%. 

With the foresight that this model can give, the marketing team or an  artist can be more proactive on the songs that actually have a chance at success instead of reactive if a song happens to be a bop.

