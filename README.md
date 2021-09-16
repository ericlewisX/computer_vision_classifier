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

## Data - Datasets Origin
The data is from the [German Traffic Sign Recgonition Benchmark](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) dataset hosted on kaggle.

![German Traffic Dataset](images/german_traffic_sign_dataset.png)

## Data
The structure of the data is a directory of 40 classes of traffic sign image directories, where each directory class contains different images of a particular sign for a total of over 50,000 images. This directory can be described as a single-image, multi-class classification dataset.

## Data Distribution
The structure of the data is a directory of 40 classes of traffic sign image directories, where each directory class contains different images of a particular sign for a total of over 50,000 images. This directory can be described as a single-image, multi-class classification dataset.

## Data Augmentation
The structure of the data is a directory of 40 classes of traffic sign image directories, where each directory class contains different images of a particular sign for a total of over 50,000 images. This directory can be described as a single-image, multi-class classification dataset.

## Convolutional Neural Network
We'll start with a `train_test_split` to get our Training and Test Data.

Our baseline will be a Logistic Regression Model where the features are the track properties columns and the target is the one hot encoded column that denotes if the track is a member of the Hot 100 Chart.

We will then use a Decision Tree, followed by a Random Forest. Hyper Parameter tuning will be tried for every applicable model.

There are three performance metrics I considered, Accuracy, Precision, and Recall. For this particular objective, they can be defined as such:

* Accuracy : (number of correct predicitions) / (total number of predictions)
* Precision : (number of true positives) / (Actual results aka [True Positive + False Positive])
* Recall : (number of true positives) / (Predicted Results aka [True Positive + False Negative])

We must also determine what true positive, true negative, false positive and false negative mean in relation to our problem. Once we verbalize that, we can choose the appropriate performance metric to use.

* True Postive : Our model predicted that the song can make it to the Billboard Hot 100 Chart and it actually made it to the Hot 100 Chart.
* True Negative : Our model predicted that the song can not make it to the Billboard Hot 100 Chart and it actually did not make it to the Hot 100 Chart.
* False Positive : Our model predicted that the song can make it to the Billboard Hot 100 Chart but in actuality it did not make it to the Hot 100 Chart.
* False Negative : Our model predicted that the song can not make it to the Billboard Hot 100 Chart but in actuality it did make it to the Hot 100 Chart.

I think the Precision metric is more important to increase but we'll see all three performance metrics just in case readers want to make their own interpretation/justification on which one to use for whatever problem they are approaching. 

The final model and performance metric values are shown in the [Results](#results) section.

#### Logistic Regression
|  | |
|:-------------------------:|:-------------------------:|
|![Logistic Regression Confusion Matrix](plots/confusionLR.png) | ![Logistic Regression Normalized Confusion Matrix](plots/confusionnormLR.png)|


|  | Accuracy | Precision | Recall |
|:----------:|:---------------:| :---: | :---: |
| Logistic Regression | 0.601 | 0.668 | 0.593 |

I used a Logistic Regression model as a baseline. It was better than a random guess at a little over 50%. 

![Logistic Regression Best Model Coefficients](plots/Logistic%20Regression%20Best%20Model%20Coefficients.png)

I also found the Coefficients for each Track property for the best model after hyper parameter tuning.

Since this is just the baseline model we move onto something a bit more promising. 

#### Decision Trees
|  | |
|:-------------------------:|:-------------------------:|
|![Decision Tree Confusion Matrix](plots/confusionDT.png) | ![Decision Tree Normalized Confusion Matrix](plots/confusionnormDT.png)|

|  | Accuracy | Precision | Recall |
|:----------:|:---------------:| :---: | :---: |
| Decision Tree | 0.622 | 0.768 | 0.591 |

After hyper-parameter tuning, we see a signficant boost in all of the performance metrics.

|_ |_|
|:-------------------------:|:-------------------------:|
|![Decision Tree ROC Curve](plots/rocDT.png) | ![Decision Tree Song Permutation Importance](plots/DTFI.png)|

We see that from the Permutation Importance that acousticness has more of an effect on the model's predictive than any other property. However it seems unlikely that two of the several metrics don't have any effect at all, so we'll use something more robust than a Decision Tree; multiple Decision Trees, or a Random Forest.

#### Random Forest
|_ |_|
|:-------------------------:|:-------------------------:|
|![Random Forest Confusion Matrix](plots/confusionRF.png) | ![Random Forest Normalized Confusion Matrix](plots/confusionnormRF.png)|

|  | Accuracy | Precision | Recall |
|:----------:|:---------------:| :---: | :---: |
| Random Forest | 0.736 | 0.899 | 0.674 |

After hyper-parameter tuning, all of our performance metrics increase to an acceptable level and so this is our finalized model.

|_ |_|
|:-------------------------:|:-------------------------:|
|![Decision Tree ROC Curve](plots/roc%20RF.png) | ![Decision Tree Song Permutation Importance](plots/rfFI.png)|

Unlike the plot for Decision Trees the Permutation Importance for Random Forest shows that every track property has a some affect on the model's outcome which seems more realistic. We would need to do further analysis to see each features' individual effect but generally we can see tha acousticness has the most effect on accuracy out of the features we considered for the model.

## Results
The results of the each of the models are shown in the table below:

|            |  Accuracy | Precision | Recall |
|:----------:|:--------------:| :---: | :---: |
| Logistic Regression | 0.601 | 0.668 | 0.593 |
| Decision Tree | 0.622 | 0.768 | 0.591 |
| Random Forest | 0.736 | 0.899 | 0.674 |

Since Random Forest is the obvious winner in all performance metrics considered, the finalized model is the one built on it. 

## Burdens of This project


## Conclusions / Future-Steps

If a record label was using this Random Forest Model, knowing if a song will be a on Billboard'ss The Hot 100 Chart 70% (accuracy) of the time can be extremely lucrative. 

However, the precision metric score in my opinion is more important than the accuracy. 
The translation for precision in relation to our objective is:
* Out of all the tracks that the model predicted made it to the Hot 100, how many actually made it. Our model judged a little under 90%. 

With the foresight that this model can give, the marketing team or an  artist can be more proactive on the songs that actually have a chance at success instead of reactive if a song happens to be a bop.

