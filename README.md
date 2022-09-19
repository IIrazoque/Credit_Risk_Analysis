# Credit_Risk_Analysis

## Overview of the analysis
The purpose of the analysis is use supervised ML (machine learning) models to help predict credit risk in LendingClub’s dataset. LendingClub is a peer-to-peer lending services company.

Resources: [LoanStats_2019Q1.csv](https://github.com/IIrazoque/Credit_Risk_Analysis/blob/d35166dbce6ba6f937b788431156c7176d1e84de/LoanStats_2019Q1.csv), supervised ML, imbalanced-learn and scikit-learn libraries

The data we’re dealing with would fall under the “Classification” type and we will create features and target variables to help us label the data. Then split the Dataset into two sets: training dataset (X= features) and testing dataset (AKA y= target) (figure1). We will then use 4 Resampling models to predict credit risk. Lastly Use Ensemble Learners to enhance the predictions. 

*Figure 1. Read the CSV and Perform Basic Data Cleaning*

![This is an image]( https://github.com/IIrazoque/Credit_Risk_Analysis/blob/6b44c56bcbf534ef904abda5add448ecdd2c007b/Images/Image1.PNG)
 
 
Resampling Models  
-	Naïve Random Oversampling 
-	SMOTE Oversampling 
-	Cluster Centriod Undersampling 
-	SMOTEENN 

Combining Prediction Models / Ensemble Learners 
-	Balanced Random Forest Classifier 
-	Easy Ensemble AdaBoost Classifier 

## Resampling Models
### Naïve Random Oversampling

Naïve Random Oversampling is randomly selecting the minority class and adding data points until both classes are equally balanced. Using the RandomOverSample function we were able to resample the low_risk & high_risk classes to; 51366 data points. The following values were generated: Precision High Risk: 1%, Precision Low Risk: 100%, Recall High Risk: 71%, Recall Low Risk: 58% 

*Figure 2. NAÏVE Random Oversampling Modeling*

![This is an image]( https://github.com/IIrazoque/Credit_Risk_Analysis/blob/6b44c56bcbf534ef904abda5add448ecdd2c007b/Images/Image2.PNG)
 
Accuracy Score: 65% 
Average f1 Score: 73%

### SMOTE Oversampling 
Synthetic minority oversampling technique is when random sampling in increased from the minority class. The Accuracy score was calculated along with the confusion matrix. The classification report was also generated with the following values. Precision High Risk: 1%, Precision Low Risk: 100%, Recall High Risk: 63%, Recall Low Risk: 64%. 

*Figure 3. SMOTE Oversampling Modeling*

![This is an image]( https://github.com/IIrazoque/Credit_Risk_Analysis/blob/6b44c56bcbf534ef904abda5add448ecdd2c007b/Images/Image3.PNG)
 
Accuracy Score: 66% 
Average F1 Score: 81%

### Cluster Centriod Undersampling
As the name implies were removed data points from the largest class and make this equal to the minority class. Using the ClusterCentroids function we were able to equate class datasets to 246. The following values were generated; Precision high_risk: 1%, Precision low_risk: 100%, Recall High Risk: 69%, Recall Low Risk: 40%. 

*Figure 4. Cluster Centriod Undersampling Modeling*

![This is an image]( https://github.com/IIrazoque/Credit_Risk_Analysis/blob/6b44c56bcbf534ef904abda5add448ecdd2c007b/Images/Image4.PNG)
 
Accuracy Score: 66% 
Average F1 Score: 56%

### SMOTEENN
The SMOTEENN is the combination of Oversampling and Undersampling. This is accomplished by combining the SMOTE and Edited Nearest Neighbors (ENN) algorithms (hence the name). The minority class is oversampled then outliers are removed from each class. The following values were generated; Precision high_risk: 1%, Precision low_risk: 100%, Recall High Risk: 72%, Recall Low Risk: 57%. 

*Figure 5. SMOTEENN Modeling*

![This is an image]( https://github.com/IIrazoque/Credit_Risk_Analysis/blob/6b44c56bcbf534ef904abda5add448ecdd2c007b/Images/Image5.PNG)
 
Accuracy Score: 65%
Average F1 Score: 72%

Now let’s use another way of making predictions by using Ensemble Learners.

## Ensembles Learners 
Ensemble Learners is multiple learning algorithms (or decision trees) to help make better predictions. It lowers error, less overfitting and essentially removes the bias that a prediction model may have. We can use the following prediction models to make these predictions :
-	Balanced Random Forest Classifier 
-	Easy Ensemble AdaBoost Classifier

### Balanced Random Forest Classifier 
We generated ~100 “decision trees” to partake in the RandomForestClassifier function, then split the data into training and testing datasets (figure 6). The following values were generated; Precision high_risk: 88%, Precision low_risk: 100%, Recall High Risk: 37%, Recall Low Risk: 100%.

*Figure 6. Generating the datasets*

![This is an image]( https://github.com/IIrazoque/Credit_Risk_Analysis/blob/6b44c56bcbf534ef904abda5add448ecdd2c007b/Images/Image6.PNG)
 
Figure 7. Balanced Random Forest Classifier Output 

![This is an image]( https://github.com/IIrazoque/Credit_Risk_Analysis/blob/6b44c56bcbf534ef904abda5add448ecdd2c007b/Images/Image7.PNG)
 
Accuracy Score: 68% 
Average F1 Score: 100%

### Easy Ensemble AdaBoost Classifier
Adaptive Boosting (AdaBoost for short) feeds off the previous model and evaluates the “weights” from each model as it moves to the next iteration. Again, we set the n_estimator to 100 and used the EasyEnsembleClassifier function. The following values were generated; Precision high_risk: 88%, Precision low_risk: 100%, Recall High Risk: 37%, Recall Low Risk: 100%. Like the outputs for Balanced Random Forest Classifier, yet the accuracy score is ideal!

*Figure 8. Easy Ensemble AdaBoost Classifier Output*

![This is an image]( https://github.com/IIrazoque/Credit_Risk_Analysis/blob/6b44c56bcbf534ef904abda5add448ecdd2c007b/Images/Image8.PNG)
 
Accuracy Score: 93% 
Average F1 Score: 100%

### Summary 
Based on all predictions, accuracy and F1 scores we determined that Easy Ensemble Adaboost Classification modeling is the best model in predicting credit risk. It has the highest accuracy score if 93%. 
Our credit risk dataset oversampling and undersampling prediction methods were no not reliable methods for predictions, mostly scoring under 70%. 
Oversampling methods are growing the minority group which is done by generating data points based on the distance of the neighboring dataset. These could be outliers and can misinterpret the analysis. Another downside to Undersampling would be the loss of data, hurting the analysis if the dataset set not large enough to begin with. In our case we created a “prediction” trained it based on error weight and passed to the next prediction and so on forth. This minimizes the error rate and hence generated a better prediction scores. 
