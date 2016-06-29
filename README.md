# Investigating Enron financial and email data
David Hurley

## Data Exploration
(https://en.wikipedia.org/wiki/Enron_Corpus)[Email Corpus Wiki] 
Before Enron filed for bankruptcy in 2001 it was one of the largest energy companies in the world employing approximately 20,000 people and claimed revenues of $111 billion just a year earlier.  This downfall was due to a highly publicized account fraud and corruption scandal. The financial and email data are not public record. 

The purpose of this project is to build a person of interest identifier based on financial and email data made public as a result of the Enron scandal. The folks at Udacity have been nice enough to put all the relevant data is a dictionary which contains 14 financial features ('salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']), 6 email features ('to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'), and a POI label.

A machine learning analysis would be beneficial to understand some of the causes of this type of fraud. The features that have the most impact would give us insight into the motivations behind those employees that were corrupted. Similar models may even be used in the future to detect and hopefully prevent something like this happening again. 

Of the 146 data points 18 are labeled as a POI which is not a significant amount of POI data. Printing out the complete data set I see that there are many values that are NaN. I removed these data points as needed which I will dicuss in the next section on outliers. 


## Outlier Investigation
For outliers I started by visualizing salary versus bonus and there is one clear outlier. I found this data point by excluding people who had a bonus of NaN and finding the max value which belonged to 'TOTAL'. Obviously it makes sense to exclude this data going forward. 

I then looked at the top salaries while excluding those salaries that were NaN and came across two key people as listed on the Enron wiki page, https://en.wikipedia.org/wiki/Enron_scandal. While these are outliers it made sense not to exclude them. 


## New Features
I decided to add two new features to the data set, the ratio of emails a person recieved from a POI (ratio_from_poi) and the ratio of emails sent to a POI (ratio_to_poi). Intuitively if most of your email exchange is with a POI you most likely at least know about the scandal and are a POI yourself. I started by visualizing the relationship between from_messages and to_messages and saw that people typically recieve many more emails then they send. Taking the ratio here makes sense because we view the data in a scale between 0 - 1. 

## Selected Features
I found this to be the hardest part of the assignment. At first I started with features I started with only features I thought we relevant like those related to stock, earnings, and correspondance with POIs. I originally tried SVM, Naive Bayes, and a Decision Tree classifiers. Supervised learning algorithms makes sense here because we know beforehand who is a POI.  For each I checked the accurancy, percision and recall of each. I did not find a combination of features where SVM had a percision and recall greater then 3 so I disregarded it almost right away.

I then used a recursive approach to finding the most important features using feature_importances_ to rank each feature. I started with all features and then would remove features when they had an importance of 0. Each I did this accurancy, recall and percision increased after each iteration until it didn't. In my analysis we can see each iteration. 

	###First Iteration
	Naive Bayes accuracy 0.333333333333
	precision =  0.2
	recall =  0.5

	DecisionTreeClassifier
	Decision Tree accuracy 0.666666666667
	precision =  0.333333333333
	recall =  0.25

	###Second Iteration
	GaussianNB
	Naive Bayes accuracy 0.8
	precision =  1.0
	recall =  0.25

	DecisionTreeClassifier
	Decision Tree accuracy 0.866666666667
	precision =  0.75
	recall =  0.75

	###Third Iteration
	GaussianNB
	Naive Bayes accuracy 0.8
	precision =  1.0
	recall =  0.25

	DecisionTreeClassifier
	Decision Tree accuracy 0.8
	precision =  0.6
	recall =  0.75

	###Fourth Iteration
	GaussianNB
	Naive Bayes accuracy 0.8
	precision =  1.0
	recall =  0.25

	DecisionTreeClassifier
	Decision Tree accuracy 0.8
	precision =  1.0
	recall =  0.25

Looking at the results above the second iteration decision tree classifier seems most promising. An accuracy of 0.87, percision of 0.75 and recall of 0.75. In each iteration the Decision Tree Classifier out preformed GaussianNB so it made sense to work with the DecisionTreeClassifier for the rest of the analysis. 

## Feature Scaling
No feature scaling was necessary because I decided to use the Decision Tree Algorithm for the POI identifier. 

## Tune the Algorithm
I tried tuning the Decision Tree Algorithm by using different values for min_sample_split. This had little effect of the values I had previously,

	Decision Tree accuracy 0.866666666667
	precision =  0.75
	recall =  0.75 

I didn't think much of it until I tried to use KFold to split and validate algorithm. I discovered that I had recall and precision below 3 which was frustrating. 

## Validate and Evaluate 
Initially I tried to validate using 3-fold cross-validation of precision and recall scores and found that scores were under 3 so go back and change my approach to picking the right features. 

	accuracy before tuning  0.857142857143
	precision after tuning  0.4
	recall after tuning  0.666666666667

	 
	accuracy after tuning  0.892857142857
	precision after tuning  0.4
	recall after tuning  0.666666666667


	Accuracy: 0.86256	
	Precision: 0.37669	
	Recall: 0.36200	
	F1: 0.36920	
	F2: 0.36485
	Total predictions: 9000	
	True positives:  362	
	False positives:  599	
	False negatives:  638	
	True negatives: 7401
