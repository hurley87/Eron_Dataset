# Investigating Enron financial and email data
David Hurley

## Data Exploration
(https://en.wikipedia.org/wiki/Enron_Corpus)[Email Corpus Wiki] 
Before Enron filed for bankruptcy in 2001 it was one of the largest energy companies in the world employing approximately 20,000 people and claimed revenues of $111 billion just a year earlier.  This downfall was due to a highly publicized scandal. The financial and email data during this scandal are now public record. 

The purpose of this project is to build a person of interest(POI) identifier based on financial and email data that was made public. The folks at Udacity have been nice enough to put all the relevant data is a dictionary which contains 14 financial features ('salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']), 6 email features ('to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'), and wether the person was a POI or not.

A machine learning analysis would be beneficial to understand some of the causes of this type of fraud. The features that have the most impact would give us insight into the motivations behind those employees that were corrupted. Similar models may even be used in the future to detect and hopefully prevent something like this from happening again. 

Of the 146 data points 18 are labeled as a POI which is not a significant amount of POI data. Printing out the complete data set I see that there are many values that are NaN. I removed these data points as needed which is dicussed in the next section.

## Outlier Investigation
I started by visualizing salary versus bonus and there was one clear outlier. I found this data point by excluding people who had a salary of NaN and finding the max value which ended up belonging to 'TOTAL'. Obviously it made sense to exclude this data point. 

I then looked at the top salaries while excluding those salaries that were NaN and came across two key people as listed on the Enron Scandal wiki page, https://en.wikipedia.org/wiki/Enron_scandal. While the infamous Kenneth Lay and Jeffrey Skilling are outliers it made sense to include them in the analysis.  

## New Features
I decided to add two new features to the data set, the ratio of emails a person recieved from a POI (ratio_from_poi) and the ratio of emails sent to a POI (ratio_to_poi). Intuitively if most of your email exchange is with a POI you may know a little too much and end up being a POI yourself. 

## Selected Features
I found this to be the hardest part of the assignment. At first I started with only features I thought we relevant like those related to stock, earnings, and correspondance with POIs. I originally tried SVM, Naive Bayes, and a Decision Tree classifiers. Supervised learning algorithms makes sense here because we know beforehand who is a POI and we can use this information to classify accordingly. I checked the accurancy, percision and recall of each and right away I found that SVM had a percision and recall of 0 so I disregarded it almost right away. This may of been premature but I was happy with the initial results of the Decision Tree and GaussianNB classifiers so decided to stick with that.

I then used a recursive approach to find the most important features using the Decision Tree's feature_importances_ to rank each feature. I started with all features and then removed features when they had an importance of 0. Accurancy, recall and percision increased after each iteration and I continued until it didn't. Here are some results, 

	First Iteration

	Naive Bayes accuracy 0.333333333333
	precision =  0.2
	recall =  0.5

	DecisionTreeClassifier
	Decision Tree accuracy 0.666666666667
	precision =  0.333333333333
	recall =  0.25

	Second Iteration

	GaussianNB
	Naive Bayes accuracy 0.8
	precision =  1.0
	recall =  0.25

	DecisionTreeClassifier
	Decision Tree accuracy 0.866666666667
	precision =  0.75
	recall =  0.75

	Third Iteration

	GaussianNB
	Naive Bayes accuracy 0.8
	precision =  1.0
	recall =  0.25

	DecisionTreeClassifier
	Decision Tree accuracy 0.8
	precision =  0.6
	recall =  0.75

	Fourth Iteration

	GaussianNB
	Naive Bayes accuracy 0.8
	precision =  1.0
	recall =  0.25

	DecisionTreeClassifier
	Decision Tree accuracy 0.8
	precision =  1.0
	recall =  0.25

Looking at the results above the second iteration decision tree classifier seems most promising with an accuracy of 0.87, percision of 0.75 and recall of 0.75. In each iteration the Decision Tree classifier out preformed GaussianNB so it made sense to work with the DecisionTreeClassifier for the rest of the analysis. 

## Feature Scaling
No feature scaling was necessary because I decided to use the Decision Tree Algorithm. 

## Tune the Algorithm
I tried tuning the Decision Tree Algorithm by using different values for min_sample_split. This had little effect of the values I had previously,

	Decision Tree accuracy 0.866666666667
	precision =  0.75
	recall =  0.75 

I didn't think much of it until I tried to use KFold to split and validate the algorithm. I discovered that I had recall and precision below 3 which was frustrating. I had to go back to the drawing board. 

I decided I had to pick new feautures and take another shot at tuning my Decision Tree classifier. After many attempets I tried to use only features that had to do with POI correspondance, ie 'ratio_to_poi', 'shared_receipt_with_poi', 'ratio_from_poi'. At the end of the day this was a scandal and intuitively it makes sense this correspondance would have more impact then financial features. Running with my Decision Tree classifier I had these initial results,

	Decision Tree accuracy 0.884615384615
	precision =  1.0
	recall =  0.25

Right away I saw an improvement in accuracy which was encouraging but recall was still less then 3. I then tried tuning the alogorithm but trying different min sample splits. The best result was 3,

	Decision Tree accuracy 0.884615384615
	precision =  0.666666666667
	recall =  0.5

## Validate and Evaluate 
As I said before I used KFold to split and validate the algorithm which had percision and recall above 3. Validating with KFold was really important here because my initial algorithm did not generalize well and lead me to go back and find new features.

 After tuning with the new features there was certainly an inprovement in accurancy,

	accuracy before tuning  0.857142857143
	precision after tuning  0.4
	recall after tuning  0.666666666667

	accuracy after tuning  0.892857142857
	precision after tuning  0.4
	recall after tuning  0.666666666667


I then started using the tester.py file to test my work. I trying tuning my algorithm again to see if I could make any improvements. One improvement that ended up working was addng criterion='entropy' to my classifier. I went back to see if I could reproduce this with my training data and didn't see a difference. I guess more data made a difference. Here are my final results below,

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

Overall I'm happy with the results because both percision and recall were above 3. Without much experience it's hard to say if these numbers are actually good and my guess is that in the wild they probably are not. I'm sure  they could always be better. More data would be helpful as I only had a chance to look at 18 POIs which may not have been enough. It would have been interesting to redo the analysis indirectly and build a non-POI identifier. Then I would have much more data and there's a chance I would be able to identify a non-POI more accurately.

It would be interesting to look at the text included in the emails as that would be a strategy to get more data on POI correspondance. I'm sure there exist distinct patterns in POI emails like certain phrases people use when they are trying to commit fraud or stay ahead of the SEC.  

As I said before choosing the right features was the hardest part. I know there's a better way to do it programmically because it was my initiuition that ended up leading to the features I finally went with. It would be nice to try every combination available but with so many possibilities that would take to long. 






