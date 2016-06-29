#!/usr/bin/python

import sys
import pickle
sys.path.append("./tools/")
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'to_messages'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    # general data points related to pois and non-pois
    print 'Total number of data points:'
    total = len(data_dict.keys())
    print total
    poi_total = len([person['poi'] for person in data_dict.values() if person['poi']])
    print 'Number of POIs'
    print poi_total
    print 'Number of Non-POIs'
    non_pois = total - poi_total
    print non_pois

### Task 2: Remove outliers 
features = ["poi", "salary", "bonus"]
data = featureFormat(data_dict, features)

def viewScatter(data, features):
	for point in data:
	    x = point[1]
	    y = point[2]
	    matplotlib.pyplot.scatter( x, y )
	    if point[0] == 1:
        	matplotlib.pyplot.scatter(x, y, color="r")
	matplotlib.pyplot.xlabel(features[1])
	matplotlib.pyplot.ylabel(features[2])
	matplotlib.pyplot.show()

def featureOutliers(data_dict, feature, amount):
	outliers = []
	for key in data_dict:
	    val = data_dict[key][feature]
	    # skip NaN's
	    if val == 'NaN':
	        continue
	    outliers.append((key, int(val)))

	return (sorted(outliers, key=lambda x:x[1], reverse=True)[:amount])

viewScatter(data, features)

print 'Salary outlier:'
# print top salary
print featureOutliers(data_dict, 'salary', 1)

# delete TOTAL as it wont be needed for the analysis
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

# view data again
viewScatter(data, features)
print 'Salary outliers'

# top 3 salaries
print featureOutliers(data_dict, 'salary', 3)


### Task 3: Create new feature(s)

# create new features from to_messages and from_messages
features = ["pois", "to_messages", "from_messages"]
viewScatter(data, features)

# use this to create new 
def new_list(key, attribute):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key] == "NaN" or data_dict[i][attribute] == "NaN":
            new_list.append(0.)
        elif data_dict[i][key] >= 0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][attribute]))

    return new_list

# create new features, ratio of emails from and to a POI
ratio_from_poi = new_list("from_poi_to_this_person", "to_messages")
ratio_to_poi = new_list("from_this_person_to_poi", "from_messages")

# add new features to each data point
count = 0
for i in data_dict:
    data_dict[i]["ratio_from_poi"] = ratio_from_poi[count]
    data_dict[i]["ratio_to_poi"] = ratio_to_poi[count]
    count += 1

features_list = ["poi", "ratio_from_poi", "ratio_to_poi"] 

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

viewScatter(data, features)


### Task 4: Try a varity of classifiers
def run_classifiers(features_list):
	data = featureFormat(my_dataset, features_list)
	labels, features = targetFeatureSplit(data)   

	from sklearn import cross_validation
	features_train, features_test, labels_train, labels_test = \
		cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

	# try Naive Bayes
	print '\nGaussianNB'
	clf = GaussianNB()
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)
	accuracy = accuracy_score(pred, labels_test)
	print 'Naive Bayes accuracy', accuracy
	print 'precision = ', precision_score(labels_test,pred)
	print 'recall = ', recall_score(labels_test,pred)

	# try Decision Tree
	print '\nDecisionTreeClassifier'
	clf = DecisionTreeClassifier(min_samples_split=6)
	clf.fit(features_train,labels_train)
	score = clf.score(features_test,labels_test)
	pred = clf.predict(features_test)
	print 'Decision Tree accuracy', score
	print 'precision = ', precision_score(labels_test,pred)
	print 'recall = ', recall_score(labels_test,pred)

	print '\nFeature importance: '
	importances = clf.feature_importances_
	import numpy as np
	for i in range(len(features_list)-2):
		if importances[i+1] != 0:
			print "{}: {}".format(features_list[i+1], importances[i+1])	

	return clf

print '\nfirst iteration' 
features_list = ["poi", "ratio_from_poi", "ratio_to_poi", 'salary', 
				'deferral_payments', 'total_payments', 'loan_advances', 
				'bonus', 'restricted_stock_deferred', 'deferred_income', 
				'total_stock_value', 'expenses', 'exercised_stock_options', 
				'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 
				'to_messages', 'from_poi_to_this_person', 'from_messages', 
				'from_this_person_to_poi', 'shared_receipt_with_poi']		

clf = run_classifiers(features_list)

# take important features from first iteration
print '\nsecond iteration'
features_list = ["poi", "ratio_from_poi", 'salary','total_payments', 'deferred_income', 
				'total_stock_value', 'expenses', 'long_term_incentive']

clf = run_classifiers(features_list)

# take important features from second iteration
print '\nthird iteration'
features_list = ["poi", 'salary','total_payments', 'deferred_income', 
				'total_stock_value', 'expenses']

clf = run_classifiers(features_list)

# take important features from third iteration
print '\nfourth iteration'
features_list = ["poi", 'salary','total_payments', 'deferred_income', 
				'total_stock_value', 'expenses']

clf = run_classifiers(features_list)

# take important features from fourth iteration	
print '\nfifth iteration'
features_list = ["poi", 'salary','total_payments', 'deferred_income']

clf = run_classifiers(features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. 

features_list = ["poi", 'ratio_to_poi', 'shared_receipt_with_poi', 'ratio_from_poi']

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)  

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = \
	cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)	

# check results of new features
clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred = clf.predict(features_test)
print '\nDecision Tree accuracy', score
print 'precision = ', precision_score(labels_test, pred)
print 'recall = ', recall_score(labels_test,pred)	
print '\n'

# tune by selecting different min_samples_split to see the effect on accuracy, precision and recall
for i in range(6):
	print i
	clf = DecisionTreeClassifier(min_samples_split=i+1)
	clf.fit(features_train,labels_train)
	score = clf.score(features_test,labels_test)
	pred = clf.predict(features_test)
	print 'Decision Tree accuracy', score
	print 'precision = ', precision_score(labels_test, pred)
	print 'recall = ', recall_score(labels_test,pred)	


# use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred= clf.predict(features_test)
print 'accuracy before tuning ', score
print 'precision after tuning ', precision_score(labels_test,pred)
print 'recall after tuning ', recall_score(labels_test,pred)


# try KFold with tuning applied
print '\n '
clf = DecisionTreeClassifier(min_samples_split=3)
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
print 'accuracy after tuning ', score
print 'precision after tuning ', precision_score(labels_test,pred)
print 'recall after tuning ', recall_score(labels_test,pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
clf = DecisionTreeClassifier(min_samples_split=3, criterion='entropy')
features_list = ["poi", 'ratio_to_poi', 'shared_receipt_with_poi', 'ratio_from_poi']
data = featureFormat(my_dataset, features_list)
dump_classifier_and_data(clf, my_dataset, features_list)