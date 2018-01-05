#!/usr/bin/python

from __future__ import division
import pandas as pd
import seaborn as sns

from time import time
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier


#preprocessing modules
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


#algorithims modules
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#cross-validation modules
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# other modules

from convert_feature_data_type import convert_datatype


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi'] # You will need to use more features

financial_features_list = ['salary', 
                           'bonus', 
                           'total_payments',
                           'deferral_payments',
                           'deferred_income', 
                           'director_fees',
                           'expenses', 
                           'exercised_stock_options', 
                           'loan_advances', 
                           'long_term_incentive',
                           'restricted_stock', 
                           'restricted_stock_deferred',
                           'total_stock_value'
                           ]  

email_features_list = ['from_messages', 
                       'from_poi_to_this_person',
                       'from_this_person_to_poi',
                       'shared_receipt_with_poi', 
                       'to_messages'
                       ]

features_list = features_list + financial_features_list + email_features_list

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

df = pd.DataFrame(data_dict).transpose()

# number of persons and features
print "Number of Persons and Features", df.shape


# convert the data types to appropriate pandas data type for further investagation
df = convert_datatype(df)

print "Number of Features", df.columns.values

print "Persons in data set", df.index.values

# scatter plot to check for any outliers
print sns.lmplot(x = 'salary',y = 'bonus', data = df, fit_reg = True) 

#Persons with maximum salary
print df[df['salary'] == df['salary'].dropna().max()]


# remove outliers 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK'
del data_dict["TOTAL"]
del data_dict["THE TRAVEL AGENCY IN THE PARK"]

df = pd.DataFrame(data_dict).transpose()

df = convert_datatype(df)

# scatter plot to check for outliers after removing 'TOTAL'
print sns.lmplot(x = 'salary',y = 'bonus', data = df, fit_reg = True) 
    
# further investigate the person with maximum salary
print df[df['salary'] == df['salary'].dropna().max()]

# check the 3 persons with largest salary
print df.nlargest(3, 'salary')

# manual examination of the financial data for outlier removal

print df[['salary', 'poi']]
print df[['bonus', 'poi']]
print df[['deferral_payments', 'poi']]
print df[['deferred_income', 'poi']]
print df[['director_fees', 'poi']]
print df[['total_payments', 'poi']]
print df[['loan_advances', 'poi']]
print df[['restricted_stock_deferred', 'poi']]
print df[['total_stock_value', 'poi']]
print df[['expenses', 'poi']]
print df[['other', 'poi']]
print df[['long_term_incentive', 'poi']]
print df[['restricted_stock', 'poi']]


print df.loc["BHATNAGAR SANJAY"]
print df.loc["BELFER ROBERT"]

# correct the data of Mr. Sanjay Bhatnagar (sb)

sb_total_stock_value = 15456290
sb_restricted_stock_deferred = -2604490
sb_restricted_stock = 2604490
sb_exercised_stock_options = 15456290
sb_total_payments = 137864
sb_director_fees = 'NaN'
sb_expenses = 137864

data_dict["BHATNAGAR SANJAY"]["total_stock_value"] = sb_total_stock_value
data_dict["BHATNAGAR SANJAY"]["restricted_stock_deferred"]  = sb_restricted_stock_deferred
data_dict["BHATNAGAR SANJAY"]["restricted_stock"] = sb_restricted_stock
data_dict["BHATNAGAR SANJAY"]["exercised_stock_options"] = sb_exercised_stock_options
data_dict["BHATNAGAR SANJAY"]["total_payments"]  = sb_total_payments
data_dict["BHATNAGAR SANJAY"]["director_fees"]  = sb_director_fees
data_dict["BELFER ROBERT"]["expenses"]  = sb_expenses

# coorect the data of Mr. Robert Belfer (rb)

rb_total_stock_value = 'NaN'
rb_restricted_stock_deferred = -44093
rb_restricted_stock = 44093
rb_exercised_stock_options = 'NaN'
rb_total_payments = 3285
rb_director_fees = 102500
rb_expenses = 3285
rb_deferred_income = -102500

data_dict["BELFER ROBERT"]["total_stock_value"] = rb_total_stock_value
data_dict["BELFER ROBERT"]["restricted_stock_deferred"]  = rb_restricted_stock_deferred
data_dict["BELFER ROBERT"]["restricted_stock"]  = rb_restricted_stock
data_dict["BELFER ROBERT"]["exercised_stock_options"]  = rb_exercised_stock_options
data_dict["BELFER ROBERT"]["total_payments"]  = rb_total_payments
data_dict["BELFER ROBERT"]["director_fees"]  = rb_director_fees
data_dict["BELFER ROBERT"]["expenses"]  = rb_expenses
data_dict["BELFER ROBERT"]["deferred_income"]  = rb_deferred_income


print df.shape

### Remove features which are not useful in predicting persons of interests

# delete email_address feature as it not adding any information
for key, value in data_dict.iteritems():
    del value['email_address']


### Task 3: Create new feature(s)
    
# new financial feature cost_to_compnay    
    
df['cost_to_company'] = df['total_payments'] + df['total_stock_value']

features_list.append('cost_to_company')

#df.replace(to_replace='NaN', value=np.NaN, inplace=True)

data_dict = df.to_dict('index')


### Store to my_dataset for easy export below.

my_dataset = data_dict

# replace numpy nan with NaN to avoid error

for key, value in my_dataset.iteritems():
    for key1, value1 in value.iteritems():
        if pd.isnull(my_dataset[key][key1]):
            my_dataset[key][key1] = 'NaN'
        else:
            continue
    

my_features_list = ['poi',
                    'exercised_stock_options',
                    'total_stock_value',
                    'bonus',
                    'salary',
                    'cost_to_company',
                    'deferred_income',
                    'long_term_incentive',
                    'restricted_stock',
                    'total_payments',
                    'shared_receipt_with_poi'
                    ]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


#univeriate feature selection

selector = SelectKBest(k = 11)

selector.fit_transform(features, labels)

feature_indices = selector.get_support(indices = True)

features_selected_kbest = [features_list[i+1] for i in feature_indices]


best_features_name_score = zip(features_list[1:], selector.scores_)

best_features_name_score = pd.DataFrame(best_features_name_score, 
                                    columns=['Feature_Name', 'Feature_Score'],
                                    index = None)

best_features_name_score = best_features_name_score.sort_values(['Feature_Score'], ascending=False)

print "-------------------------------------------------"
print "---------* Best Features and Scores *------------"
print "-------------------------------------------------"
print best_features_name_score
print "-------------------------------------------------"
print "-------------------------------------------------"



features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=42)



### Task 4: Try a varity of classifiers


### Task 4.1 Naive Bayes Algorithm

folds = 80

shuffle = StratifiedShuffleSplit(labels_train, folds, random_state = 42)



#gnb_pipe = Pipeline([
#        ('kbest', SelectKBest()),
#        ('classification', GaussianNB())
#        ])
#
#gnb_parameters = {"kbest__k" : range(12,14)
#                  }
#t0 = time()
#
#gnb_clf = GridSearchCV(estimator = gnb_pipe, param_grid = gnb_parameters, cv = shuffle)
#
#gnb_clf.fit(features_train,labels_train)
#
#print "Naive Bayes search and training time: ", round(time()-t0, 3), "s"
#
#t1 = time()
#
#pred = gnb_clf.predict(features_test)
#
#print "Naive Bayes prediction time: ", round(time()-t1, 3), "s"
#
#print "-------------------------------------------------"
#print "---------* Naive Bayes GaussianNB *--------------"
#print "-------------------------------------------------"
#
#print "Naive Bayes GaussianNB Classifier Accuracy Score", round(accuracy_score(pred, labels_test),2)
#print "Naive Bayes GaussianNB Classifier Precision Score", round(precision_score(pred, labels_test),2)
#print "Naive Bayes GaussianNB Classifier Recall Score", round(recall_score(pred, labels_test),2)
#print "Naive Bayes GaussianNB Classifier F1 Score", round(f1_score(pred, labels_test),2)
#
#print "-------------------------------------------------"
#print "-------------------------------------------------"
#
#print gnb_clf.best_params_

#clf = gnb_clf.best_estimator_

### Task 4.2 Support Vector Machine

#svm_pipe = Pipeline([
#        ('kbest', SelectKBest()),
#        ('reduce_dim', PCA()), 
#        ('classification', SVC())        
#        ])
#
#svm_parameters = {"kbest__k" : [12],
#        "reduce_dim__n_components" : [4], 
#        "classification__C" : [10],
#        "classification__kernel": ['linear'],
#        "classification__gamma" : [0.001]
#        }
#
#t0 = time()
#    
#grid_search = GridSearchCV(estimator = svm_pipe, param_grid = svm_parameters, cv = shuffle)
#
#grid_search.fit(features_train,labels_train)
#
#print "SVM search and training time", round(time() - t0, 4), "s"
#
#t1 = time()
#
#pred = grid_search.predict(features_test)
#
#print "SVM Prediction time: ", round(time() - t1, 4), "s"
# 
#print "-------------------------------------------------"
#print "---------* Support Vector Machine Classifier *------------"
#print "-------------------------------------------------"
#
#print "Support Vector Machine Classifier Accuracy Score", round(accuracy_score(pred, labels_test),3)
#print "Support Vector Machine Classifier Precision Score", round(precision_score(pred, labels_test),3)
#print "Support Vector Machine Classifier Recall Score", round(recall_score(pred, labels_test),3)
#print "Support Vector Machine Classifier F1 Score", round(f1_score(pred, labels_test),3)
#
#print "-------------------------------------------------"
#print "-------------------------------------------------"
#
#print grid_search.best_params_
#print grid_search._estimator_type
#
#clf = grid_search.best_estimator_
#
#
#best_parameters = grid_search.best_estimator_.get_params()
#for param_name in sorted(svm_parameters.keys()):
#    print '\t%s: %r' % (param_name, best_parameters[param_name])


### Task 4.3 Decision Tree Classifier Algorithm

dt_pipe = Pipeline([
        ('kbest', SelectKBest()),
        ('classification', DecisionTreeClassifier())
        ])


dt_parameters = {"kbest__k" : [12, 13, 14],
        "classification__criterion" : ['gini', 'entropy'],
        "classification__min_samples_split": [2, 3, 4, 5],
        "classification__min_samples_leaf": [1, 2, 3],
        "classification__max_depth" : [None, 1, 2],
        "classification__class_weight" : ['balanced', None]
        }

t0 = time()
    
grid_search = GridSearchCV(estimator = dt_pipe, param_grid = dt_parameters, 
                           cv = shuffle, scoring = 'f1')

grid_search.fit(features_train,labels_train)

print "Decision Tree search and training time", round(time() - t0, 4), "s"

t1 = time()

pred = grid_search.predict(features_test)


print "Decision Tree Prediction time: ", round(time() - t1, 4), "s"
 
print "-------------------------------------------------"
print "---------* Decesion Tree Classifier *------------"
print "-------------------------------------------------"

print "Decesion Tree Classifier Accuracy Score", round(accuracy_score(pred, labels_test),3)
print "Decesion Tree Classifier Precision Score", round(precision_score(pred, labels_test),3)
print "Decesion Tree Classifier Recall Score", round(recall_score(pred, labels_test),3)
print "Decesion Tree Classifier F1 Score", round(f1_score(pred, labels_test),3)

print "-------------------------------------------------"
print "-------------------------------------------------"

print grid_search.best_params_
print grid_search._estimator_type

clf = grid_search.best_estimator_


best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(dt_parameters.keys()):
    print '\t%s: %r' % (param_name, best_parameters[param_name])



### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

print "-------------------------------------------------"
print "---------* Classification Report *---------------"
print "-------------------------------------------------"

t0 = time()

test_classifier(clf, my_dataset, features_list)

print "-------------------------------------------------"

print "Tester Prediction Time: " , round(time()-t0, 4), "s"