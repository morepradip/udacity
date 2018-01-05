
# Understanding the Dataset and Questions
## Data Exploration

Goal of the project is to identify persons of interest in the Enron fraud. Persons of interest are the people who were responsible for the scandal. This was biggest corporate fraud in the American history at that time. In 2001 , it was big news India too due to Enron's association with Dabhol_Power_Company. I was in 11th standard at that time and was regularly following updates about the events related to Dabhol_Power_Company which was build by the Enron. So, personally it is quite interesting for me to see who were responsible for the fraud. I think Machine Learning can be quite helpful in predicting the person of interests. Machine learning has ability to learn from data and make predictions on the data. The fraud of such large scale could not have happened overnight. There might large exchange of information, financial manipulation that has happened. In such situation Machine learning can be quite helpful in getting hidden insight on the data and predicting the persons of the interest.

#### total number of data points
```python
print df.shape
```
    (146, 21)

There are total 146 rows and 21 features/variables.

#### allocation across classes (POI/non-POI)
```python
print pd.value_counts(df.poi.values, sort=False)
```

    False    128
    True      18
    dtype: int64    

There EW 18 Persons of Interest (PoI) and and 128 non PoI's

#### number of features used

```python
print len(df.columns)
```
    21   

There are total 21 features in the data set

#### feature names
```python
for feature in df.columns.values:
    print feature
```

    bonus
    deferral_payments
    deferred_income
    director_fees
    email_address
    exercised_stock_options
    expenses
    from_messages
    from_poi_to_this_person
    from_this_person_to_poi
    loan_advances
    long_term_incentive
    other
    poi
    restricted_stock
    restricted_stock_deferred
    salary
    shared_receipt_with_poi
    to_messages
    total_payments
    total_stock_value



## Outlier Investigation

Let's examine the names of the people provided in the dataset. We can see that row named `TOTAL` and `THE TRAVEL AGENCY IN THE PARK` are not useful. These two rows have been removed.

With Outlier (`TOTAL`)     |  Without Outlier (`TOTAL`)
:-------------------------:|:-------------------------:
![](WithOutlier.png)       |  ![](WithoutOutlier.png)


After carefully examining field values and verifying with the pdf file `(enron61702insiderpay.pdf)` provided, we can find some of the mistakes associated with the data with two persons `BHATNAGAR SANJAY` and `BELFER ROBERT`. Those mistakes have been fixed in the data.

```python
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
```

# Optimize Feature Selection/Engineering

## Create New Features

Fraud of such scale can be only created by highly influential people in the organization. These people might have huge interest in making money out of this scandal. Their total cost to the company might throw some light on identifying the persons of interest. For this reason I have engineered new financial feature named `cost_to_compnay` This feature is sum of `total_payments` and `total_stock_value`.

## Intelligently select features

Feature selection is one of the important step in the preprocessing. It is used for simplification of the model, reduce the training time and to reduce the over fitting. The top features were selected by `SelectKBest` method. This is univariate feature selection. This method selects the best `k` features and removes the rest of them. The features and their ranking based on score is given below.  We can see that the new feature `cost_to_compnay` features at the 5th rank. **In the context of this project I have used `GridSearchCV` to tune for k from SelectKBest**.

| Feature_Name        | Feature_Score          |
| ------------- |---------------|
| exercised_stock_options      | 24.815080  |
| total_stock_value      | 24.182899       |  
| bonus | 20.792252     |    
| salary |      18.289684 |  
| cost_to_company |   17.808791  |
| deferred_income |      11.458477 |
| long_term_incentive |      9.922186  |
| restricted_stock |     9.212811  |
| total_payments|      8.772778  |
| shared_receipt_with_poi|       8.589421  |

## Properly scale features

In this analysis the features scaling has not been deployed.

# Pick and Tune an Algorithm

## Pick an algorithm

I used Naïve Bayes, Support Vector Machines and Decision Tree algorithms. It was difficult to tune the SVC algorithm, I could not find the optimal solution  with the support vector machine. I could find the desired precision and recall score with the decision tree algorithm. In the final analysis the decision tree algorithm has been used.

## Discuss parameter tuning and its importance

Parameter tuning is one of the important step in implementing machine learning. Algorithms can not tune or learn parameters by their own. They have to be tuned manually like knob of machine to avoid over fitting or high variance. In scikit-learn these parameters are passed as an arguments to the constructor of the estimator.

## Tune the algorithm

Here `GridSearchCV` module is used to find out the optimized combination of the parameters. For each algorithm following parameters and their optimal values are provided. I also used `Pipeline` which allows to systematically pass several steps together while setting different parameters.

1. Naïve Bayes `None`
2. Support Vector Machine `C`, `gamma`, `kernel`
3. Decision Tree
`{"criterion" : ['gini', 'entropy'],
"min_samples_split": [2, 3, 4, 5],
"min_samples_leaf": [1, 2, 3],
"max_depth" : [None, 1, 2],
"class_weight" : ['balanced', None]}`

# Validate and Evaluate

## Usage of Evaluation Metrics

For evaluation I used accuracy, precision, recall and F1 score.  

The accuracy score calculates the fraction of correct predictions, so mathematically it is  ratio of number of correct prediction to total number of predictions.    

Precision score is `tp/(tp+fp)` where `tp` is true positive and `fp` is false positive. Precision is nothing but how many instances are relevant out of all classified instances.
In the context of this project precision may not be the right criteria to evaluate the algorithm performance. If at all we label a non PoI a PoI we are not going to prosecute them immediately. Algorithm predicted PoI's will be only prosecuted by following proper due diligence.

Recall score is `tp/(tp+fn)` where `tp` is true positive and `fn` is false negative. Intuitively it is the probability that the relevant PoIs are predicted by the algorithm. Recall should be as high as possible as we do not want to miss any PoIs. The decision tree gives high recall so it has been used in the final analysis.

F1 Score is the harmonic mean   `F1 = 2 * (precision * recall) / (precision + recall)`. This is the weighted average of the precision and recall.

It was difficult to find the optimal solution with the SVM algorithm. With Naïve Bayes algorithm there are not much parameters to tune. Finally I could achieve the desired recall and precision with Naïve Bayes and Decision Tree algorithm.



## Discuss validation and its importance

In validation we evaluate the performance of the machine learning algorithm. We should not use the same data for training and testing. If we use same data for training and testing then accuracy will be higher and this condition is called as over fitting,  We will not be able to use overfitted model in unseen data. So, it is important that we test and validate  model before we use it for prediction.

## Validation Strategy

I have split the data in training and test set. Out of entire data, `20%` of the data has been used for testing and remaining `80%` data is used for the training. The training data is also used for cross validation using `StratifiedShuffleSplit`. In stratified shuffle split method, the testing data is divided in k folds. The proportion of features and labels in training and validation remains same. Here I am using 1000 folds as used in `tester.py`.

## Algorithm Performance

After running the given `tester.py` I got the following scores of evaluation matrix.

** Comparisons of Validation Parameters for Decision Tree (DT) and Naïve Bayes (NB) **

|  Parameter | DT Score| NB Score|   
|------------|---------|---------|
|Accuracy| 0.72633|0.84700  |
|Precision| 0.31276|0.40729|
|Recall| 0.87900|0.32400|
|F1| 0.46136|0.36090|

# References

- [https://en.wikipedia.org/wiki/Enron_scandal](https://en.wikipedia.org/wiki/Enron_scandal)
- [https://en.wikipedia.org/wiki/Dabhol_Power_Company](https://en.wikipedia.org/wiki/Dabhol_Power_Company)
- [https://en.wikipedia.org/wiki/Machine_learning](https://en.wikipedia.org/wiki/Machine_learning)
- [https://discussions.udacity.com/t/confused-about-feature-selection-and-outliers/25018/12](https://discussions.udacity.com/t/confused-about-feature-selection-and-outliers/25018/12)
- [https://discussions.udacity.com/t/outlier-removal/7446](https://discussions.udacity.com/t/outlier-removal/7446)
-[http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [https://stats.stackexchange.com/questions/62621/recall-and-precision-in-classification](https://stats.stackexchange.com/questions/62621/recall-and-precision-in-classification)
- [https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)
- [https://stats.stackexchange.com/questions/19048/what-is-the-difference-between-test-set-and-validation-set](https://stats.stackexchange.com/questions/19048/what-is-the-difference-between-test-set-and-validation-set)
- [https://discussions.udacity.com/t/recall-undefined-b-c-no-true-positive-predictions-using-rbf-kernel/157936/3](https://discussions.udacity.com/t/recall-undefined-b-c-no-true-positive-predictions-using-rbf-kernel/157936/3)
- [https://en.wikipedia.org/wiki/Precision_and_recall](https://en.wikipedia.org/wiki/Precision_and_recall)
- [https://en.wikipedia.org/wiki/Overfitting](https://en.wikipedia.org/wiki/Overfitting)
