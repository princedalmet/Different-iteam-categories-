# Different-iteam-categories
Using NLP to Predict categories of the items


![Range-3](https://user-images.githubusercontent.com/99526815/154831847-1ee05d4f-c6b9-4b71-92e4-c0f756f46486.jpg)


Getting Started


Objective


Categorize the items into 5 different categories Home & Kitchen, Tools & Home Improvement, Office Products, Grocery & Gourmet Food, Industrial & Scientific, Electronics using Natural Language Processing for an e-commerce firm. Given is a dataset of training data and need to predict the categories on the test data.


Importing libraries


Reading and understanding the data

Cleaning the data


We can observe that there are many columns with None. Lets remove those rows since they'll hinder our predictions.


Step by Step Detection


To help with debugging and understanding the model that provide a lot of visualizations and allow running the model step by step to inspect the output at each point. Here are a few examples:


Train-Test Split


# feature
X = df_train['title']


# label
y = df_train['category']



Model Creation



from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


Predictions


prediction1 = model1.predict(X_test)


print(confusion_matrix(y_test, prediction1))
[[  28    0   11    1    6   13]
 [   0  286   34    2    0    1]
 [   0   12 1411    7   17   52]
 [   1    3   31   83   10   41]
 [   3    2   69    9  267   16]
 [   1    0   96    8   18  481]]
 
 
 print(accuracy_score(y_test, prediction1))
0.8463576158940397


The model is having quite a good accuracy and f1 score for most of the categories, let's improve the model.


Finding best parameters for the Pipeline


GridSearchCV


grid_search.best_estimator_
Pipeline(steps=[('tfidf', TfidfVectorizer(max_df=0.25)),
                ('clf', LinearSVC(random_state=42))])
                
                
Best Parameters


grid_search.best_params_
{'tfidf__max_df': 0.25,
 'tfidf__max_features': None,
 'tfidf__ngram_range': (1, 1)}
 
 
 print(classification_report(y_test, prediction2))
                          precision    recall  f1-score   support

             Electronics       0.86      0.51      0.64        59
  Grocery & Gourmet Food       0.92      0.90      0.91       323
          Home & Kitchen       0.85      0.94      0.90      1499
 Industrial & Scientific       0.77      0.53      0.63       169
         Office Products       0.87      0.71      0.78       366
Tools & Home Improvement       0.80      0.80      0.80       604

                accuracy                           0.85      3020
               macro avg       0.85      0.73      0.78      3020
            weighted avg       0.85      0.85      0.84      3020

Getting the categories of the test data using the model 2

# feature
X_t = df_test['title']

# label
y_t = df_test['category']


df_test['category'] = getting_categories


![1](https://user-images.githubusercontent.com/99526815/154832326-fa291dee-0b70-402d-b5ba-c52b6c11f962.PNG)


So, now we have predicted the categories of the test data.

