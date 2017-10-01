#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 22:29:54 2017

@author: AlexChen
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic = pd.DataFrame.from_csv('titanic3.csv')


df_sliced = titanic[["survived","sex","age","pclass","fare","sibsp","parch"]]
df_sliced = df_sliced.dropna(axis=0, how='any')
sex = {'male':1,'female':0}
df_sliced['sex'] = df_sliced['sex'].map(sex)

X = np.array(df_sliced[["sex","age","pclass","fare","sibsp","parch"]])
y = np.array(df_sliced["survived"])

#titanic.head(3)

"Exploratory Data Analysis"
def EDA_1():
    sns.countplot(x="sex", hue="survived", data=titanic,palette="Set1")
    plt.show()
    sns.countplot(x="pclass", hue="survived", data=titanic,palette="GnBu_d")
    plt.show()
    sns.barplot(x="sex", y="survived", hue="pclass", data=titanic,palette="Reds")
    plt.show()
    
def EDA_2(): 
    sns.violinplot(x="survived", y="age", hue="sex", data=titanic,
               split=True, palette="Set2")
    plt.show()
    sns.swarmplot(x="survived", y="age", hue="sex", data=titanic)
    plt.show()
    sns.swarmplot(x="survived", y="age", hue="pclass", data=titanic)
    plt.show()

def linearRegplot(data,X,Y):
    #sns.lmplot(x=X,y=Y,data=data)
    sns.regplot(x=X, y=Y, data=data, color='blue', label='order 1')
    plt.show()
    sns.residplot(x=X, y=Y, data=data, color='green')
    plt.show()
#linearRegplot(titanic,"age","fare")


def boxplot(data,X,Y):
    sns.boxplot(x=X, y=Y, data=data)
    plt.show()
#boxplot(df_test1,"survived","age")

"Machine Learning"
"KNN"
def KNN_choose_K(X,y,K): 
    # Import necessary modules
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    
    # Split into training and test set, stratify the split according to the labels so that they are distributed in the training and test sets as they are in the original dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1, stratify=y)
    
    # Setup arrays to store train and test accuracies
    neighbors = np.arange(1, K)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    # Loop over different values of k
    for i, k in enumerate(neighbors):
        # Setup a k-NN Classifier with k neighbors: knn
        knn = KNeighborsClassifier(n_neighbors=k)
        # Fit the classifier to the training data
        knn.fit(X_train,y_train)    
        #Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)
        #Compute accuracy on the testing set
        test_accuracy[i] = knn.score(X_test, y_test)
    # Generate plot
    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()


def KNN_CrossValid(X,y,K,CV):
    # Import necessary modules
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    # Split into training and test set, stratify the split according to the labels so that they are distributed in the training and test sets as they are in the original dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  stratify=y)
    knn = KNeighborsClassifier(n_neighbors=K)
    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    accuracy = knn.score(X_test, y_test)   
    cv_Nfold = cross_val_score(knn, X, y, cv=CV)
    # Import necessary modules
    #print(accuracy)
    #print(cv_Nfold)
    #print(np.mean(cv_Nfold))
    return accuracy, np.mean(cv_Nfold)

def KNN_simulate(X,y,K,CV,n):
    accuracy = np.empty(n)
    CV_N_fold = np.empty(n)
    
    for i in range(0,n):
        accuracy[i], CV_N_fold[i] = KNN_CrossValid(X,y,K,CV)
    print("Accuracy without CV:",np.mean(accuracy))
    print("Accuracy with CV:",np.mean(CV_N_fold))
    #print("Accuracy without CV:",accuracy)
    #print("Accuracy with CV:",CV_N_fold)
#KNN_simulate(X,y,10,10,100)

"Logistic"
def logit_CrossValid_Plot(X,y,CV):
    from sklearn.model_selection import train_test_split
    # Import the necessary modules
    from sklearn.linear_model import LogisticRegression

    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4,random_state=1)
    # Create the classifier: logreg
    logreg = LogisticRegression()
    logreg.fit(X_train,y_train)
    
    # Import necessary modules
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score

    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = logreg.predict_proba(X_test)[:,1]
    # Compute and print AUC score
    print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
    # Compute cross-validated AUC scores: cv_auc
    cv_auc = cross_val_score(logreg, X, y, cv=CV, scoring='roc_auc')
    # Print list of AUC scores
    print("AUC scores computed using N-fold cross-validation: {}".format(cv_auc))
    print("Mean of N-fold CV: {}".format(np.mean(cv_auc)))
    
    import matplotlib.pyplot as plt
    # Import necessary modules
    from sklearn.metrics import roc_curve

    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = logreg.predict_proba(X_test)[:,1]
    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    
def logit_CrossValid(X,y,CV):
    from sklearn.model_selection import train_test_split
    # Import the necessary modules
    from sklearn.linear_model import LogisticRegression

    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
    # Create the classifier: logreg
    logreg = LogisticRegression()
    logreg.fit(X_train,y_train)
    
    # Import necessary modules
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score
    
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = logreg.predict_proba(X_test)[:,1]
    # Compute cross-validated AUC scores: cv_auc
    cv_auc = cross_val_score(logreg, X, y, cv=CV, scoring='roc_auc')
    
    return roc_auc_score(y_test, y_pred_prob), np.mean(cv_auc)
    

def logit_simulate(X,y,CV,n):
    AUC = np.empty(n)
    CV_N_fold = np.empty(n)
    
    for i in range(0,n):
        AUC[i], CV_N_fold[i] = logit_CrossValid(X,y,CV)
    print("AUC without CV:",np.mean(AUC))
    print("AUC with CV:",np.mean(CV_N_fold))
    #print("AUC without CV:",AUC)
    #print("AUC with CV:",CV_N_fold)

#logit_simulate(X,y,10,100)



