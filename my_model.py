#!/usr/bin/python

import os

import matplotlib
matplotlib.use('Agg')
import pylab as pl
import numpy as np
import pandas as pd

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print train_df.shape, test_df.shape
    print train_df.columns
    print test_df.columns

    xtrain = train_df.values[:,:-1]
    ytrain = train_df.values[:,-1]
    xtest = test_df.values[:,1:]
    ytest = test_df.values[:,0]

    print xtrain.shape, ytrain.shape, xtest.shape, ytest.shape
    return xtrain, ytrain, xtest, ytest

def score_model(model, xtrain, ytrain):
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(xtrain,
                                                                     ytrain,
                                                                     test_size=0.4, random_state=randint)
    model.fit(xTrain, yTrain)

    return model.score(xTest, yTest)

def compare_models(xtraindata, ytraindata):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.lda import LDA
    from sklearn.qda import QDA

    classifier_dict = {
                #'linSVC': LinearSVC(),
                #'kNC5': KNeighborsClassifier(),
                #'kNC6': KNeighborsClassifier(6),
                #'SVC': SVC(kernel="linear", C=0.025),
                #'DT': DecisionTreeClassifier(max_depth=5),
                'RF200': RandomForestClassifier(n_estimators=200, n_jobs=-1),
                'RF400': RandomForestClassifier(n_estimators=400, n_jobs=-1),
                'RF800': RandomForestClassifier(n_estimators=800, n_jobs=-1),
                'RF1000': RandomForestClassifier(n_estimators=1000, n_jobs=-1),}
                #'Ada': AdaBoostClassifier(),
                #'Gauss': GaussianNB(),
                #'LDA': LDA(),
                #'QDA': QDA(),
                #'SVC2': SVC(),}

    results = {}
    for name, model in classifier_dict.items():
        print name
        results[name] = score_model(model, xtraindata, ytraindata)
        print name, results[name]
    print '\n\n\n'
    for name, result in sorted(results.items(), key=lambda x: x[1]):
        print name, result


def prepare_submission(model, xtrain, ytrain, xtest, ytest):
    model.fit(xtrain, ytrain)
    ytest2 = model.predict(xtest)
    ids = ytest

    df = pd.DataFrame({'id': ids, 'hand': ytest2}, columns=('id','hand'))
    df.to_csv('submission.csv', index=False)

    return

if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = load_data()

    pca = PCA(n_components=4)
    x_pca = np.vstack([xtrain, xtest])
    print x_pca.shape
    pca.fit(xtrain)

    xtrain = pca.transform(xtrain)
    xtest = pca.transform(xtest)

    #compare_models(xtrain, ytrain)
    model = RandomForestClassifier(n_estimators=400, n_jobs=-1)
    print 'score', score_model(model, xtrain, ytrain)
    print model.feature_importances_
    prepare_submission(model, xtrain, ytrain, xtest, ytest)
