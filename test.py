#https://github.com/etas/SynCAN/blob/master/train_2.zip
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from genetic_selection import GeneticSelectionCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf

def nearest_spider(spider, spiders):

        spudis = list(spiders)

        try:
            pos = spudis.index(spider)
            spudis.pop(pos)
        except ValueError:
            pass

        dists = np.array([np.linalg.norm(spider - s) for s in spudis])
        m = dists.argmin()
        d = dists[m]

        return d, m

def main():
    train = pd.read_csv('dataset/CAN.csv',nrows=14000)
    train.fillna(0,inplace=True)
    print(train)
    print(train.shape)

    le = LabelEncoder()
    train['ID'] = pd.Series(le.fit_transform(train['ID']))
    print(train)

    X = train.values[:, 1:7] 
    Y = train.values[:, 0]
    print(Y)
    #X = tf.easom_function(X[0].astype(int))
    #print(X)
    alh = SwarmPackagePy.ssa(10, tf.easom_function, -10, 6, 2, 20,0.4)
    print(nearest_spider(0, X[0]))
    print(nearest_spider(1, X[1]))
    print(nearest_spider(2, X[2]))
    print(nearest_spider(3, X[3]))

    '''
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    estimator = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 0)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    hr = accuracy_score(y_test,y_pred)*100
    mr = precision_score(y_test, y_pred,average='macro') * 100
    fr = recall_score(y_test, y_pred,average='macro') * 100
    cr = f1_score(y_test, y_pred,average='macro') * 100
    print(str(hr)+" "+str(mr)+" "+str(fr)+" "+str(cr))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(str(fp)+" "+str(fn))

    X = train.values[:, 3:7] 
    Y = train.values[:, 0]
    print(Y)


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    estimator = KNeighborsClassifier()
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    hr = accuracy_score(y_test,y_pred)*100
    mr = precision_score(y_test, y_pred,average='macro') * 100
    fr = recall_score(y_test, y_pred,average='macro') * 100
    cr = f1_score(y_test, y_pred,average='macro') * 100
    print(str(hr)+" "+str(mr)+" "+str(fr)+" "+str(cr))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(str(fp)+" "+str(fn))

    X = train.values[:, 4:7] 
    Y = train.values[:, 0]
    print(Y)


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    estimator = DecisionTreeClassifier(max_features=2)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    hr = accuracy_score(y_test,y_pred)*100
    mr = precision_score(y_test, y_pred,average='macro') * 100
    fr = recall_score(y_test, y_pred,average='macro') * 100
    cr = f1_score(y_test, y_pred,average='macro') * 100
    print(str(hr)+" "+str(mr)+" "+str(fr)+" "+str(cr))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(str(fp)+" "+str(fn))

    X = train.values[:, 1:7] 
    Y = train.values[:, 0]
    print(Y)


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    estimator = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 0)
    selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=6,
                                  n_population=5,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=5,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=2,
                                  caching=True,
                                  n_jobs=-1)
    selector = selector.fit(X_train, y_train)
    y_pred = selector.predict(X_test)
    hr = accuracy_score(y_pred,y_pred)*100
    mr = precision_score(y_pred, y_pred,average='macro') * 100
    fr = recall_score(y_pred, y_pred,average='macro') * 100
    cr = f1_score(y_pred, y_pred,average='macro') * 100
    print(str(hr)+" "+str(mr)+" "+str(fr)+" "+str(cr))
    tn, fp, fn, tp = confusion_matrix(y_pred, y_pred).ravel()
    print(str(fp)+" "+str(fn))
'''
if __name__ == "__main__":
    main()
    
