import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, learning_curve

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer

def plot_corr(df,size=10):
    corr = df.corr()  #the default method is pearson
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr,cmap=plt.cm.Oranges)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)    
    plt.show()

def main():
    df = pd.read_csv('./data/wildfire_dataset.csv')

    le = preprocessing.LabelEncoder()
    df['STAT_CAUSE_DESCR'] = le.fit_transform(df['STAT_CAUSE_DESCR'])
    df['STATE'] = le.fit_transform(df['STATE'])
    df['DATE'] = le.fit_transform(df['DATE'])

    df = df.drop('OBJECTID',axis=1)
    df = df.drop('NWCG_REPORTING_UNIT_NAME',axis=1)
    df = df.drop('FIRE_SIZE_CLASS',axis=1)
    df = df.drop('OWNER_DESCR',axis=1)
    df = df.drop('DATE',axis=1)
    df = df.dropna()

    print(df.head())

    X = df.drop(['STAT_CAUSE_DESCR'], axis=1).values
    y = df['STAT_CAUSE_DESCR'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0) #30% for testing, 70% for training

    # clf_rf = RandomForestClassifier(n_estimators=200, max_features=int(2*np.sqrt(df.shape[1])))
    # clf_rf = clf_rf.fit(X_train, y_train)
    # y_pred = clf_rf.predict(X_test)
    # f1 = f1_score(y_test, y_pred, average='macro')
    # print ('Test F1 for Random Forests:', f1)

    # ntree = [100, 200]
    # mtry = [int(0.5*np.sqrt(df.shape[1])), int(np.sqrt(df.shape[1])), int(2*np.sqrt(df.shape[1]))]
    ntree = [50]
    mtry = [int(2*np.sqrt(df.shape[1]))]

    for i in ntree:
        for j in mtry:
            RF_clf = RandomForestClassifier(n_estimators=i, max_features=j, oob_score=True)
            RF_clf.fit(X_train, y_train)
            print(f"Random Forest Model ntree={i} mtry={j}")
            print(f"RF score {RF_clf.score(X_test, y_test)}")
            print(f"OOB score {RF_clf.oob_score_}")
            print("-----------------------------------\n")
    return 0

if __name__ == "__main__":
    main()