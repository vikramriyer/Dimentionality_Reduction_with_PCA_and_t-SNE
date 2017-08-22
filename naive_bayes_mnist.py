import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

X = pd.read_csv('./datasets/train.csv')
y = X['label']
X.drop('label', inplace=True, axis=1)

'''
Xt = pd.read_csv('./datasets/test.csv')
yt = X['label']
Xt.drop('label', inplace=True, axis=1)
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
#Xt_train, Xt_test, yt_train, yt_test = train_test_split(Xt, yt, random_state=0, test_size=0.2)

clf = GaussianNB()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print accuracy_score(pred, y_test)