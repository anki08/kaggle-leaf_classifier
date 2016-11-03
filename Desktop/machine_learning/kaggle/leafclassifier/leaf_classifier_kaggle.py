import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

np.random.seed(42)
#data is highle colinear  found out while running lda :p
train = pd.read_csv('train.csv')
train[train.dtypes[(train.dtypes=="float64")|(train.dtypes=="int64")]
                        .index.values].hist()
print(train.head())
x_train = train.drop(['id', 'species'], axis=1).values
#to encode label

le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
#plt.scatter(x_train['margin1'],y_train)
plt.show()
test = pd.read_csv('test.csv')
test_ids = test.pop('id')
x_test = test.values
#(x-mean)/std-deviation
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)
params = {'C':[1,960,930, 940, 950,945], 'tol': [0.001]}
log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial')
clf = GridSearchCV(log_reg,params, scoring='log_loss', refit='True', n_jobs=1, cv=6)
clf.fit(x_train, y_train)
y_test = clf.predict_proba(x_test)
#print((clf.score(x_train,y_train)))
print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))
submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submission.csv')
