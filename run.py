import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


# Read in Data Set
# Raw Data Set can be found at https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
headers = ['sepal_length','sepal_width', 'petal_length','petal_width','iris_class']
df = pd.read_csv('iris.data',names=headers,index_col=False)

# Create train/test
X = df.drop(['iris_class'], axis=1)
Y = df['iris_class']
rows = np.random.choice(len(df), replace=False, size=int(round(len(df)*.80)))
x_train, y_train = X.ix[rows],Y.ix[rows]
x_test,y_test = X.drop(rows),Y.drop(rows)



params = {'n_estimators': 200, 'max_depth': 3,'learning_rate': 0.1}
model = GradientBoostingClassifier(**params)
model.fit(x_train, y_train)
# predict class labels
pred = model.predict(x_test)

# score on test data (accuracy)
acc = model.score(x_test, y_test)
print('ACC: %.4f' % acc)


