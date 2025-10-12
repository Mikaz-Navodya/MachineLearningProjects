from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from Decision_Tree import decision_tree


data = datasets.load_breast_cancer()
X,y = data.data, data.target

X_train, X_test, y_Train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

classifier = decision_tree(max_depth=10)
classifier.fit(X_train,y_Train)
predictions = classifier.predict(X_test)

def accuracy(y_test,y_pred):
    return np.sum(y_test==y_pred)/len(y_test)

acc = accuracy(y_test,predictions)
print(acc)