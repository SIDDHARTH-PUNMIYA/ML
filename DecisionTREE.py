import pandas as pd
from  sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  train_test_split
from  sklearn import tree

dataset = pd.read_csv("csv/heart.csv")
X = dataset.iloc[:,:13]
Y =dataset["target"]


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.3,random_state=10)

clf_gini = DecisionTreeClassifier(criterion="gini",random_state=10,min_samples_leaf=5,max_depth=3)

clf_gini.fit(X_train,Y_train)
ypred = clf_gini.predict(X_test)

print(ypred)
print(accuracy_score(Y_test,ypred)*100)

