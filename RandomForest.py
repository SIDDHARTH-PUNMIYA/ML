import  pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score,classification_report,confusion_matrix

data = pd.read_csv("csv/possum.csv")

data = data.dropna() #to drop null rows

X = data.drop(["case", "site", "Pop", "sex"], axis=1)
y = data["sex"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
rfc =RandomForestClassifier(n_estimators=50,random_state=40)
rfc.fit(X_train,y_train)

pred = rfc.predict(X_test)

print(pred)
print(classification_report(y_test,pred))
print(accuracy_score(y_test,pred)*100)
print(confusion_matrix(y_test,pred))