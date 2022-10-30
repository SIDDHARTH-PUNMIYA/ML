import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from matplotlib.colors import ListedColormap


dataset = pd.read_csv("Social_Network_Ads.csv")

X =dataset[["Age","EstimatedSalary"]]
Y=dataset["Purchased"]


X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

sc =StandardScaler()

X_Train=sc.fit_transform(X_Train)
X_Test=sc.fit_transform(X_Test)

classifier =SVC(kernel="linear")
classifier.fit(X_Train,Y_Train)


ypred =classifier.predict(X_Test)
print(ypred)

# print(confusion_matrix(Y_Test,ypred))
# print(classification_report(Y_Test,ypred))
# print(accuracy_score(Y_Test,ypred)*100)

X_set,Y_set =X_Train,Y_Train

X1,X2 =np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
print(X1)
print(X2)

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             cmap=ListedColormap(('red','green')) )


plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0] ,X_set[Y_set==j,1] , c=ListedColormap(('blue','pink'))(i) ,label=j )

plt.xlabel('AGE')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()