import  pandas as pd
import  numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder,OrdinalEncoder
import  matplotlib.pyplot as plt

data = pd.read_csv("cars.csv")

le = LabelEncoder()
data["transmission"] = le.fit_transform(data["transmission"])
oe = OrdinalEncoder(categories=[['First Owner','Second Owner', 'Fourth & Above Owner', 'Third Owner'
 ,'Test Drive Car']] ,dtype=int)
data[["owner"]] = oe.fit_transform(data[["owner"]] )

X = data[["year_bought","km_driven","transmission","owner"]]
Y =data["selling_price"]
X.insert(0, 'x0', 1)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.7,test_size=0.3,random_state=10)
X_train=X_train.values
Y_train=Y_train.values



# (XT*X)-1*XT*Y
def cal_thetha(X,Y):
 return np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),Y)


thetha= cal_thetha(X_train,Y_train)
print("Parameter of MLR")
print(thetha.shape)

def predict(X,thetha):
 return np.matmul(X,thetha)
ypred = predict(X_test.values,thetha)

print(ypred)


plt.figure(figsize=(5,4))
ax= plt.axes()
ax.scatter(range(len(Y_test)),Y_test)
ax.scatter(range(len(Y_test)),ypred)
ax.ticklabel_format(style='plain')
plt.legend(['Actual','Predicted'])
plt.show()
