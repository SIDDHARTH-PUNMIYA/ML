import numpy as np
import pandas as pd
from sklearn.preprocessing import  StandardScaler
from sklearn.preprocessing import  LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import  LogisticRegression
data = pd.read_csv("titanic_train.csv")
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
input = data[['Pclass','Sex','Fare']].copy()

target=data.Survived
sc = StandardScaler()

finalinp =sc.fit_transform(input)
print(finalinp)

pca = PCA(n_components=2)
x=pca.fit_transform(finalinp)
print("Eigen Vectors\n",pca.components_)
print("Eigen Values\n",pca.explained_variance_)
print("Updated dataset")
print(x)

def splitdata(input,target,testsize):
    X_train,X_test,Y_train,Y_test = train_test_split(input,target,test_size=testsize)
    return X_train,X_test,Y_train,Y_test

def accuracy(X_train,X_test,Y_train,Y_test):
    lc =LogisticRegression()
    lc.fit(X_train,Y_train)
    score = lc.score(X_test,Y_test)
    return score

numpy_array_x = np.array(x)
df_x = pd.DataFrame(numpy_array_x)
df_x.head()

#original data
X_train,X_test,Y_train,Y_test = splitdata(input,target,0.3)
print(accuracy(X_train,X_test,Y_train,Y_test))

#new updated data

X_train,X_test,Y_train,Y_test = splitdata(x,target,0.3)
print(accuracy(X_train,X_test,Y_train,Y_test))