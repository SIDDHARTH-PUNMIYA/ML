import pandas as pd
import  matplotlib.pyplot as plt


# ypred = m*x + C
#
m=0
c=0
L=0.01
epoch =2000

data = pd.read_csv("data.csv")
X= data["YearsExperience"]
Y = data["Salary"]
n = len(X)
for i in range(epoch):
    Ypred = m*X+c
    Dm=(-2/n)*sum(X*(Y-Ypred))
    Dn=(-2/n)*sum(Y-Ypred)
    m=m-L*Dm
    c=c-L*Dn

Ypred = m*X + c


plt.scatter(X,Y)
plt.plot(X,Ypred,color="red")
plt.show()