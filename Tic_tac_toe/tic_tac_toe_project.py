import pandas as pd
import numpy as np

#Reading the data..
data=pd.read_csv(r"C:\Users\prake\Desktop\dsk folders\AIML\spy\Tic_tac_toe\tic-tac-toe.csv")


data.shape
#Checking if the dataset contain empty values or not...
data.isna().sum()

#Checking the values in every column...
data['TL'].value_counts()
data['TM'].value_counts()
data['TR'].value_counts()
data['ML'].value_counts()
data['MM'].value_counts()
data['MR'].value_counts()
data['BL'].value_counts()
data['BM'].value_counts()
data['BR'].value_counts()


#label encoding the dataset...
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

s=['TL','TM','TR','ML','MM','MR','BL','BM','BR']
for i in s:
    data[i]=le.fit_transform(data[i])
    

#Splitting the data into 2 ...
x=np.array(data.iloc[ : , :-1])
y=np.array(data.iloc[ : ,-1])

#splitting data into testing and training...
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=1)

xtrain.shape
xtest.shape 

ytrain.shape
ytest.shape

#By using classification ...
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=6)

model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

#Finding accuracy...
from sklearn.metrics import accuracy_score
accuracy_score(ytest, ypred)*100


a=list(map(str,input().split()))
for i in range(len(a)):
    if(a[i]=='x'):
        a[i]=2
    elif(a[i]=='o'):
        a[i]=1
    else:
        a[i]=0
a_array = np.array(a)
a_2d = a_array.reshape(1, -1)
val=model.predict(a_2d)
print(val)