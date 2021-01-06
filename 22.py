import pandas as pd
import numpy as np

from sklearn import preprocessing  #数据预处理，对进行编码
from sklearn.model_selection import train_test_split  #将样本几何分割成训练集和验证集返回的是划分好的训练集和验证集
from sklearn.metrics import log_loss

from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB  #朴素贝叶斯(伯努利)
from sklearn.linear_model import LogisticRegression  #逻辑回归
from sklearn.ensemble import RandomForestClassifier  #随机森林

train = pd.read_csv('C:\\Users\\49210\Desktop\\22.csv')
test = pd.read_csv('C:\\Users\\49210\Desktop\\22.csv')

train.head()
le = preprocessing.LabelEncoder()
crime_type_encode = le.fit_transform(train['EVENT_TYPE'])


MONTH = pd.to_datetime(train['EVENT_DATE']).dt.month
MONTH = pd.get_dummies(MONTH)#月份训练
number = pd.get_dummies(train['事件数'])
SHENFEN = pd.get_dummies(train['ADMIN1'])#省份训练
train_set = pd.concat([MONTH,number,SHENFEN],axis=1)
train_set['crime_type'] = crime_type_encode

#训练样本特征因子化
MONTH_t =pd.to_datetime(train['EVENT_DATE']).dt.month
MONTH_t = pd.get_dummies(MONTH_t)
number_t = pd.get_dummies(test['事件数'])
SHENFEN_t = pd.get_dummies(test['ADMIN1'])
test_set = pd.concat([MONTH_t,number,SHENFEN_t],axis=1)

x = train_set.loc[:,train_set.columns!='crime_type']
y = train_set['crime_type']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)

model = BernoulliNB()#伯努利模型
model.fit(x_train,y_train)
y_pred = model.predict(x_test)  #预测值#将预测值与实际值进行对比，输出精确度的值
print("伯努利model accuracy: ",metrics.accuracy_score(y_test,y_pred))
model_LR = LogisticRegression(C=0.1)#逻辑回归
model_LR.fit(x_train,y_train)
y_pred = model_LR.predict(x_test)
print("逻辑回归model accuracy: ",metrics.accuracy_score(y_test,y_pred))
model_RF = RandomForestClassifier()
model_RF.fit(x_train,y_train)
y_pred = model_RF.predict(x_test)
print("随机森林model accuracy: ",metrics.accuracy_score(y_test,y_pred))#随机森林
















