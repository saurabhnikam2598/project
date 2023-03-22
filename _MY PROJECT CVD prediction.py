#!/usr/bin/env python
# coding: utf-8

# # Cardio Vascular Disease Prediction

# Cardiovascular disease(CVDs) are the leading cause of death golbally,taking an estimated 17.9 miilion lives each year.
# CVD are the group of disorders 0f the heart and blood vessels and include coronary heart disease,cerebrovascular disease,rheumatic heart disease and other conditions.
# more than foue uot of five CVD deaths are due to heart and strokes,and one third of these death occurs permaturely in people under 70 yearss of age.

# # problem statement:
# 
# 
# 

# the prediction approach using Machine Learning Algorithm to classify the patient to be healthy or suffering from cardiovascular disease on the thre diffrent attributes

# # Import all libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ## import dataset 

# In[2]:


CVD=pd.read_csv("C:\\Users\\saura\\OneDrive\\Desktop\\CVD_Disease.csv")


# In[3]:


CVD.set_index("id",inplace=True)


# In[4]:


CVD = CVD.rename(columns={'TenYearCHD(Coronary heart disease )': 'HeartDisease'})


# In[5]:


CVD.shape


# In[6]:


CVD


# # # #Dataset Atribute explanation

# Features:
# 
# 
# 

# In[7]:


CVD.head()


# In[8]:


CVD.tail()


# # Data cleaning

# In[9]:


CVD.info()


# In[10]:


CVD.isna().sum()


#  creating new column ofage_group based on the age
#  
#  
#  
# Age Group will be divided as 
# 
# age <= 5:Child
# 
# 5 <Age < 18:Teen
# 
# 18 <= Age < 60: Adult
# 
# Age > 60 : Senior_Citizen
# 
# 

# In[11]:


CVD.loc[CVD['age'] <= 5, "Age_Group"] = 'Child'
CVD.loc[(CVD['age'] >5 ) & (CVD['age'] < 18), "Age_Group"] = 'Teen'
CVD.loc[CVD['age'] >=18 & (CVD['age'] < 60), "Age_Group"] = 'Adult'
CVD.loc[CVD['age'] >=60 , "Age_Group"] = 'Senior_Citizen'


# In[12]:


CVD.head()


# In[13]:


plt.figure(figsize=(15,5))
sns.boxplot(x=CVD["Age_Group"],y=CVD["BMI(Body Mass Index )"],hue = CVD["sex"])


# Repacing the missing valuesin continuos data with 
# 
# Age
# 
# TotalCholestrol
# 
# Systolic blood pressure
# 
# diastolic blood pressure
# 
# Body mass index
# 
# Heartrate
# 
# Glucose
# 
# Cigerates Per Day

# In[14]:


num_col=["age","totChol","sysBP","diaBP","BMI(Body Mass Index )","heartRate","glucose","cigsPerDay"]
for col in num_col:
    CVD[col]=pd.to_numeric(CVD[col])
    CVD[col].fillna(CVD[col].median(),inplace=True)
CVD.head(10)


# In[15]:


CVD.isna().sum()


# Repacing missing values in categorical column with mode

# categorical columnis
# 
# BPMeds
# 
# Education

# In[16]:


num_col2=["BPMeds","education"]
for col2 in num_col2:
    #CVD[col2]=pd.to_numeric(CVD[col2])
    CVD[col2].fillna(CVD[col2].mode()[0], inplace=True,)
CVD.head()


# In[17]:


CVD.isna().sum()


# # to check duplicates

# In[18]:


CVD.duplicated().sum()


# In[19]:


CVD["sex"].value_counts()


# # Exploratory Data Analytics

# In[20]:


CVD.columns


# # boxplot for continous data variable

# In[21]:


def boxplot(df,col):
    #sns.boxplot(column=[col])
    sns.boxplot(df[col],color="red")
    plt.grid(False)
    plt.show()


# In[22]:


c=["age","totChol","sysBP","diaBP","BMI(Body Mass Index )","heartRate","glucose","cigsPerDay"]


# In[23]:


for i in c:
    boxplot(CVD,i)


# # Heatmap

# finding correlation between features

# In[24]:


plt.figure(figsize = (15,10))        # Size of the figure
sns.heatmap(CVD.corr(),annot = True)
plt.show()


# # countplot

# In[25]:


categorical = ['sex', 'is_smoking', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',"education"]

for i in categorical:
    sns.countplot(CVD[i], hue = CVD['HeartDisease'])
    plt.show()


# # Continous vs continous

# In[26]:


plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
sns.scatterplot(data=CVD,x='age',y='BMI(Body Mass Index )',hue='HeartDisease')
plt.plot()
plt.grid(True)
plt.subplot(2,2,2)
sns.scatterplot(data=CVD,x='age',y='glucose',hue='HeartDisease')
plt.plot()
plt.grid(True)
plt.subplot(2,2,3)
sns.scatterplot(data=CVD,x='age',y='totChol',hue='HeartDisease')
plt.plot()
plt.grid(True)

plt.savefig('cont_vs_cont_variable')


# # categorical vs continuos

# In[27]:


sns.histplot(data=CVD,x='age',hue='diabetes',alpha = 0.5,kde = True)
plt.savefig('age vs diabetes')
plt.show()

sns.histplot(data=CVD,x='sysBP',hue='HeartDisease',alpha = 0.5,kde = True)
plt.savefig('systolic BP Vs heartDisease')
plt.show()

sns.histplot(data=CVD,x='BMI(Body Mass Index )',hue='HeartDisease',alpha = 0.5,kde = True)
plt.savefig('Body Mass Index Vs heartDisease')
plt.show()


# # feature engineering

# In[28]:


CVD.head()


# In[29]:


CVD=CVD.drop(['Age_Group'],axis=1)
CVD.head()


# In[30]:


#creating dummies for the categorical variable

categorical = ['sex', 'is_smoking', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',"education"]

dummies=pd.get_dummies(CVD[['sex', 'is_smoking', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',"education"]], drop_first=True)

dummies


# In[31]:


CVD = pd.concat([CVD, dummies], axis=1)


# In[32]:


CVD


# In[33]:


CVD.drop(['sex', 'is_smoking', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', "education"],axis=1,inplace=True)


# # data distrubution analysis

# over sampling technique

# In[34]:


CVD["HeartDisease"].value_counts()


# In[35]:


sns.countplot(x=CVD["HeartDisease"])
plt.show()


# 0 - Healthy
# 
# 1 - HeartDisease

# As you can see that there is huge imbalance in the dataset
# 
# This oversampling happens based on KNN algorithm who tries to create the new neighbours based on existing neighbour in minority class
# 
# Advantages of SMOTE
# 
# Minimize the problem of overfitting
# 
# No loss of data or useful information

# # Implementation

# In[36]:


class_count_0, class_count_1 = CVD["HeartDisease"].value_counts()
# Separate class
class_0 = CVD[CVD["HeartDisease"] == 0]
class_1 = CVD[CVD["HeartDisease"] == 1]

print('class 0:', class_0.shape)
print('class 1:', class_1.shape)
    
class_1_over = class_1.sample(class_count_0, replace=True)
CVD = pd.concat([class_0,class_1_over], axis=0)

print("total values of 1 and 0:",CVD["HeartDisease"].value_counts())

CVD["HeartDisease"].value_counts().plot(kind='bar')
plt.show()


# # Mchine Learning Models

# In[37]:


from sklearn.model_selection import train_test_split


# In[40]:


X = CVD.drop(['HeartDisease'], axis=1)

X.head()


# In[41]:


y = CVD['HeartDisease']

y.head()


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30,random_state = 100)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[43]:


from sklearn.preprocessing import StandardScaler


# In[44]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# # Decision Tree

# In[45]:


from sklearn.tree import DecisionTreeClassifier


# In[46]:


dtree = DecisionTreeClassifier()


# In[47]:


dtree.fit(X_train,y_train)


# In[48]:


predictions = dtree.predict(X_test)


# In[49]:


from sklearn.metrics import classification_report,confusion_matrix


# In[50]:


print(classification_report(y_test,predictions))


# In[51]:


print(confusion_matrix(y_test,predictions))


# In[52]:


sns.heatmap(confusion_matrix(y_test,predictions), annot = True, fmt = "d")


# # calculating accuracy

# Accuracy = TP+TN/Total observation

# Accuracy = 753+830/573+140+5+830 = 1583/1728 = 0.916 = 91%

# In[53]:


def metrics_calculation(y_testing,pred):
    confusion_mat = confusion_matrix(y_testing, pred)
    TP = confusion_mat[0,0:1]
    FP = confusion_mat[0,1:2]
    FN = confusion_mat[1,0:1]
    TN = confusion_mat[1,1:2]
    
    accuracy = (TP+TN)/((FN+FP)+(TP+TN))
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    precision = TP/(TP+FP)
    recall =  TP / (TP + FN)
    fScore = (2 * recall * precision) / (recall + precision)
    
    print("Confusion Matrix:\n",confusion_mat)
    print("Accuracy :",accuracy)
    print("Sensitivity :",sensitivity)
    print("Specificity :",specificity)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F-score:",fScore)
    


# In[54]:


metrics_calculation(y_test,predictions)


# In[55]:


from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test,predictions)

auc_score = roc_auc_score(y_test,predictions)


plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'r--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.show()


# # Random forest

# In[56]:


from sklearn.ensemble import RandomForestClassifier
randomfc = RandomForestClassifier(n_estimators=100)
randomfc.fit(X_train, y_train)


# In[57]:


randomfc_pred = randomfc.predict(X_test)


# In[58]:


print(confusion_matrix(y_test,randomfc_pred ))


# In[59]:


print(classification_report(y_test,randomfc_pred))


# In[60]:


sns.heatmap(confusion_matrix(y_test,randomfc_pred), annot = True, fmt = "d")


# In[61]:


metrics_calculation(y_test,randomfc_pred)


# In[62]:


from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test,randomfc_pred)

auc_score = roc_auc_score(y_test,randomfc_pred)


plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'r--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.show()


# # Logistic Regression

# In[63]:


from sklearn.linear_model import LogisticRegression


# In[64]:


lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
report = classification_report(y_test, lr_pred)
print(report)


# In[65]:


print(confusion_matrix(y_test,lr_pred))


# In[66]:


sns.heatmap(confusion_matrix(y_test,lr_pred), annot = True, fmt = "d")


# In[67]:


metrics_calculation(y_test,lr_pred)


# In[68]:


from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test,lr_pred)

auc_score = roc_auc_score(y_test,lr_pred)


plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'r--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.show()


# # support vector machine

# In[69]:


from sklearn.svm import SVC


# In[70]:


svc = SVC()


# In[71]:


svc.fit(X_train, y_train)


# In[72]:


y_pred_svc = svc.predict(X_test)


# In[73]:


SVM_confusion_matrix = confusion_matrix(y_test, y_pred_svc)
SVM_confusion_matrix


# In[74]:


metrics_calculation(y_test, y_pred_svc)


# In[75]:


sns.heatmap(confusion_matrix(y_test,y_pred_svc), annot = True, fmt = "d")


# In[76]:


from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test,y_pred_svc)

auc_score = roc_auc_score(y_test,y_pred_svc)


plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'r--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.show()


# # K Nearest Neighbhour

# In[77]:


from sklearn.neighbors import KNeighborsClassifier


# In[78]:


knn = KNeighborsClassifier()


# In[79]:


knn.fit(X_train, y_train)


# In[80]:


y_pred_knn = knn.predict(X_test)


# In[81]:


knn_confusion_matrix = confusion_matrix(y_test, y_pred_knn)
knn_confusion_matrix


# In[82]:


metrics_calculation(y_test, y_pred_knn)


# In[83]:


sns.heatmap(confusion_matrix(y_test,y_pred_knn), annot = True, fmt = "d")


# In[84]:


from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test,y_pred_knn)

auc_score = roc_auc_score(y_test,y_pred_knn)


plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'r--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.show()


# # comparing the Accuracy of the algorithms

# In[85]:


data={"Algorithm":["DecisonTree","RandomForest","LogisticReg","KNN","SVM"],"Accuracy":[91,95,67,69,78]}


# In[86]:


df=pd.DataFrame(data)


# In[87]:


df


# In[88]:


ax=sns.barplot(x=df["Algorithm"],y=df["Accuracy"])
ax.set_xlabel("Algorithm",size=13)
ax.set_ylabel("Accuracy",size=13)
plt.title("Accuracy Comparison",size=18)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




