#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install niapy')
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# importing utility modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
# importing machine learning models for prediction
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
# importing voting classifer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import cv2
import os
import glob
import math
from scipy import signal
from pywt import dwt2
import skimage
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import VGG19
from keras.models import Model
from pickle import dump

from builtins import range, input
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


from builtins import range, input
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import cv2
import os
import glob
import math
from scipy import signal
from pywt import dwt2
import skimage
from keras.applications.vgg19 import decode_predictions
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
from pickle import dump
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization
from niapy.algorithms.basic import CatSwarmOptimization
from sklearn.model_selection import cross_val_score,KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# all imports
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.datasets import  make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[2]:


df = pd.read_csv("C:/Users/Tuhin/Desktop/Nomophobia/Nomophobia_Initial1.CSV")


# In[3]:


dataset = df


# In[4]:


# Standard plotly imports
from chart_studio import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)


# In[5]:


#Time spent vs Gender
plt.figure(figsize=(14,8))
sns.set(font_scale=1.35)
sns.histplot(x='Time_spent_on_Mobile_per_day', data=df, kde=True, hue='Gender')


# In[6]:


#Time spent vs Addiction Level
plt.figure(figsize=(14,8))
sns.set(font_scale=1.35)
sns.histplot(x='Time_spent_on_Mobile_per_day', data=df, kde=True, hue='Addiction_Level')


# In[8]:


#Histogram Time spend Gender
plt.figure(figsize=(14,8))
sns.set(font_scale=1.35)
sns.histplot(data=df, x="Time_spent_on_Mobile_per_day", hue="Gender")


# In[9]:


#Histogram Time spend and Social Media
plt.figure(figsize=(14,8))
sns.set(font_scale=1.35)
sns.histplot(data=df, x="Time_spent_on_Mobile_per_day", hue="Social_Media_User")


# In[10]:


#Histogram Time spend and Gamer
plt.figure(figsize=(14,8))
sns.set(font_scale=1.35)
sns.histplot(data=df, x="Time_spent_on_Mobile_per_day", hue = 'Gaming_User')


# In[11]:


plt.figure(figsize=(14,8))
sns.set(font_scale=1.35)
sns.boxenplot(data=df, x="Time_spent_on_Mobile_per_day", y="Addiction_Level", hue = 'Gaming_User')


# In[12]:


#Scatter plot of Time spend Vs Age
plt.figure(figsize=(14,8))
sns.set(font_scale=1.35)
sns.scatterplot(data=df, y="Time_spent_on_Mobile_per_day", x="Age", hue = 'Gender')


# In[13]:


#Scatter plot of Time spend Vs Academician
plt.figure(figsize=(14,8))
sns.set(font_scale=1.35)
sns.scatterplot(data=df, y="Time_spent_on_Mobile_per_day", x="Age", hue = 'Academician')


# In[14]:


dataset = df


# In[15]:


#Encoding data
labelDict = {}
for feature in dataset:
    le = preprocessing.LabelEncoder()
    le.fit(dataset[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    dataset[feature] = le.transform(dataset[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] =labelValue
    
for key, value in labelDict.items():     
    print(key, value)


# In[16]:


#correlation matrix
corrmat = dataset.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()

#treatment correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.index
f, ax = plt.subplots(figsize=(12, 9))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.32)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 14}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[17]:


dataset


# # Data Load For ML

# In[18]:


# Scaling Age
scaler = MinMaxScaler()
dataset['Age'] = scaler.fit_transform(dataset[['Age']])
dataset.head()


# In[19]:


# define X and y
feature_cols = ['Age', 'Gender', 'Occupation', 'Time_spent_on_Mobile_per_day', 'Addiction_Level', 'Social_Media_User', 'Gaming_User', 'Academician']
X = dataset[feature_cols]
y = dataset.Addiction_Level


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[21]:


model1 = SVC()
model2 = RandomForestClassifier()
model3 = DecisionTreeClassifier()
model4 = GaussianNB()
model5 = XGBClassifier()
model6 = KNeighborsClassifier()
model7 = LogisticRegression()


# In[22]:


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    #Build a forest and compute the feature importances
    
    
#---------Extra Tree
forest = ExtraTreesClassifier(n_estimators=150,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
print(importances)
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

labels = []
for f in range(X.shape[1]):
    labels.append(feature_cols[f])      
    
# Plot the feature importances of the forest
plt.figure(figsize=(10,6))
plt.title("Feature importances by ExtraTrees Classifiers")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), labels, rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.show()



# In[23]:


pip install aif360


# In[24]:


import sklearn
from aif360.sklearn import metrics
model = model1
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("======================Score===================")
#print(classification_report(y_test,prediction))
print('Accuracy: %.4f' % accuracy_score(y_test, prediction))
print('Precision: %.4f' % precision_score(y_test, prediction, average='macro'))
print('Recall: %.4f' % recall_score(y_test, prediction, average='macro'))
print('F1 Score: %.4f' % f1_score(y_test, prediction, average='macro'))
print('Specificity: %.4f' % metrics.specificity_score(y_test, prediction, pos_label=1, sample_weight=None, zero_division='warn'))
print('Kappa: %.4f' % sklearn.metrics.cohen_kappa_score(y_test, prediction, labels=None, weights=None, sample_weight=None))

#-----------------------K-fold validation---------------------------------------
for i in range(5):
 kf=KFold(n_splits=2+i)
 score=cross_val_score(model, X_test, y_test,cv=kf)
 print("\n--------------------------------------")
 print("Cross Validation Scores are {}".format(score))
 print("Average Cross Validation score :{}".format(score.mean()))
 print("\n--------------------------------------")
    


# In[52]:


#---------Random Forest
forest = RandomForestClassifier(n_estimators=150,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
print(importances)
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

labels = []
for f in range(X.shape[1]):
    labels.append(feature_cols[f])      
    
# Plot the feature importances of the forest
plt.figure(figsize=(10,6))
plt.title("Feature importances by Random Forest Classifiers")
plt.bar(range(X.shape[1]), importances[indices],
       color="b", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), labels, rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.show()


# In[53]:


import sklearn
from aif360.sklearn import metrics
model = model5
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("======================Score===================")
#print(classification_report(y_test,prediction))
print('Accuracy: %.4f' % accuracy_score(y_test, prediction))
print('Precision: %.4f' % precision_score(y_test, prediction, average='macro'))
print('Recall: %.4f' % recall_score(y_test, prediction, average='macro'))
print('F1 Score: %.4f' % f1_score(y_test, prediction, average='macro'))
print('Specificity: %.4f' % metrics.specificity_score(y_test, prediction, pos_label=1, sample_weight=None, zero_division='warn'))
print('Kappa: %.4f' % sklearn.metrics.cohen_kappa_score(y_test, prediction, labels=None, weights=None, sample_weight=None))

#-----------------------K-fold validation---------------------------------------
for i in range(5):
 kf=KFold(n_splits=2+i)
 score=cross_val_score(model, X_test, y_test,cv=kf)
 print("\n--------------------------------------")
 print("Cross Validation Scores are {}".format(score))
 print("Average Cross Validation score :{}".format(score.mean()))
 print("\n--------------------------------------")
    
#--------------------Learning Curve---------------------------------------------
train_sizes, train_scores, test_scores = learning_curve(model, X_test, y_test, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.subplots(1, figsize=(7,5))
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="red", label="Cross-validation score")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()

#----------------------Confusion Matrix--------------------------------
def plot_confusion_matrix(normalize):
  plt.figure(figsize=(10,6))
  classes = ['No_Addiction','Mild', 'Moderate', 'Severe']
  tick_marks = [0.5,1.5, 2.5, 3.5]
  cn = confusion_matrix(y_test, prediction,normalize=normalize)
  sns.heatmap(cn,cmap='BuPu',annot=True)
  plt.xticks(tick_marks, classes)
  plt.yticks(tick_marks, classes)
  plt.title('Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
print('Confusion Matrix for LR with Normalized Values')
plot_confusion_matrix(normalize='true')

#----------------------Roc curves-----------------------
def plot_roc_curve(y_test, y_pred):

  n_classes = len(np.unique(y_test))
  y_test = label_binarize(y_test, classes=np.arange(n_classes))
  y_pred = label_binarize(y_pred, classes=np.arange(n_classes))

  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  fpr = dict()
  lw=2
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
  colors = cycle(['blue', 'red', 'green', 'brown'])
  for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([-0.05, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve for multi-class Addiction Classification')
  plt.legend(loc="lower right")
  
  plt.show()
plot_roc_curve(y_test, prediction)


# In[33]:


get_ipython().system('pip install mrmr_selection')


# In[49]:


#---------MRMR
import mrmr
from mrmr import mrmr_classif

selected_features = mrmr_classif(X=X, y=y, K=5)
print(selected_features)     
# Plot the feature importances of the fores
plt.show()
New_df = df[selected_features]
X_train, X_test, y_train, y_test = train_test_split(New_df, y, test_size=0.20, random_state=0)


# In[50]:


model = model3
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("======================Score===================")
#print(classification_report(y_test,prediction))
print('Accuracy: %.4f' % accuracy_score(y_test, prediction))
print('Precision: %.4f' % precision_score(y_test, prediction, average='macro'))
print('Recall: %.4f' % recall_score(y_test, prediction, average='macro'))
print('F1 Score: %.4f' % f1_score(y_test, prediction, average='macro'))
print('Specificity: %.4f' % metrics.specificity_score(y_test, prediction, pos_label=1, sample_weight=None, zero_division='warn'))
print('Kappa: %.4f' % sklearn.metrics.cohen_kappa_score(y_test, prediction, labels=None, weights=None, sample_weight=None))


# In[33]:


get_ipython().system('pip install mifs')


# In[ ]:




