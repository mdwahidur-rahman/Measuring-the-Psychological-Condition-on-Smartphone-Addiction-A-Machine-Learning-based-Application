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


# In[71]:


df = pd.read_csv("C:/Users/Tuhin/Desktop/Nomophobia/Nomophobia_Initial1.CSV")


# In[72]:


dataset = df


# In[73]:


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


# In[74]:


#correlation matrix
corrmat = dataset.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()

#treatment correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.index
f, ax = plt.subplots(figsize=(14, 9))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.32)
hm = sns.heatmap(cm, cmap="YlGnBu", cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 14}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[75]:


dataset


# # Data Load for ML

# In[76]:


# Scaling Age
scaler = MinMaxScaler()
dataset['Age'] = scaler.fit_transform(dataset[['Age']])
dataset.head()


# In[103]:


# define X and y
feature_cols = ['Age', 'Gender', 'Occupation', 'Time_spent_on_Mobile_per_day', 'Addiction_Level', 'Social_Media_User', 'Gaming_User', 'Academician']
X = dataset[feature_cols]
y = dataset.Addiction_Level


# In[104]:


model1 = SVC()
model2 = RandomForestClassifier()
model3 = DecisionTreeClassifier()
model4 = GaussianNB()
model5 = XGBClassifier()
model6 = KNeighborsClassifier()
model7 = LogisticRegression()
Final_Model = VotingClassifier(estimators=[('svc', model1), ('rf', model2), ('dt', model3), ('bayes', model7), ('xgb', model5), 
                                             ('knn', model6), ('lr', model7), ], voting='hard')


# # Original and Ensemble

# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[26]:


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


# In[29]:


#----------------------Confusion Matrix--------------------------------
def plot_confusion_matrix(normalize):
  plt.figure(figsize=(10,6))
  classes = ['No_Addiction','Mild', 'Moderate', 'Severe']
  tick_marks = [0.5,1.5, 2.5, 3.5]
  cn = confusion_matrix(y_test, prediction,normalize=normalize)
  sns.heatmap(cn,cmap='Blues',annot=True)
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
  plt.title('ROC Curve for multi-class Addiction Classification with Original')
  plt.legend(loc="lower right")
  
  plt.show()
plot_roc_curve(y_test, prediction)


# # Feature Importance Part

# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[44]:


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    #Build a forest and compute the feature importances
    
    
#---------Extra Tree
forest = RandomForestClassifier(n_estimators=5,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_

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


# In[45]:


import sklearn
from aif360.sklearn import metrics
model = forest
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


# In[48]:


#----------------------Confusion Matrix--------------------------------
def plot_confusion_matrix(normalize):
  plt.figure(figsize=(10,6))
  classes = ['No_Addiction','Mild', 'Moderate', 'Severe']
  tick_marks = [0.5,1.5, 2.5, 3.5]
  cn = confusion_matrix(y_test, prediction,normalize=normalize)
  sns.heatmap(cn,cmap='crest',annot=True)
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
  plt.title('ROC Curve for multi-class Addiction Classification with RFC')
  plt.legend(loc="lower right")
  
  plt.show()
plot_roc_curve(y_test, prediction)


# # Feature Selection

# In[105]:


#---------MRMR
import mrmr
from mrmr import mrmr_classif

selected_features = mrmr_classif(X=X, y=y, K=5)
print(selected_features)     
# Plot the feature importances of the fores
plt.show()
New_df = df[selected_features]
X_train, X_test, y_train, y_test = train_test_split(New_df, y, test_size=0.20, random_state=0)


# In[106]:


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

#----------------------Confusion Matrix--------------------------------
def plot_confusion_matrix(normalize):
  plt.figure(figsize=(10,6))
  classes = ['No_Addiction','Mild', 'Moderate', 'Severe']
  tick_marks = [0.5,1.5, 2.5, 3.5]
  cn = confusion_matrix(y_test, prediction,normalize=normalize)
  sns.heatmap(cn,cmap='Accent',annot=True)
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
  plt.title('ROC Curve for multi-class Addiction Classification with mRMR')
  plt.legend(loc="lower right")
  
  plt.show()
plot_roc_curve(y_test, prediction)


# In[107]:


# define X and y
feature_cols = ['Age', 'Gender', 'Occupation', 'Time_spent_on_Mobile_per_day', 'Addiction_Level', 'Social_Media_User', 'Gaming_User', 'Academician']
X = dataset[feature_cols]
y = dataset.Addiction_Level


# In[108]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[109]:


ss=StandardScaler()
X_trains=ss.fit_transform(X_train)
X_tests=ss.fit_transform(X_test)


# In[110]:


from sklearn.svm import SVC
svc=SVC()
svc.fit(X_trains,y_train)


# In[111]:


import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

forward_fs_best=sfs(estimator = svc, k_features = 'best', forward = True,verbose = 1, scoring = 'r2')
sfs_forward_best=forward_fs_best.fit(X_trains,y_train)

X_trains_df=pd.DataFrame(X_trains,columns=X_train.columns)
from sklearn.feature_selection import RFE
svc_lin=SVC(kernel='linear')
svm_rfe_model=RFE(estimator=svc_lin)
svm_rfe_model_fit=svm_rfe_model.fit(X_trains_df,y_train)
feat_index = pd.Series(data = svm_rfe_model_fit.ranking_, index = X_train.columns)
signi_feat_rfe = feat_index[feat_index==1].index
print('Significant features from RFE',signi_feat_rfe)


#---New Load
New_df = df[signi_feat_rfe]
X_train, X_test, y_train, y_test = train_test_split(New_df, y, test_size=0.20, random_state=0)


# In[112]:


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

#----------------------Confusion Matrix--------------------------------
def plot_confusion_matrix(normalize):
  plt.figure(figsize=(10,6))
  classes = ['No_Addiction','Mild', 'Moderate', 'Severe']
  tick_marks = [0.5,1.5, 2.5, 3.5]
  cn = confusion_matrix(y_test, prediction,normalize=normalize)
  sns.heatmap(cn,cmap='BuPu_r',annot=True)
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
  plt.title('ROC Curve for multi-class Addiction Classification with SVC')
  plt.legend(loc="lower right")
  
  plt.show()
plot_roc_curve(y_test, prediction)


# # Feature Reduction

# In[155]:


# define X and y
feature_cols = ['Age', 'Gender', 'Occupation', 'Time_spent_on_Mobile_per_day', 'Addiction_Level', 'Social_Media_User', 'Gaming_User', 'Academician']
X = dataset[feature_cols]
y = dataset.Addiction_Level
Y=y
columns = dataset.columns


# In[156]:


train_x, test_x, y_train, test_y = train_test_split(X, Y, test_size=0.2)


# In[143]:


#pca analysis
sc = StandardScaler()
x_train = sc.fit_transform(train_x)
x_test = sc.transform(test_x)

pca = PCA(n_components= 5)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
#fit = pca.fit(x_train)


# In[153]:


model = model1
model.fit(x_train,y_train)
prediction = model.predict(x_test)
print("======================Score===================")
#print(classification_report(y_test,prediction))
print('Accuracy: %.4f' % accuracy_score(test_y, prediction))
print('Precision: %.4f' % precision_score(test_y, prediction, average='macro'))
print('Recall: %.4f' % recall_score(test_y, prediction, average='macro'))
print('F1 Score: %.4f' % f1_score(test_y, prediction, average='macro'))
print('Kappa: %.4f' % sklearn.metrics.cohen_kappa_score(test_y, prediction, labels=None, weights=None, sample_weight=None))

#----------------------Confusion Matrix--------------------------------
def plot_confusion_matrix(normalize):
  plt.figure(figsize=(10,6))
  classes = ['No_Addiction','Mild', 'Moderate', 'Severe']
  tick_marks = [0.5,1.5, 2.5, 3.5]
  cn = confusion_matrix(test_y, prediction,normalize=normalize)
  sns.heatmap(cn,cmap='BuGn',annot=True)
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
  plt.title('ROC Curve for multi-class Addiction Classification with PCA')
  plt.legend(loc="lower right")
  
  plt.show()
plot_roc_curve(test_y, prediction)


# In[169]:


#lda Analysis
lda = LinearDiscriminantAnalysis(n_components=3)
x_train = lda.fit_transform(train_x, y_train)
x_test = lda.transform(test_x)
sc = StandardScaler()
x_train = sc.fit_transform(train_x)
x_test = sc.transform(test_x)


# In[170]:


model = model4
model.fit(x_train,y_train)
prediction = model.predict(x_test)
print("======================Score===================")
#print(classification_report(y_test,prediction))
print('Accuracy: %.4f' % accuracy_score(test_y, prediction))
print('Precision: %.4f' % precision_score(test_y, prediction, average='macro'))
print('Recall: %.4f' % recall_score(test_y, prediction, average='macro'))
print('F1 Score: %.4f' % f1_score(test_y, prediction, average='macro'))
print('Kappa: %.4f' % sklearn.metrics.cohen_kappa_score(test_y, prediction, labels=None, weights=None, sample_weight=None))

#----------------------Confusion Matrix--------------------------------
def plot_confusion_matrix(normalize):
  plt.figure(figsize=(10,6))
  classes = ['No_Addiction','Mild', 'Moderate', 'Severe']
  tick_marks = [0.5,1.5, 2.5, 3.5]
  cn = confusion_matrix(test_y, prediction,normalize=normalize)
  sns.heatmap(cn,cmap='OrRd',annot=True)
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
  plt.title('ROC Curve for multi-class Addiction Classification with LDA')
  plt.legend(loc="lower right")
  
  plt.show()
plot_roc_curve(test_y, prediction)


# In[ ]:




