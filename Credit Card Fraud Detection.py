#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression


# In[3]:


df = pd.read_csv(r"C:\Users\user\Downloads\creditcard.csv")


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.isna().sum()


# In[10]:


for i in df.columns:
    plt.boxplot(df[i])
    plt.title('box plot for ' +str(i))
    plt.show()
    


# In[11]:


X = df.drop(['Class'], axis = 1)
y = df.Class


# In[12]:


for i in X.columns:
    winsor = Winsorizer(capping_method='iqr', fold = 1.5, tail = 'both', variables = [i])
    X[i] = pd.DataFrame(winsor.fit_transform(X[[i]]))


# In[13]:


for i in df.columns:
    plt.boxplot(df[i])
    plt.title('box plot for ' +str(i))
    plt.show()


# In[42]:


stdscale = StandardScaler()
X = pd.DataFrame(stdscale.fit_transform(X), columns = X.columns)


# In[43]:


X


# In[14]:


y.value_counts()


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=None)


# In[45]:


len(X_train), len(X_test), len(y_train), len(y_test), y_train.value_counts()


# In[46]:


kfcv = KFold(n_splits = 5, random_state=None, shuffle = False)


# In[47]:


log_class = LogisticRegression()


# In[48]:


from imblearn.over_sampling import SMOTE


# In[49]:


smote = SMOTE(random_state=20)


# In[50]:


X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
len(X_train_sm), len(y_train_sm), y_train_sm.value_counts()


# In[51]:


grid = {'C' : 10.0**np.arange(-2, 3), 'penalty' : ['l1', 'l2']}


# In[52]:


gridsrch = GridSearchCV(log_class, param_grid=grid, cv=kfcv)


# In[53]:


gridsrch.fit(X_train_sm, y_train_sm)


# In[54]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, roc_auc_score, recall_score, precision_score, f1_score


# In[55]:


test_pred = gridsrch.predict(X_test)


# In[69]:


print(confusion_matrix(y_test, test_pred))
print(recall_score(y_test, test_pred))
print(f1_score(y_test, test_pred))
print(roc_auc_score(y_test, test_pred))
#print(classification_report(y_test, test_pred))


# In[57]:


train_pred = gridsrch.predict(X_train)


# In[70]:


print(confusion_matrix(y_train, train_pred))
print(recall_score(y_train, train_pred))
print(f1_score(y_train, train_pred))
print(roc_auc_score(y_train, train_pred))
#print(classification_report(y_train, train_pred))


# In[ ]:




