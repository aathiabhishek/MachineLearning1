#!/usr/bin/env python
# coding: utf-8

# # Task 7: AutoFeatureSelector Tool
# ## This task is to test your understanding of various Feature Selection methods outlined in the lecture and the ability to apply this knowledge in a real-world dataset to select best features and also to build an automated feature selection tool as your toolkit
# 
# ### Use your knowledge of different feature selector methods to build an Automatic Feature Selection tool
# - Pearson Correlation
# - Chi-Square
# - RFE
# - Embedded
# - Tree (Random Forest)
# - Tree (Light GBM)

# ### Dataset: FIFA 19 Player Skills
# #### Attributes: FIFA 2019 players attributes like Age, Nationality, Overall, Potential, Club, Value, Wage, Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight, LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB, Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, and Release Clause.

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats


# In[6]:


player_df = pd.read_csv("fifa19.csv")


# In[9]:


numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']


# In[11]:


numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']
player_df = player_df[numcols+catcols]
traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
features = traindf.columns

traindf = traindf.dropna()
traindf = pd.DataFrame(traindf,columns=features)
y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']


# In[13]:


player_df = player_df[numcols+catcols]


# In[15]:


traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
features = traindf.columns

traindf = traindf.dropna()


# In[17]:


traindf = pd.DataFrame(traindf,columns=features)


# In[19]:


y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']


# In[21]:


X.head()


# In[23]:


len(X.columns)


# ### Set some fixed set of features

# In[26]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
feature_name = list(X.columns)
# no of maximum features we need to select
num_feats=30


# ## Filter Feature Selection - Pearson Correlation

# ### Pearson Correlation function

# In[30]:


def cor_selector(X, y,num_feats):
    # Your code goes here (Multiple lines)
    cor_list=[]
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)

    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    
    cor_support = [True if i in cor_feature else False for i in feature_name]
    # Your code ends here
    return cor_support, cor_feature


# In[32]:


cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')


# ### List the selected features from Pearson Correlation

# In[35]:


cor_feature


# ## Filter Feature Selection - Chi-Sqaure

# In[38]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler


# ### Chi-Squared Selector function

# In[41]:


def chi_squared_selector(X, y, num_feats):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    chi2_selector = SelectKBest(chi2, k=num_feats)
    X_new = chi2_selector.fit_transform(X_scaled, y)
    
    chi_support = chi2_selector.get_support()
    chi_feature = X.columns[chi_support].tolist()
    
    return chi_support, chi_feature


# In[43]:


chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
print(str(len(chi_feature)), 'selected features')


# ### List the selected features from Chi-Square 

# In[46]:


chi_feature


# ## Wrapper Feature Selection - Recursive Feature Elimination

# In[49]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
# no of maximum features we need to select
num_feats=30


# ### RFE Selector function

# In[52]:


from sklearn.preprocessing import StandardScaler
def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    
    rfe = RFE(estimator=model, n_features_to_select=num_feats)
    rfe.fit(X_scaled, y)
    rfe_support = rfe.support_  
    rfe_feature = X.columns[rfe_support]

    # Your code ends here
    return rfe_support, rfe_feature


# In[54]:


rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
print(str(len(rfe_feature)), 'selected features')


# ### List the selected features from RFE

# In[57]:


rfe_feature


# ## Embedded Selection - Lasso: SelectFromModel

# In[65]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# In[67]:


def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    logreg = LogisticRegression()
    embedded_lr_selector = SelectFromModel((logreg), max_features=num_feats)
    embedded_lr_selector = embedded_lr_selector.fit(X, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()
    # Your code ends here
    return embedded_lr_support, embedded_lr_feature


# In[69]:


embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
print(str(len(embedded_lr_feature)), 'selected features')


# In[71]:


embedded_lr_feature


# ## Tree based(Random Forest): SelectFromModel

# In[74]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[76]:


def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    rf = RandomForestClassifier(n_estimators=100)
    embedded_rf_selector = SelectFromModel(rf,max_features=num_feats)
    embedded_rf_selector = embedded_rf_selector.fit(X,y)
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature



# In[78]:


embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
print(str(len(embedded_rf_feature)), 'selected features')


# In[80]:


embedded_rf_feature


# ## Tree based(Light GBM): SelectFromModel

# In[83]:


from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier


# In[85]:


def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    lgbmc = LGBMClassifier(n_estimators=500,
                      learning_rate=0.05,
                      num_leaves=32,
                      colsample_bytree=0.2,
                      reg_alpha=3,
                      reg_lambda=1,
                      min_split_gain=0.01,
                      min_child_weight=40
     )
    embedded_lgbm_selector = SelectFromModel(estimator=lgbmc, threshold='median', max_features=num_feats)
    embedded_lgbm_selector = embedded_lgbm_selector.fit(X,y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
    # Your code ends here
    return embedded_lgbm_support, embedded_lgbm_feature


# In[87]:


embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
print(str(len(embedded_lgbm_feature)), 'selected features')


# In[664]:


embedded_lgbm_feature


# ## Putting all of it together: AutoFeatureSelector Tool

# In[89]:


pd.set_option('display.max_rows', None)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embedded_lr_support,
                                    'Random Forest':embedded_rf_support, 'LightGBM':embedded_lgbm_support})
# count the selected times for each feature
#feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
feature_selection_df['Total'] = feature_selection_df.iloc[:, 1:].sum(axis=1)  

# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)


# ## Can you build a Python script that takes dataset and a list of different feature selection methods that you want to try and output the best (maximum votes) features from all methods?

# In[92]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=200)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression()
model.fit(X_scaled, y)
model = LogisticRegression(solver='liblinear', max_iter=200)

def preprocess_dataset(dataset_path):
    # Load the dataset
    player_df = pd.read_csv(dataset_path)

    # Specify numerical and categorical columns
    numcols = ['Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'LongPassing', 
               'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Stamina', 'Volleys', 
               'FKAccuracy', 'Reactions', 'Balance', 'ShotPower', 'Strength', 'LongShots', 
               'Aggression', 'Interceptions']
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality', 'Weak Foot']
    
    # Filter relevant columns
    player_df = player_df[numcols + catcols]
    
    # One-hot encode categorical columns
    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols], drop_first=True)], axis=1)
    features = traindf.columns

    # Drop rows with missing values
    traindf = traindf.dropna()

    # Ensure traindf contains the specified columns
    traindf = pd.DataFrame(traindf, columns=features)

    # Define target variable 'y' and feature matrix 'X'
    y = traindf['Overall']
    X = traindf.copy()
    del X['Overall']

    # Define feature names and maximum number of features to select
    feature_name = list(X.columns)
    num_feats = 30

    return X, y, num_feats


# In[94]:


def autoFeatureSelector(dataset_path, methods=[]):
    X, y, num_feats = preprocess_dataset(dataset_path)
    feature_lists = []
    
    # Run each method and collect the best features
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y, num_feats)
        feature_lists.append(cor_feature)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y, num_feats)
        feature_lists.append(chi_feature)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y, num_feats)
        feature_lists.append(rfe_feature)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
        feature_lists.append(embedded_lr_feature)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
        feature_lists.append(embedded_rf_feature)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
        feature_lists.append(embedded_lgbm_feature)
    
    # Flatten the list of feature lists
    all_features = [feature for sublist in feature_lists for feature in sublist]
    
    # Count occurrences of each feature
    feature_count = pd.Series(all_features).value_counts()
    
    # Select the most frequently occurring features
    best_features = feature_count[feature_count == feature_count.max()].index.tolist()
    
    return best_features


# Pearson correlation selector
def cor_selector(X, y, num_feats):
    cor_list = []
    for col in X.columns:
        cor = np.corrcoef(X[col], y)[0, 1]  # Pearson correlation
        cor_list.append(cor if not np.isnan(cor) else 0)  # Handle NaN values

    # Select top features based on absolute correlation
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    cor_support = [col in cor_feature for col in X.columns]
    
    return cor_support, cor_feature


# Chi-square selector
def chi_squared_selector(X, y, num_feats):
    bestfeatures = SelectKBest(score_func=chi2, k=num_feats)
    bestfeatures.fit(X, y)
    
    chi_support = bestfeatures.get_support()
    chi_feature = X.columns[chi_support].tolist()
    
    return chi_support, chi_feature


# Recursive feature elimination (RFE)
def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    
    rfe = RFE(estimator=model, n_features_to_select=num_feats)
    rfe.fit(X_scaled, y)
    rfe_support = rfe.support_  
    rfe_feature = X.columns[rfe_support]

    # Your code ends here
    return rfe_support, rfe_feature


def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    logreg = LogisticRegression()
    embedded_lr_selector = SelectFromModel((logreg), max_features=num_feats)
    embedded_lr_selector = embedded_lr_selector.fit(X, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()
    # Your code ends here
    return embedded_lr_support, embedded_lr_feature



# Random forest embedded selector
def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    rf = RandomForestClassifier(n_estimators=100)
    embedded_rf_selector = SelectFromModel(rf,max_features=num_feats)
    embedded_rf_selector = embedded_rf_selector.fit(X,y)
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature


# LightGBM embedded selector
def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    lgbmc = LGBMClassifier(n_estimators=500,
                      learning_rate=0.05,
                      num_leaves=32,
                      colsample_bytree=0.2,
                      reg_alpha=3,
                      reg_lambda=1,
                      min_split_gain=0.01,
                      min_child_weight=40
     )
    embedded_lgbm_selector = SelectFromModel(estimator=lgbmc, threshold='median', max_features=num_feats)
    embedded_lgbm_selector = embedded_lgbm_selector.fit(X,y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
    # Your code ends here
    return embedded_lgbm_support, embedded_lgbm_feature
    
    best_features = feature_count[feature_count == feature_count.max()].index.tolist()
    return best_features


# In[96]:


best_features = autoFeatureSelector(dataset_path="fifa19.csv", methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
best_features


# ### Last, Can you turn this notebook into a python script, run it and submit the python (.py) file that takes dataset and list of methods as inputs and outputs the best features

# In[ ]:




