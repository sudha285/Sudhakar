

# import libraries 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm 
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# Reading Train data


df_train=pd.read_csv('C:/Users/sudhakaran.srinivasa/Desktop/Personal/house-prices-advanced-regression-techniques/train.csv')


# In[375]:


df_train.columns


# In[376]:


df_train['SalePrice'].describe()


# In[377]:


sns.distplot(df_train['SalePrice'])


# In[378]:


fig, ax = plt.subplots()
ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[379]:


fig, ax = plt.subplots()
ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[380]:


fig, ax = plt.subplots()
ax.scatter(x = df_train['TotalBsmtSF'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
plt.show()


# Log transformation of train


df_train['SalePrice']=np.log1p(df_train['SalePrice'])


# In[382]:


sns.distplot(df_train['SalePrice'], fit=norm)


# In[383]:


ntrain = df_train.shape[0]


# In[384]:


ntrain


# reading Test data


df_test=pd.read_csv('C:/Users/sudhakaran.srinivasa/Desktop/Personal/house-prices-advanced-regression-techniques/test.csv')


# In[386]:


ntest=df_test.shape[0]


# In[387]:


ntest


# In[388]:


train_ID = df_train['Id']
test_ID = df_test['Id']


# In[389]:


df_train.columns


# In[390]:


ntrain = df_train.shape[0]
ntest = df_test.shape[0]


# In[391]:


y_train = df_train.SalePrice.values


# combining test and train data


all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)


# In[393]:


all_data.shape


# % missing value


ss=all_data.isnull().sum()
ss
percent=(ss/2907)*100
percent
percent_final=percent.sort_values(ascending=False)
percent_final
percent_final.head(34)

quantitative = [f for f in all_data.columns if all_data.dtypes[f] != 'object']


# In[402]:


all_data


# Outlier treatment
# filling NA into None

for col in ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
           "GarageType", "GarageFinish", "GarageQual", "GarageCond",
           "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
            "BsmtFinType2", "MSSubClass", "MasVnrType"):
    all_data[col] = all_data[col].fillna("None")

# The area of the lot out front is likely to be similar to the houses in the local neighbourhood
#use the median value of the houses in the neighbourhood to fill
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


#missing values with 0 
for col in ("GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", 
           "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea",
           "BsmtFullBath", "BsmtHalfBath"):
    all_data[col] = all_data[col].fillna(0)



#Mode
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data["Functional"] = all_data["Functional"].fillna(all_data['Functional'].mode()[0])

# dropping utilities
all_data = all_data.drop(['Utilities'], axis=1)


# Feature Engineeering 

# Age of building
all_data['AgeWhenSold']  = all_data['YrSold'] - all_data['YearBuilt']
all_data['YearsSinceRemod']  = all_data['YrSold'] - all_data['YearRemodAdd']


# In[307]:


all_data.to_csv('all_datacheck.csv',index=False)


# In[405]:


all_data = all_data.drop(["YearBuilt"],axis=1)
all_data = all_data.drop(["YearRemodAdd"],axis=1)
all_data = all_data.drop(["YrSold"],axis=1)
all_data = all_data.drop(["GarageYrBlt"],axis=1)


# In[406]:


all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

# Overall condition
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)


# label encoder 


from sklearn.preprocessing import LabelEncoder
columns = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in columns:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))
        
print('Shape all_data: ', all_data.shape)



# Overall quality of the house
all_data["OverallGrade"] = all_data["OverallQual"] * all_data["OverallCond"]
# Overall quality of the exterior
all_data["ExterGrade"] = all_data["ExterQual"] * all_data["ExterCond"]
# Overall kitchen score
all_data["KitchenScore"] = all_data["KitchenAbvGr"] * all_data["KitchenQual"]
# Overall fireplace score
all_data["FireplaceScore"] = all_data["Fireplaces"] * all_data["FireplaceQu"]


# Total number of bathrooms
all_data["TotalBath"] = all_data["BsmtFullBath"] + (0.5 * all_data["BsmtHalfBath"]) + all_data["FullBath"] + (0.5 * all_data["HalfBath"])
# Total SF for house
all_data["TotalSF"] = all_data["GrLivArea"] + all_data["TotalBsmtSF"]
# Total SF for 1st + 2nd floors
all_data["FloorsSF"] = all_data["1stFlrSF"] + all_data["2ndFlrSF"]
# Total SF for porch
all_data["PorchSF"] = all_data["OpenPorchSF"] + all_data["EnclosedPorch"] + all_data["3SsnPorch"] + all_data["ScreenPorch"]
# Has masonry veneer or not
all_data["HasMasVnr"] = all_data.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, "Stone" : 1, "None" : 0})
# House completed before sale or not
all_data["CompletedBFSale"] = all_data.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})


# Month Solda


all_data['mnth_sin'] = np.sin((all_data.MoSold-1)*(2.*np.pi/12))
all_data['mnth_cos'] = np.cos((all_data.MoSold-1)*(2.*np.pi/12))


# In[410]:


all_data["TotalBuiltSF"] = all_data["GrLivArea"] + all_data["GarageArea"]+ all_data["OpenPorchSF"]+ all_data["PoolArea"]
all_data["BuiltInRatio"] = all_data["TotalBuiltSF"] /all_data["LotArea"]



# In[417]:


all_data["GarageGrade"] = all_data["GarageQual"] * all_data["GarageCond"]


# In[418]:



all_data = all_data.drop(["Exterior2nd"   ],axis=1)
all_data = all_data.drop(["Fence"   ],axis=1)
all_data = all_data.drop(["Functional"       ],axis=1)
all_data = all_data.drop(["GarageCond"       ],axis=1)
all_data = all_data.drop(["GarageFinish"      ],axis=1)
all_data = all_data.drop(["GarageQual"    ],axis=1)
all_data = all_data.drop(["KitchenAbvGr"       ],axis=1)
all_data = all_data.drop(["KitchenQual"       ],axis=1)
all_data = all_data.drop(["MiscFeature"       ],axis=1)

all_data = all_data.drop(["MiscVal"       ],axis=1)
all_data = all_data.drop(["OverallCond"       ],axis=1)
all_data = all_data.drop(["OverallQual"       ],axis=1)
all_data = all_data.drop(["Foundation"   ],axis=1)



# Top  features polynomials

# Quadratic

all_data["GarageCars-2"] = all_data["GarageCars"] ** 2
all_data["TotalBath-2"] = all_data["TotalBath"] ** 2


# Cubic
all_data["GarageCars-3"] = all_data["GarageCars"] ** 3
all_data["TotalBath-3"] = all_data["TotalBath"] ** 3


# Square Root
all_data["GarageCars-Sq"] = np.sqrt(all_data["GarageCars"])
all_data["TotalBath-Sq"] = np.sqrt(all_data["TotalBath"])


# In[420]:


corr = df_train.corr()
corr.sort_values(["SalePrice"], ascending = True, inplace = True)
corr = corr.SalePrice


# In[421]:


plt.subplots(figsize =(15, 10))
corr.plot(kind='barh');


# In[422]:


from scipy import stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check how skewed they are
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

plt.subplots(figsize =(15, 25))
skewed_feats.plot(kind='barh');


# In[423]:



skewness = skewed_feats[abs(skewed_feats) > 0.5]
print("There are ", skewness.shape[0],  "skewed numerical features to Box-Cox transform")

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
all_data = pd.get_dummies(all_data)
print(all_data.shape)


# In[424]:


all_data = all_data.fillna(all_data.mean())


# In[425]:


train = all_data[:ntrain]
test = all_data[ntrain:]

print(train.shape)
print(test.shape)


# In[426]:


y_train



# Train model 


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from lightgbm import LGBMRegressor


# cross validation


n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[431]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[432]:


score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# In[433]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


# In[434]:


score = rmsle_cv(lasso)
print("lasso score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# In[435]:


import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# In[436]:


score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[437]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


# In[438]:


score = rmsle_cv(KRR)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[439]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# In[440]:


score = rmsle_cv(GBoost)
print("\nGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[441]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# In[442]:


score = rmsle_cv(ENet)
print("\nENet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[450]:


from sklearn.svm import SVR
svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003, ))


# In[451]:


score = rmsle_cv(svr)
print("\nSVR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# Averaged model


averaged_models = AveragingModels(models = (ENet, lasso,model_xgb,GBoost))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# stacked model


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
 def __init__(self, base_models, meta_model, n_folds=5):
     self.base_models = base_models
     self.meta_model = meta_model
     self.n_folds = n_folds

 # We again fit the data on clones of the original models
 def fit(self, X, y):
     self.base_models_ = [list() for x in self.base_models]
     self.meta_model_ = clone(self.meta_model)
     kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
     
     # Train cloned base models then create out-of-fold predictions
     # that are needed to train the cloned meta-model
     out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
     for i, model in enumerate(self.base_models):
         for train_index, holdout_index in kfold.split(X, y):
             instance = clone(model)
             self.base_models_[i].append(instance)
             instance.fit(X[train_index], y[train_index])
             y_pred = instance.predict(X[holdout_index])
             out_of_fold_predictions[holdout_index, i] = y_pred
             
     # Now train the cloned  meta-model using the out-of-fold predictions as new feature
     self.meta_model_.fit(out_of_fold_predictions, y)
     return self

 #Do the predictions of all base models on the test data and use the averaged predictions as 
 #meta-features for the final prediction which is done by the meta-model
 def predict(self, X):
     meta_features = np.column_stack([
         np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
         for base_models in self.base_models_ ])
     return self.meta_model_.predict(meta_features)


# In[461]:


stacked_averaged_models = StackingAveragedModels(base_models = (ENet,GBoost,model_xgb,model_lgb),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# In[463]:


lgb_model_full_data = model_lgb.fit(train,y_train)


# In[464]:



lasso_model_full_data = lasso.fit(train,y_train)


# In[465]:



xgb_model_full_data = model_xgb.fit(train,y_train)


# In[466]:



gboost_model_full_data = GBoost.fit(train,y_train)


# In[467]:



svr_model_full_data = svr.fit(train,y_train)
enet_model_full_data = ENet.fit(train,y_train)


# Final submission -stacked model 


submission=pd.read_csv('C:/Users/sudhakaran.srinivasa/Desktop/Personal/house-prices-advanced-regression-techniques/sample_submission.csv')
submission
stacked_output=averaged_models.fit(train,y_train)
ped_averaged_models=stacked_output.predict(test)
submission=submission.drop('SalePrice',1)
pred_avg_mdl=np.expm1(ped_averaged_models)
submission['SalePrice']=pred_avg_mdl
submission.to_csv('final_jan15.csv',index=False)






