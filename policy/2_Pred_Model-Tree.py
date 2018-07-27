
# coding: utf-8

# In[1]:
def main():

	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt

	# https://datahack.analyticsvidhya.com/contest/mckinsey-analytics-online-hackathon-4/#

	## TODO: implement bootstrapping for getting coefficient error bars --> random forest is already doing that!
	## TODO: implement proper data transformation pipelining  --> Done!
	## TODO: normalize/scale inputs to allow for use of regularization  --> can be skipped in case of xgboost random forest
	## TODO: implement gridsearch to get best hyperparameters



	file = "train_ZoGVYWq.csv"
	data = pd.read_csv(file)


	# ### Pre-prep data (type and name adjustments only, data independent)

	# In[2]:


	def pre_prep(df):
		data = df.copy()
		data.set_index('id', inplace=True)
		data['sourcing_channel'] = data['sourcing_channel'].astype('category') 
		data['residence_area_type'] = data['residence_area_type'].astype('category')
		new_names = {'perc_premium_paid_by_cash_credit': 'cash_credit',
					'Count_3-6_months_late': 'late3',
					'Count_6-12_months_late': 'late6',
					'Count_more_than_12_months_late': 'late12',
					'application_underwriting_score': 'score',
					'residence_area_type': 'residence',
					'no_of_premiums_paid': 'prems_paid',
					'sourcing_channel': 'channel'}
		data.rename(index=str, columns=new_names, inplace=True)
		return data

	data['renewal'] = data['renewal'].astype('bool')
	data_pre_prepped = pre_prep(data)


	# ### Numerical/Visual inspection
	# Full visual inspection: see other notebook for plots, etc

	# In[3]:


	data_pre_prepped.T


	# In[4]:


	data_pre_prepped.info()


	# In[5]:


	data_pre_prepped.describe()


	# # Split sets

	# In[6]:


	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(data_pre_prepped.drop('renewal', axis=1), 
														data_pre_prepped['renewal'], 
														test_size=0.20, 
														random_state=42)

	# train_test_split(data_pre_prepped, data_pre_prepped['renewal'], test_size=0.20, random_state=42)
	len(X_train)


	# In[7]:


	X_train.info()


	#  # Creating the transformation pipeline 

	# In[8]:


	# from sklearn.base import BaseEstimator, TransformerMixin
	# from sklearn.feature_extraction import DictVectorizer
	# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	from sklearn.pipeline import Pipeline, FeatureUnion
	from dftrans import DFFeatureUnion, DFImputer, ColumnExtractor, Log1pTransformer, DummyTransformer, DFFunctionTransformer


	# ### Dummy transformation pipelines

	# In[9]:


	# channel categorical: dummy transformation #TODO: remove redundant column
	pip_channel = Pipeline([('extract_channel', ColumnExtractor(['channel'])),
							('get_channel_dummies', DummyTransformer())
						   ])

	# residence categorical: dummy transformation #TODO: remove redundant column
	pip_residence = Pipeline([('extract_residence', ColumnExtractor(['residence'])),
							  ('get_residence_dummies', DummyTransformer())
							 ])



	# ### Numerical transformation pipeline

	# In[10]:


	# Numerical pipeline: components:
	pip_score = Pipeline([('select_score', ColumnExtractor(['score'])),
						  ('imp_score', DFImputer(strategy='mean')),
						  ('exp_stretch', DFFunctionTransformer(lambda x: np.exp(x/100)))
						 ])

	pip_late_counts = Pipeline([('select_late_counts', ColumnExtractor(['late3', 'late6', 'late12'])),
								('imp_late_counts', DFImputer(strategy='most_frequent'))
							   ])

	pip_logtrans_income = Pipeline([('select_income', ColumnExtractor(['Income'])),
									('logtrans_income', Log1pTransformer())
								   ])

	pip_others = Pipeline([('extractor', ColumnExtractor(['cash_credit', 'age_in_days', 'prems_paid', 'premium']))
						  ])

	# construct numericals pipeline
	pip_numericals = DFFeatureUnion([('pip_score', pip_score),
						  ('pip_late_counts', pip_late_counts),
						  ('pip_logtrans_income', pip_logtrans_income),
						  ('others', pip_others)])


	# ### Combining all pipelines

	# In[11]:


	pip = DFFeatureUnion([('pip_channel', pip_channel),
						  ('pip_numericals', pip_numericals),
						  ('pip_residence', pip_residence)
						 ])


	# In[12]:


	# from sklearn.cross_validation import Bootstrap
	# b = Bootstrap(6, n_bootstraps=3, n_train=1.0, random_state=42)


	# In[13]:




	X_train_trans = pip.fit_transform(X_train)
	X_train_trans.head()


	# ## Define AUC ROC curve plot function

	# In[14]:


	#X_train.to_csv("train_imputed_encoded.csv")
	#X_train.describe()

	def plot_curve(fpr, tpr, roc_auc, title):
		plt.figure()
		lw = 2
		plt.plot(fpr, tpr, color='darkorange',
				 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title(title)
		plt.legend(loc="lower right")
		plt.show()
		# return plt



	# # Train Model

	# In[19]:


	from sklearn.linear_model import LogisticRegression
	from xgboost import XGBClassifier
	from sklearn.model_selection import GridSearchCV


	# In[20]:


	m = XGBClassifier(eval_metric='auc', objective='binary:logistic')
	# m = LogisticRegression()


	# In[23]:


	# XGBoost with grid search

	params={
		'max_depth': [2,4,6,8], # 5 is good but takes too long in kaggle env
		'subsample': [0.4,0.6,0.8,1.0],
		'colsample_bytree': [0.5,0.6,0.8],
		'n_estimators': [1000,2000,3000],
		'reg_alpha': [0.01, 0.02, 0.04]
	}


	rs = GridSearchCV(m,
					  params,
					  cv=2,
					  scoring="roc_auc",
					  n_jobs=3,
					  verbose=2)
	rs.fit(X_train_trans, y_train)
	best_est = rs.best_estimator_
	print(best_est)

	
if __name__ == '__main__':
	main()
