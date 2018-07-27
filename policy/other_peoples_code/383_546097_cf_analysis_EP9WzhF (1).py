import numpy as np
import pandas as pd


#machine learning libraries
from bayes_opt import BayesianOptimization
import xgboost as xgb
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

iter_no = 5
gp_params = {'alpha': 1e-5}
cv_splits = 8


def treesCV(eta, gamma,max_depth,min_child_weight,subsample,colsample_bytree,n_estimators):
    #function for cross validation gradient boosted trees
    return cross_val_score(xgb.XGBRegressor(objective='binary:logistic',
    											tree_method = 'hist',
                                                learning_rate=max(eta,0),
                                                gamma=max(gamma,0),
                                                max_depth=int(max_depth),
                                                min_child_weight=int(min_child_weight),
                                                silent=True,
                                                subsample=max(min(subsample,1),0.0001),
                                                colsample_bytree=max(min(colsample_bytree,1),0.0001),
                                                n_estimators=int(n_estimators),
                                                seed=42,nthread=-1), X=X_train, y=y_train, scoring=obj_bo, cv=cv_splits, n_jobs=-1).mean()




def data_prep(data_df):

    #how to handle types
    data_df_num = data_df.select_dtypes(exclude=object)
    data_df_obj = data_df.select_dtypes(include=object)

    #how to handle nan
    data_df_num = data_df_num.fillna(data_df_num.mean())

    #get dummy variables
    data_df_obj = pd.get_dummies(data_df_obj, dummy_na=True)


    return pd.concat([data_df_num, data_df_obj],axis=1)

# incentive function, combining the given functions, first derivative set zero
def incentive_function(premium,bench):
    from scipy.special import lambertw
    res_val = float(-400*(-lambertw((2000*np.exp(2))/(premium*bench)) + np.log(1000/(premium*bench)) + 2))
    return res_val if res_val >= 0 else 0


if __name__ == '__main__':
    # reading data
    data_train = pd.read_csv('data/train.csv', sep=',')
    data_train = data_prep(data_train)

    data_pred = pd.read_csv('data/test.csv', sep=',')
    data_pred = data_prep(data_pred)


    #train test split doesnt actually split
    X_train, X_test, y_train, y_test = train_test_split(np.array(data_train.drop(['renewal','id'],axis=1)), np.array(data_train['renewal']), test_size=0, random_state=42)
    X_test = data_pred.drop(['id'],axis=1)



    #Bayesian Hyper parameter optimization of gradient boosted trees
    treesBO = BayesianOptimization(treesCV,{'eta':(0.001,0.4),
                                            'gamma':(8,12),
                                            'max_depth':(400,700),
                                            'min_child_weight':(0.1,1),
                                            'subsample':(0.3,0.6),
                                            'colsample_bytree':(0.6,1),
                                            'n_estimators':(600,800)})
    treesBO.maximize(n_iter=iter_no, **gp_params)
    tree_best = treesBO.res['max']

    #train tree with best paras
    trees_model = xgb.XGBRegressor(objective='binary:logistic',
    								tree_method = 'hist',
                                    seed=42,
                                    learning_rate=max(tree_best['max_params']['eta'],0),
                                    gamma=max(tree_best['max_params']['gamma'],0),
                                    max_depth=int(tree_best['max_params']['max_depth']),
                                    min_child_weight=int(tree_best['max_params']['min_child_weight']),
                                    silent=True,
                                    subsample=max(min(tree_best['max_params']['subsample'],1),0.0001),
                                    colsample_bytree=max(min(tree_best['max_params']['colsample_bytree'],1),0.0001),
                                    n_estimators=int(tree_best['max_params']['n_estimators']),nthread=-1)
    trees_model.fit(X_train, y_train)
    y_hat = trees_model.predict(np.array(X_test))



    # get incentives
    temp_list = []
    for i in range(len(X_test)):
        temp_list.append(incentive_function(X_test['premium'][i],y_hat[i]))

    submission = pd.DataFrame({'id':np.array(data_pred['id']),'renewal':y_hat,'incentives':temp_list})

    #write to file for submission
    submission.to_csv('submission.csv',sep=',', index=False)
