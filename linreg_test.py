import statsmodels.api as sm
from sklearn import datasets

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


data = datasets.load_boston()

print(data)


df = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.DataFrame(data.target, columns=["MEDV"])
# df_norm = (df-df.mean)/df.std
# target_norm = (target-target.mean)/target.std


X = df#["RM"]
y = target["MEDV"]
#X = sm.add_constant(X)

foo = X.corr()
fig, ax = plt.subplots()
im = ax.imshow(foo)

0
# Note the difference in argument order
model = sm.regression.linear_model.OLS(y, X)
result = model.fit_regularized(alpha=5)
# model = sm.OLS(y, X).fit()
predictions = result.predict(X) # make the predictions by the model

# Print out the statistics




result.summary()

predictions.plot()
y.plot()

print(model)

