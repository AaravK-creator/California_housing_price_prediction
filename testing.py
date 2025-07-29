import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

#1. loading data
housing = pd.read_csv("housing.csv")

#2. create a stratified test set based on income cat
housing["income_cat"] = pd.cut(housing['median_income'], bins=[0,1.5,3.0,4.5,6.0, np.inf], labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat",axis=1)

#working on a copy of trained data
housing = strat_train_set.copy()

#3. separate predictors and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

#4. separate numerical and categorical values
num_attribs= housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribs= ["ocean_proximity"]

#5. pipeline
#numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

#categorical pipeline
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

#full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

#6. transform the data
housing_prepared = full_pipeline.fit_transform(housing)

#housing_prepared is now a numpy array ready for training
print(housing_prepared.shape)

#7. training model

#linear regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds= lin_reg.predict(housing_prepared)
#lin_rmse= root_mean_squared_error(housing_labels, lin_preds)
lin_rmses = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
#print(f"the root mean squared error for linear regression is {lin_rmse}")
print(pd.Series(lin_rmses).describe())

#decision tree model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_preds= dec_reg.predict(housing_prepared)
#dec_rmse= root_mean_squared_error(housing_labels, dec_preds)
dec_rmses = cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
#print(f"the root mean squared error for decision tree regression is {dec_rmse}")
print(pd.Series(dec_rmses).describe())

#random forest regression model
rf_reg = RandomForestRegressor()
rf_reg.fit(housing_prepared, housing_labels)
rf_preds= rf_reg.predict(housing_prepared)
#rf_rmse= root_mean_squared_error(housing_labels, rf_preds)
rf_rmses = cross_val_score(rf_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
#print(f"the root mean squared error for random forest regression is {rf_rmse}")
print(pd.Series(rf_rmses).describe())