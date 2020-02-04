import numpy as np
import pandas as pd
import tarfile
import sys
import os
import time
from itertools import combinations
from pprint import pprint
from sklearn import linear_model
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from scipy.stats import norm, skew 
from sklearn.metrics import (
    mean_squared_error,
    mean_squared_log_error,
    r2_score
)
import matplotlib.pyplot as plt
import math

headers = []

ord_dict = {
    "Manhattan": 1,
    "Euclidean": 2
}
features_per_header = {}
data={}
DATA = []

min_max_list = [] 
exclude_col = None
meanHour = 0
meanAge = 0

data_dir = os.path.join(os.curdir, "hw2-data")

NUMERICAL_NA = 0.000000001

def positive_ratio(data):
    length = len(data)

    target_col = [row[-1] for row in data]
    num_pos = sum(list(filter(lambda x: x == 1, target_col)))

    return round(num_pos / length * 100, 2)

def add_bias(X):
    ones = np.ones(X.shape[0]).reshape(X.shape[0],1)
    return np.append(ones, X, axis=1)

def read_file(path, start_pos=1):
    headers = []
    content = []

    with open(path) as f:
        for l, line in enumerate(f):
            lineData=line[:-1].split(",")
            lineData = lineData[start_pos:]
            if l == 0:
                headers = lineData
            else:
                for c, cell in enumerate(lineData):
                    try:
                        lineData[c] = float(cell)
                    except:
                        pass
                content.append(lineData)
    return (headers, content)

def write_submission(predictions, title):
    (headers, data) = read_file(os.path.join(data_dir, "test.csv"), 0)

    if "output" not in os.listdir():
        os.mkdir(os.path.join(os.curdir, "output"))

    with open(os.path.join(os.curdir, "output",  f"kaggle_submission_{title}_regr.csv"), "w") as f:
        f.write("Id,SalePrice\n")
        for p, prediction in enumerate(predictions):
            f.write(f"{int(data[p][0])},{prediction}\n")

def binarize(train, dev, isFullyBinarized, mean = False, variance = False):
    # exclude_col  = [len(headers)-1]
    # (cat_data, num_data) = seperate_columns(train, exclude_col)
    (all_data_train, type_data_train) = defineType([row[:-1] for row in train], isFullyBinarized)
    global min_max_list
    for i in range(len(type_data_train)):
        if type_data_train[i] == "numerical":
            min_max_list.append(min_max(all_data_train, i))
        else:
            min_max_list.append(None)

    (all_data_dev, type_data_dev) = defineType([row[:-1] for row in dev], isFullyBinarized)

    (bin_norm_train, bin_norm_dev, mapping) = setup(all_data_train, all_data_dev, type_data_train, mean, variance)

    x_train = bin_norm_train#add_bias(bin_norm_train)
    x_dev = bin_norm_dev#add_bias(bin_norm_dev)  

    return (x_train, x_dev, mapping)

def defineType(data, isFullyBinarized):
    combined = []
    type_data = []
    all_data = []

    if isFullyBinarized:
        return (data, ["categorical" for i in range(len(data[0]))])

    for c in range(len(data[0])):
        column = [row[c] for row in data]

        isCategorical = False
        hasNA = False
        for i, item in enumerate(column):
            try:
                float(item)
            except:
                if item == "NA":
                    hasNA = True
                else:
                    isCategorical = True
                    break
        
        if hasNA and not isCategorical:
            for i, item in enumerate(column):
                if item == "NA":
                    column[i] = NUMERICAL_NA

        type_data.append("categorical" if isCategorical else "numerical")

        combined.append(column)
    print(len(combined[0]))
    for r in range(len(combined[0])):
        row = []
        for c in range(len(combined)):
            row.append(combined[c][r])
        all_data.append(row)        

    return (all_data, type_data)

def run_regression(data, isBinarized, mean, variance, headers, regrOptions={"name": "linear"}):
    train = data["train"]
    dev = data["dev"]
    kfolds = KFold(n_splits=10, shuffle=True, random_state=1)
    (x_train, x_dev, mapping) = binarize(train, dev, isBinarized, mean, variance)

    y_train = np.log(np.asarray([row[-1] for row in train]))
    y_dev = np.log(np.asarray([row[-1] for row in dev]))

    header_mapping = headers#header_to_newCol(headers, mapping)

    models = {
        "linear": linear_model.LinearRegression(),
        "ridge" : linear_model.RidgeCV(cv=kfolds),
        "lasso" : linear_model.LassoCV(cv=kfolds),
        "GradientBoostingRegressor" :  GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =5),
        "elastic": linear_model.ElasticNet(alpha=0.1, copy_X=True, fit_intercept=True,
        l1_ratio=0.5, max_iter=1000, normalize=False, random_state=None,
        selection='cyclic', tol=0.0001, warm_start=False),
        "SGD": linear_model.SGDRegressor()
    }

    if regrOptions["name"] == "linear":
        regr = linear_model.LinearRegression()
    
    elif regrOptions["name"] == "ridge":
        regr = linear_model.RidgeCV(  cv=kfolds) #alpha = regrOptions["alpha"])

    elif regrOptions["name"] == "poly":
        new_matrix = []
        header_mapping = header_to_newCol(headers, mapping)
        headersToInclude = ["GrLivArea", "LotArea"]
        new_train_matrix = add_polynomial_columns(header_mapping, headersToInclude, x_train)
        new_dev_matrix = add_polynomial_columns(header_mapping, headersToInclude, x_dev)
                
        new_train_matrix = Pipeline(
            [("polynomial_features", preprocessing.PolynomialFeatures(degree=2, include_bias=False))]
        ).fit_transform(new_train_matrix)

        new_dev_matrix = Pipeline(
            [("polynomial_features", preprocessing.PolynomialFeatures(degree=2, include_bias=False))]
        ).fit_transform(new_dev_matrix)

        x_train = np.concatenate((np.asarray(x_train), new_train_matrix), axis=1)
        print(x_dev.shape, new_dev_matrix.shape)
        x_dev = np.concatenate((np.asarray(x_dev), new_dev_matrix), axis=1)

        regr = linear_model.LinearRegression()

    elif regrOptions["name"] == "lasso":
        regr = linear_model.LassoCV(cv=kfolds)
    
    elif regrOptions["name"] == "GradientBoostingRegressor":
        regr = GradientBoostingRegressor(n_estimators = 3000, learning_rate = 0.05, max_depth = 4, max_features = 'sqrt',
                min_samples_leaf = 15, min_samples_split = 10, 
                loss='huber', random_state = 5)
    
    elif regrOptions["name"] == "MultiTaskElasticNet":
        regr = linear_model.MultiTaskElasticNet()
    
    elif regrOptions["name"] == "stacking":
        predictions = pd.DataFrame()
        test_predictions = pd.DataFrame()
        model = linear_model.LinearRegression()
        for name, model in models.items():
            if name == "ridge":
                continue

            print(f"x_train: {pd.DataFrame(x_train).shape}")
            print(f"y_train: {pd.DataFrame(y_train).shape}")

            (test_pred, train_pred) = stacking(model = model, n_fold = 10, xtrain = pd.DataFrame(x_train), 
            x_dev = pd.DataFrame(x_dev), ytrain = pd.DataFrame(y_train))

            print(f"test_pred: {pd.DataFrame(test_pred).shape}")
            print(f"train_pred: {pd.DataFrame(train_pred).shape}")
            
            predictions = pd.concat([predictions, pd.DataFrame(train_pred)], axis=1)
            print(f"predictions: {predictions.shape}")
            test_predictions = pd.concat([test_predictions, pd.DataFrame(test_pred)], axis=1)
            print(test_predictions.shape)
            print("print")
        regr = models["ridge"]
        regr.fit(predictions, pd.DataFrame(y_train))
        #score = regr.score(test_predictions, pd.DataFrame(y_dev))
        #print(score)
        print((test_predictions).shape)
        rmsle = np.sqrt(mean_squared_error(test_predictions, y_dev))
        print("Whats up doc")
        print(score)
        print(rmsle)
        return (rmsle, regr)

    elif regrOptions["name"] == "average": 
        models["linear"].fit(x_train, y_train)
        # models["lasso"].fit(x_train, y_train)
        models["GradientBoostingRegressor"].fit(x_train, y_train)
        models["ridge"].fit(x_train, y_train)
        models["SGD"].fit(x_train, y_train)

        pred1=models["linear"].predict(x_dev)
        # pred2=models["lasso"].predict(x_dev)
        pred3=models["GradientBoostingRegressor"].predict(x_dev)
        pred4=models["ridge"].predict(x_dev)
        pred5=models["SGD"].predict(x_dev)

        predict = (pred1+pred3+pred4+pred5)/4 #+pred2
        rmsle = np.sqrt(mean_squared_error(y_dev, predict))
        print(rmsle)
        return (rmsle, models, x_train, y_train)
        

    if regrOptions["name"] == "GradientBoostingRegressor":
        regr.fit(x_train, y_train)
        regr.score(x_train, y_train)

    else:
        regr.fit(x_train, y_train)
        coef = regr.coef_

    y_pred = regr.predict(x_dev)

    rmsle = np.sqrt(mean_squared_error(y_dev, y_pred))

    if regrOptions["name"] == "linear":
        top_bottom(mapping, headers, coef, 10)
    print('Intercept: \n', regr.intercept_)

    return (rmsle, regr)

def add_polynomial_columns(header_mapping, headersToInclude, data):
    new_matrix = []
    
    for r, row in enumerate(data):
        row_data = []

        for c, col in enumerate(row):
            if header_mapping[c] in headersToInclude:
                row_data.append(col)

        new_matrix.append(row_data)

    return new_matrix

def header_to_newCol(headers, mapping):
    header_mapping = {}

    for key, value in mapping.items():
        v = value + 1
        ogColHeader = headers[key[0]]

        if v not in header_mapping:
            header_mapping[v] = ogColHeader

    header_mapping[0] = "bias"

    return header_mapping

def stacking(model, xtrain, ytrain, x_dev, n_fold):
    folds = KFold(n_splits=n_fold, random_state=1)
    test_pred = np.array([])#np.empty((x_dev.shape[0],1), float)
    #print(test_pred)
    train_pred = np.array([])#np.empty((0,1), float)
    for train_indices, val_indices in folds.split(xtrain, ytrain.values):
        x_train, x_val = xtrain.iloc[train_indices], xtrain.iloc[val_indices]
        y_train, y_val = ytrain.iloc[train_indices], ytrain.iloc[val_indices]
        model.fit(X = x_train, y = y_train)
        try:
            train_pred = np.append(train_pred, model.predict(x_val), axis=0)
            test_pred = np.append(test_pred,model.predict(x_dev), axis=0)
        except:
            train_pred = np.append(train_pred, model.predict(x_val))
            test_pred = np.append(test_pred,model.predict(x_dev))
    print(test_pred.shape)
    print("Made It Here")
    return (test_pred, train_pred)#(test_pred.reshape(-1,1), train_pred)

def main():
    global headers, exclude_col

    try:
        tar = tarfile.open(name="hw3-data.tgz", mode='r:gz')
        tar.extractall(path=".", members=None, numeric_owner=False)
    except:
        print("Did not work")

    files_to_read = ["my_train", "my_dev", "test"]
    rename = ["train", "dev", "test"]
    for i, fname in enumerate(files_to_read):
        path = os.path.join(data_dir, f"{fname}.csv")
        (header_list, content) = read_file(path)

        if i == 0:
            headers = header_list

        data[rename[i]] = content
    #================================================================
    #Binarize Data
    #================================================================
    isBinarized = True
    mean = False
    deviation = False # Don't touch

    rmsle, regr = run_regression(data, isBinarized, mean, deviation, headers)

    #================================================================
    #Part 2
    #================================================================
    #Question 1 & 2
    print_num_features([True for item in data["train"][0]], data["train"])

    print(rmsle) #answer to question 4

    #Question 6
    (x_train, x_test, mapping) = binarize(data["train"], data["test"], isBinarized, mean, deviation)

    predictions = np.exp(regr.predict(x_test))

    write_submission(predictions, "linear_full")

    # ================================================================
    # Part 3
    # ================================================================
    predictions_avg = []
    isBinarized = False
    mean = True
    deviation = False # Don't touch

    rmsle, regr = run_regression(data, isBinarized, mean, deviation, headers)

    #Question 1 & 2
    # print_num_features([True for item in data["train"][0]], data["train"])

    print(rmsle) #answer to question 4

    #Question 6
    (x_train, x_test, mapping) = binarize(data["train"], data["test"], isBinarized, mean, deviation)

    predictions = np.exp(regr.predict(x_test))
    predictions_avg.append(predictions)

    write_submission(predictions, "linear_partial")

    # ================================================================
    # Part 4
    # ================================================================
    isBinarized = False
    mean = True
    deviation = False # Don't touch
    regrOptions = {
        "name": "ridge",
        "alpha": 0.001
    }
    print("Ridge Regression")

    rmsle, regr = run_regression(data, isBinarized, mean, deviation, headers, regrOptions)
    
    print(rmsle) #answer to question 4

    #Question 6
    (x_train, x_test, mapping) = binarize(data["train"], data["test"], isBinarized, mean, deviation)

    predictions = np.exp(regr.predict(x_test))
    predictions_avg.append(predictions)

    write_submission(predictions, "ridge_partial")

#     Part 4 Question 2
#     ================================================================

# =======================# Writing is broken ==============================
    isBinarized = False
    mean = True
    deviation = False # Don't touch
    regrOptions = {
        "name": "poly",
        "degree": 2,
        "inter": False
    }

    print("Polynomial Regression")

    rmsle, regr = run_regression(data, isBinarized, mean, deviation, headers, regrOptions)

    print(rmsle) #answer to question 4

    # #Question 6
    # (x_train, x_test, mapping) = binarize(data["train"], data["test"], isBinarized, mean, deviation)

    # predictions = np.exp(regr.predict(x_test))

    # write_submission(predictions, "poly_partial")

    # =============== Combine Columns ============================================================
    
    # combo = [12,13]#[3,45]#
    # data2 = {}

    # isnumerical = False
    # operation = np.subtract

    # # data2["train"] = combine_columns(data["train"], combo, isnumerical, operation)
    # # data2["dev"] = combine_columns(data["dev"], combo, isnumerical, operation)
    # # data2["test"] = combine_columns(data["test"], combo, isnumerical, operation)

    # combo = [19,18]#[3,44]#
    # data2 = {}

    # isnumerical = True

    # data2["train"] = combine_columns(data["train"], combo, isnumerical, operation)
    # data2["dev"] = combine_columns(data["dev"], combo, isnumerical, operation)
    # data2["test"] = combine_columns(data["test"], combo, isnumerical, operation)

    # isBinarized = False
    # mean = True
    # deviation = False # Don't touch
    # # regrOptions = {
    # #     "name": "ridge",
    # #     "alpha": 0.001
    # # }

    # rmsle, regr = run_regression(data2, isBinarized, mean, deviation, headers, regrOptions)

    # print(rmsle) #answer to question 4

    # # #Question 6
    # (x_train, x_test, mapping) = binarize(data2["train"], data2["test"], isBinarized, mean, deviation)

    # predictions = np.exp(regr.predict(x_test))

    # write_submission(predictions, "combined_partial")


    # =============== lasso Columns ============================================================
    isBinarized = False
    mean = True
    deviation = False # Don't touch
    regrOptions = {
        "name": "lasso",
        "alpha": 1
    }

    print("Lasso Regression")

    rmsle, regr = run_regression(data, isBinarized, mean, deviation, headers, regrOptions)

    print(rmsle) #answer to question 4

    # #Question 6
    (x_train, x_test, mapping) = binarize(data["train"], data["test"], isBinarized, mean, deviation)

    predictions = np.exp(regr.predict(x_test))

    write_submission(predictions, "lasso_partial")


    # Finding the best
    (data["train"], headers1) = drop_rows(data["train"], 45, headers, True, True)
    (data["dev"], headers2) = drop_rows(data["dev"], 45, headers, False, True)
    (data["test"], headers3) = drop_rows(data["test"], 45, headers[:-1], False, False)
    headers = headers1
    
    isnumerical = False
    operation = np.subtract

    column_names = ['Street',
    'BsmtFinSF1', 'BsmtFinSF2', '1stFlrSF','TotRmsAbvGrd','2ndFlrSF', 'Utilities']#, 'FullBath', 'HalfBath','BsmtFullBath','BsmtHalfBath']#, 'Electrical','Heating', 'HeatingQC', 'CentralAir' ]
    columns = [headers.index(col) for col in column_names]
    (_data, _headers) = drop_combine_columns(data, columns, headers, "drop", isnumerical)

    # isnumerical = True
    # operation = np.add

    # column_names = ['YrSold', 'MoSold']#, 'Condition2']
    # combo = [_headers.index(col) for col in column_names] #combo = [11,12]#[3,45]
    # (_data, _headers) = drop_combine_columns(_data, combo, _headers, "combine", isnumerical, operation)

    # # # print(_headers)

    # column_names = ['Neighborhood', 'Condition1']
    # combo = [new_headers.index(col) for col in column_names]
    # isnumerical = True
    # operation = np.subtract

    # column_names = ['YearBuilt', 'YearRemodAdd']#['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    # combo = [_headers.index(col) for col in column_names] #combo = [11,12]#[3,45]
    # isnumerical = True
    # (_data, _headers) = drop_combine_columns(_data, combo, _headers, "combine", isnumerical, operation)

    # isnumerical = False
    # operation = np.add

    # column_names =['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    # combo = [_headers.index(col) for col in column_names] #combo = [11,12]#[3,45]
    # isnumerical = True
    # (_data, _headers) = drop_combine_columns(_data, combo, _headers, "combine", isnumerical, operation)

    # combo = [19,18]#[3,44]#
    # data2 = {}

    # 

    # data2["train"] = combine_columns(data["train"], combo, isnumerical, operation)
    # data2["dev"] = combine_columns(data["dev"], combo, isnumerical, operation)
    # data2["test"] = combine_columns(data["test"], combo, isnumerical, operation)

    isBinarized = False
    mean = True
    deviation = False # Don't touch
    regrOptions = {
        "name": "average"#,
        #"alpha" : 0.0001 #GradientBoostingRegressor
    }
    print("Average")#MultiTaskElasticNetCV Regression")

    (rmsle, regr, x_train, y_train) = run_regression(_data, isBinarized, mean, deviation, _headers, regrOptions)
    
    # print(rmsle) #answer to question 4

    #Question 6
    (x_train, x_test, mapping) = binarize(_data["train"], _data["test"], isBinarized, mean, deviation)

    # header_mapping = header_to_newCol(_headers, mapping)

    # headersToInclude = ["GrLivArea", "LotArea"]
    # new_test_matrix = add_polynomial_columns(header_mapping, headersToInclude, x_train)
            
    # new_test_matrix = Pipeline(
    #     [("polynomial_features", preprocessing.PolynomialFeatures(degree=2, include_bias=False))]
    # ).fit_transform(new_test_matrix)
    # x_train2 = np.concatenate((np.asarray(x_train), new_test_matrix), axis=1)

    # header_mapping = header_to_newCol(_headers, mapping)

    # headersToInclude = ["GrLivArea", "LotArea"]
    # new_test_matrix = add_polynomial_columns(header_mapping, headersToInclude, x_test)
    # y_train = np.log(np.asarray([row[-1] for row in _data["train"]]))
    # new_test_matrix = Pipeline(
    #     [("polynomial_features", preprocessing.PolynomialFeatures(degree=2, include_bias=False))]
    # ).fit_transform(new_test_matrix)
    # x_test2 = np.concatenate((np.asarray(x_test), new_test_matrix), axis=1)
    # regr = linear_model.LinearRegression()
    # regr.fit(x_train2, y_train)
    # y_pred = regr.predict(x_test2)
    # print(y_pred)

    # # FOR AVERAGE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    kfolds = KFold(n_splits=10, shuffle=True, random_state=1)
    # (x_train, x_test, mapping) = binarize(_data["train"], _data["test"], isBinarized, mean, deviation)
    models = {
        "linear": linear_model.LinearRegression(),
        "ridge" : linear_model.RidgeCV(cv=kfolds),
        "lasso" : linear_model.LassoCV(cv=kfolds),
        "GradientBoostingRegressor" :  GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =5),
        "SGD": linear_model.SGDRegressor()
    }

    y_train = np.log(np.asarray([row[-1] for row in _data["train"]]))

    models["linear"].fit(x_train, y_train)
    # models["SGD"].fit(x_train, y_train)
    models["GradientBoostingRegressor"].fit(x_train, y_train)
    models["ridge"].fit(x_train, y_train)


    pred1=models["linear"].predict(x_test)
    # pred2=models["SGD"].predict(x_test)
    pred3=models["GradientBoostingRegressor"].predict(x_test)
    pred4=models["ridge"].predict(x_test)

    # train_pred1=pd.DataFrame(pred1)
    # train_pred2=pd.DataFrame(pred2)
    # train_pred3=pd.DataFrame(pred3)

    # # df = pd.concat([train_pred1, train_pred2, train_pred3], axis=1)
    # # print(df.shape)
    # # models["ridge"].fit(df,y_train)
    # # predictions = models["ridge"].predict(x_test)

    predictions = (np.exp(pred1)+np.exp(pred3)+np.exp(pred4))/3#+np.exp(pred2)
    print(rmsle)

    # regr = linear_model.LinearRegression()
    # regr.fit(x_train, y_train)
    # y_pred = regr.predict(x_test)
    # print(y_pred)
    # print(x_test.shape)
    # sys.exit()
    # pred = []
    # for model in regr.values():
    #     model.fit(x_train,y_train)
    #     pred.append(model.predict(x_test))

    # print(pred[0])
    # sys.exit()
    # predictions = (pred[0]+pred[1]+pred[2]+pred[3])/4
    print(predictions) 

    #predictions = np.exp(regr.predict(x_test))

    write_submission(predictions, "stacked_partial")

def drop_rows(data, column, headers, train, hasY):
    df2 = pd.DataFrame(data)

    df2.columns = headers[:]
    if train:
        df2 = df2[df2['GrLivArea']<4000]
    #     #df2 = df2.drop(df2[(df2['GrLivArea']>4000) & (df2['SalePrice']<300000)].index)
    # df2['MSZoning'] = df2['MSZoning'].fillna(df2['MSZoning'].mode()[0])
    # df2["Functional"] = df2["Functional"].fillna("Typ")
    # df2['Electrical'] = df2['Electrical'].fillna(df2['Electrical'].mode()[0])
    # df2['KitchenQual'] = df2['KitchenQual'].fillna(df2['KitchenQual'].mode()[0])
    # df2['Exterior1st'] = df2['Exterior1st'].fillna(df2['Exterior1st'].mode()[0])
    # df2['Exterior2nd'] = df2['Exterior2nd'].fillna(df2['Exterior2nd'].mode()[0])
    # df2['SaleType'] = df2['SaleType'].fillna(df2['SaleType'].mode()[0])
    # # df2['YrSold'] = df2['YrSold'].astype(str)
    # df2['MoSold'] = df2['MoSold'].astype(str)
    df2['LotFrontage']=df2['LotFrontage'].apply(lambda x: df2['LotFrontage'][~(df2['LotFrontage']=='NA')].median() if "NA" else x)
    for name in ['BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF',
    'FullBath','HalfBath', 'BsmtFullBath','BsmtHalfBath']:#, 'PoolArea']:#, '2ndFlrSF', 'GarageArea', 'TotalBsmtSF', 'Fireplaces']:
    #,'OpenPorchSF','3SsnPorch','EnclosedPorch',
    #'ScreenPorch','WoodDeckSF']:#
        df2[name]=df2[name].apply(lambda x: 0 if "NA" else x)
        df2[name]= pd.Series(df2[name],dtype=np.float64)
    df2['Total_sqr_footage'] = (
        df2['BsmtFinSF1'] + df2['BsmtFinSF2'] + 
        df2['1stFlrSF'] + df2['2ndFlrSF']
    )
    df2['Total_Bathrooms'] = (
        df2['FullBath'] + (0.5 * df2['HalfBath']) +
        df2['BsmtFullBath'] + (0.5 * df2['BsmtHalfBath'])
    )
    # df2['Total_porch_sf'] = (df2['OpenPorchSF'] + df2['3SsnPorch'] +
    #     df2['EnclosedPorch'] + df2['ScreenPorch'] +
    #     df2['WoodDeckSF'])
    # df2['Total_porch_sf'] = (df2['OpenPorchSF'] + df2['3SsnPorch'] +
    #         df2['EnclosedPorch'] + df2['ScreenPorch'] +
    #         df2['WoodDeckSF']
    # )
    if hasY:
        df2 = df2[[c for c in df2 if c not in ['SalePrice']] 
        + ['SalePrice']]

    return (df2.to_numpy(), list(df2.columns))

def drop_combine_columns(data, columns, headers, option, isNumerical, operation=np.add):
    h = headers[:]
    isDrop = True if option == "drop" else False
    new_headers = []
    new_data = {}

    data_names = ["train", "dev", "test"]
    for d in data_names:
        isTest = True if d == "test" else False
        h = headers[:-1] if isTest else headers

        if isDrop:
            (new_data[d], _headers) = drop_columns(data[d], columns, h)
        else:
            (new_data[d], _headers) = combine_columns(data[d], columns, isNumerical, operation, h)

        if d == "train":
            new_headers = _headers

    return (new_data, new_headers)

def columns_to_rows(data):
    new_data = []

    for r in range(len(data[0])):
        new_row = []

        for c, col in enumerate(data):
            new_row.append(data[c][r])
        
        new_data.append(new_row)

    return new_data

def get_feature_map(data, type_data):
    mapping = {}
    new_data = []
    b = [True if item == "categorical" else False for item in type_data]
    # print_num_features(b, data)

    # print(len([True for item in b if item]))
    for c in range(len(data[0])):
        column = [row[c] for row in data]
        new_col = []
        for r, row in enumerate(column):
            feature = (c, row)
            if feature not in mapping:
                if type_data[c] == "numerical":
                    if c == 0:
                        mapping[feature] = 0
                    else:
                        mapping[feature] = max([v for k, v in mapping.items() if k[0] == c-1]) + 1
                else:
                    if c == 0 and r == 0:
                        mapping[feature] = 0
                    else:
                        column_entered = [v for k, v in mapping.items() if k[0] == c]                    
                        if len(column_entered) == 0:
                            mapping[feature] = max([v for k, v in mapping.items() if k[0] == c-1]) + 1
                        else:
                            mapping[feature] = max([v for k, v in mapping.items() if k[0] == c]) + 1
            new_col.append(mapping[feature])
        new_data.append(new_col)  

    new_data = columns_to_rows(new_data)

    return (mapping, new_data)

def apply_feature_map(mapping, data, type_data):
    new_data = []
    for row in data:
        new_row = []
        for i, x in enumerate(row):
            feature = (i,x)
            if feature in mapping.keys():
                new_row.append(mapping[feature])
            else:
                if type_data == "categorical":
                    new_row.append(-1)
                else:
                    new_row.append([v for k, v in mapping.items() if k[0] == i][0])
        new_data.append(new_row)

    # column_count = len(cat_data[0])-1
    # for row_idx, row in enumerate(num_data):
    #     for col_idx, col in enumerate(row):
    #         feature = (column_count + 1 + col_idx, row[col_idx])
    #         if feature in mapping.keys():
    #             new_data[row_idx].append(mapping[feature])
    #         else:
    #             new_data[row_idx].append(-1)
    return new_data

def norm_bin(newData, data, type_data, param, mapping, normType="norm", mean = False, variance = False):
    bindata = np.zeros(param)
    for i, row in enumerate(newData):
        for x, col in enumerate(row):
            if type_data[x] == "numerical":
                if col != NUMERICAL_NA:
                    bindata[i][col] = data[i][x] #normalize(data[i][x], x, normType)
            elif col == -1: 
                bindata[i][col] = 0
            else:
                bindata[i][col] = 1

    # new_to_old = {}
    # for k, v in mapping.items():
    #     if v not in new_to_old:
    #         new_to_old[v] = k[0]

    if mean:
        feat_mean = np.mean(bindata,0)   # mean along 0 axis (vertical)
        bindata = bindata - feat_mean
        # for c in range(len(bindata[0])):
        #     if type_data[new_to_old[c]] == "categorical":
        #         continue
        #     for r, row in enumerate(bindata):
        #         bindata[r][c] = bindata[r][c] - feat_mean[c]
    if variance:
        feat_std  = np.std(bindata,0)
        feat_std[feat_std < 1e-8] = 1.0
        bindata = bindata / feat_std
    
    # bindata[:,0] = 1.0

    # print(bindata[0])
    # print(len(bindata[0]))
    # sys.exit()
    return bindata

def normalize(value, i, normType):
    if normType == "norm":
        min_ = min_max_list[i][0]
        max_ = min_max_list[i][1]
        if value < min_:
            temp = min_
        elif value > max_:
            temp = max_
        else:
            temp = value
        return (float(temp)- min_) / (max_ - min_)
    # elif normType == "zeroMean":
    #     if i == 0:
    #         return value - meanAge
    #     return value - meanHour
    # elif normType == "unit":
    #     value = int(cell)
    #     value = int(cell)
    #     min_ = min_max_list[i][0]
    #     max_ = min_max_list[i][1]
    #     if value < min_:
    #         temp = min_
    #     elif value > max_:
    #         temp = max_
    #     else:
    #         temp = value
    #     return (int(temp)- min_) / (max_ - min_)

def min_max(data, i):
    ordered=sorted(data, key=lambda row: row[i])
    return (int(ordered[0][i]),int(ordered[-1][i]))

def map_target(e):
    return DATA[e][-1]

def print_num_features(isBinary, data):
    message = "Fully Binarized" if all(isBinary) else "Partially Binarized"

    for i in range(len(headers)-1):
        column = [row[i] for row in data]
        if isBinary[i]:
            num_features = len(np.unique(np.asarray(column)))
        else:
            num_features = 1
    
        features_per_header[headers[i]] = num_features

    total_feature = 0
    print(message + ":")
    for header, num_features in features_per_header.items():
        print(f"\t{header} has {num_features} features")
        total_feature += num_features

    print(f"Total features {message} variables: {total_feature}")

def setup(trainData, devData, type_data,  mean, variance, normType="norm", combinedCol=None):
    """Train"""
    (mapping, new_data) = get_feature_map(trainData, type_data)
    param = (len(trainData), max(mapping.values())+1)
    bin_norm_train = norm_bin(new_data, trainData, type_data, param, mapping, normType, mean, variance)
    """Dev"""
    # devY = [row[-1] for row in devData]
    # (cat_data, num_data) = seperate_columns(devData, excludes)
    if combinedCol is not None:
        cat_data = combine_columns(devData, combinedCol)
    new_data = apply_feature_map(mapping, devData, type_data)
    param = (len(devData), max(mapping.values())+1)
    bin_norm_dev = norm_bin(new_data, devData, type_data, param, mapping, normType, mean, variance)

    return (bin_norm_train, bin_norm_dev, mapping)

def drop_columns(data, columns, headers):
    # Add column into data and remove old columns
    new_headers = []
    new_data_cols = []
    for c in range(len(data[0])):
        if c not in columns:
            # add into data
            column = [row[c] for row in data]
            new_data_cols.append(column)
            new_headers.append(headers[c])
    
    return (columns_to_rows(new_data_cols), new_headers)

def combine_columns(data, columns, isnumerical,operation, headers):
    new_headers = headers[:]
    if not isnumerical:
        combinedCol = np.asarray([row[columns[0]] for row in data])
        for r in columns[1:]:
            combinedCol = np.core.defchararray.add(combinedCol, np.asarray([row[r] for row in data]))
    else:
        combinedCol = np.asarray([row[columns[0]] for row in data])
        for r in columns[1:]:
            combinedCol = operation(combinedCol, np.asarray([row[r] for row in data]))

    new_header_name = " | ".join([headers[i] for i in columns])
    new_headers[columns[0]] = new_header_name
    for i in columns[1:]:
        new_headers.pop(i)
    # Add column into data and remove old columns
    new_data_cols = []
    for c in range(len(data[0])):
        if c not in columns:
            # add into data
            column = [row[c] for row in data]
            new_data_cols.append(column)
        elif c == columns[0]:
            new_data_cols.append(combinedCol)

    new_data = []
    for r in range(len(new_data_cols[0])):
        new_row_data = []
        for col in new_data_cols:
            new_row_data.append(col[r])
        new_data.append(new_row_data)

    return (new_data, new_headers)   

def get_column_combinations(data):
    indicies = [i for i in range(len(data[0]))]
    combos = []

    for i in range(2, len(data[0])):
        for c in list(combinations(indicies, i)):
            combos.append(c)

    return combos

def top_bottom(mapping, headers, coef, n):
    headerWithBias = headers #["bias"] + headers
    reverseMap = {}
    valueMap = {}

    for key, value in mapping.items():
        ogColHeader = headers[key[0]]
        featureValue = (ogColHeader, key[1])
        featureCompCol = value #+ 1
        if ogColHeader not in reverseMap:
            reverseMap[ogColHeader] = coef[featureCompCol]
        else:
            reverseMap[ogColHeader] += coef[featureCompCol]
        if featureValue not in valueMap:
            valueMap[featureValue] = coef[featureCompCol]
        else:
            valueMap[featureValue] += coef[featureCompCol]
    #valueMap[("bias", 1)] = coef[0]

    sortList = sorted([(key, value) for key,value in valueMap.items()], key=lambda k: k[1])
    print(f"\nBottom {n}:")
    print_coef_list(sortList[:n])
    print(f"\nTop {n}:")
    print_coef_list(sortList[-n:])

    #print(f"\nBias Feature Weight: {round(valueMap[('bias', 1)],5)}")

def print_coef_list(coef_list):
    for coef in coef_list:
        print(f"\t{coef[0][0]}: {round(coef[1],5)}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Duration: {end_time - start_time:.2f}s")