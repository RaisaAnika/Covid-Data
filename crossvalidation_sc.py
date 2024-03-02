import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Train data 
train_data_str = """16 392205
75 1094890
39 1517928
82 582237
29 334249
87 676876
52 379551
84 526186
24 295253
4 196150
91 899109
28 298348
90 834969
8 169032
27 304774
12 151876
15 342166
36 1162401
81 681984
70 359027
69 222579
25 245621
62 147811
2 176151
58 338407
64 99565
50 420739
1 94894
18 467901
53 417818
23 294946
57 406095
68 286510
73 886464
5 205165
79 852294
30 387634
26 280977
22 315788
65 80627
46 857250
60 233350
59 292324
11 149780
7 182246
43 1589955
38 1468482
49 465751
86 557180
47 670572
77 1022570
17 466267
76 1144251
44 1265861
89 751017
67 97037
3 219172
42 1667151
21 362662
19 431250
54 449960
66 87637
78 1037454
33 790094
80 755688
83 508543
61 181609
71 547086
34 1004845
6 198436
72 743056
10 146436
9 166325"""

# Test data 
test_data_str = """88 592394
56 467217
51 384000
40 1360122
31 449438
55 455950
35 1175483
45 1068625
63 99112
48 473485
37 1280932
14 258278
20 375909
32 559394
13 180021
74 1019966
41 1380995
85 510211"""

# Convert train and test data strings to numpy arrays
train_data = np.array([list(map(int, row.split())) for row in train_data_str.split('\n')])
test_data = np.array([list(map(int, row.split())) for row in test_data_str.split('\n')])


train_X, train_y = train_data[:, 0].reshape(-1, 1), train_data[:, 1]
test_X, test_y = test_data[:, 0].reshape(-1, 1), test_data[:, 1]

# Data normalization for train and test data
scaler_X_train = StandardScaler().fit(train_X)
scaler_y_train = StandardScaler().fit(train_y.reshape(-1, 1))
train_X_scaled = scaler_X_train.transform(train_X)
train_y_scaled = scaler_y_train.transform(train_y.reshape(-1, 1)).ravel()

scaler_X_test = StandardScaler().fit(test_X)
scaler_y_test = StandardScaler().fit(test_y.reshape(-1, 1))
test_X_scaled = scaler_X_test.transform(test_X)
test_y_scaled = scaler_y_test.transform(test_y.reshape(-1, 1)).ravel()

# Initialize KFold cross-validation
k = 15  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Model Training and evaluation
def train_and_evaluate_model(X_train, y_train, X_test, y_test, alpha):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, train_predictions))
    return model, test_rmse


best_alpha = None
best_rmse = float('inf')
best_model = None


alphas = [0.01, 0.1, 1.0, 10.0]

# Cross-validation
for alpha in alphas:
    fold_rmse = []
    for train_index, test_index in kf.split(train_X_scaled):
        X_train_fold, X_test_fold = train_X_scaled[train_index], train_X_scaled[test_index]
        y_train_fold, y_test_fold = train_y_scaled[train_index], train_y_scaled[test_index]

        model, rmse = train_and_evaluate_model(X_train_fold, y_train_fold, X_test_fold, y_test_fold, alpha)
        fold_rmse.append(rmse)

    avg_rmse = np.mean(fold_rmse)
    if avg_rmse < best_rmse:
        best_rmse = avg_rmse
        best_alpha = alpha
        best_model = model

print("Best alpha:", best_alpha)
print("Best RMSE:", best_rmse)

# Evaluate the best model on the test data
final_model, test_rmse = train_and_evaluate_model(train_X_scaled, train_y_scaled, test_X_scaled, test_y_scaled, best_alpha)
print("Test RMSE:", test_rmse)
