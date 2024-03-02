import numpy as np
from numpy.linalg import inv

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

# Normalization
train_X_mean, train_X_std = train_X.mean(), train_X.std()
train_X_scaled = (train_X - train_X_mean) / train_X_std
test_X_scaled = (test_X - train_X_mean) / train_X_std

train_y_mean, train_y_std = train_y.mean(), train_y.std()
train_y_scaled = (train_y - train_y_mean) / train_y_std
test_y_scaled = (test_y - train_y_mean) / train_y_std

# RMSE CALCULATIO n
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

best_alpha = None
best_rmse = float('inf')
best_model = None


alphas = [0.01, 0.1, 1.0, 10.0]

# no. of folds
k = 15

#k-fold cross-validation
for alpha in alphas:
    fold_rmse = []
    fold_size = len(train_X_scaled) // k

    for i in range(k):
        val_indices = list(range(i * fold_size, (i + 1) * fold_size))
        train_indices = [idx for idx in range(len(train_X_scaled)) if idx not in val_indices]

        X_train_fold, y_train_fold = train_X_scaled[train_indices], train_y_scaled[train_indices]
        X_val_fold, y_val_fold = train_X_scaled[val_indices], train_y_scaled[val_indices]

        # Add polynomial features
        X_train_fold_poly = np.column_stack((X_train_fold, X_train_fold ** 2))
        X_val_fold_poly = np.column_stack((X_val_fold, X_val_fold ** 2))

        # Fit the model
        X_train_T_X_train = np.dot(X_train_fold_poly.T, X_train_fold_poly)
        X_train_T_y_train = np.dot(X_train_fold_poly.T, y_train_fold)
        theta = np.dot(inv(X_train_T_X_train + alpha * np.identity(X_train_T_X_train.shape[0])), X_train_T_y_train)

        # Make predictions on the validation set
        y_val_pred = np.dot(X_val_fold_poly, theta)

        # Calculate RMSE
        fold_rmse.append(rmse(y_val_pred, y_val_fold))

    avg_rmse = np.mean(fold_rmse)

    if avg_rmse < best_rmse:
        best_rmse = avg_rmse
        best_alpha = alpha

# Using training data
X_train_poly = np.column_stack((train_X_scaled, train_X_scaled ** 2))
X_test_poly = np.column_stack((test_X_scaled, test_X_scaled ** 2))
X_train_T_X_train = np.dot(X_train_poly.T, X_train_poly)
X_train_T_y_train = np.dot(X_train_poly.T, train_y_scaled)
theta = np.dot(inv(X_train_T_X_train + best_alpha * np.identity(X_train_T_X_train.shape[0])), X_train_T_y_train)

# Predictions on the test set
test_predictions = np.dot(X_test_poly, theta)


test_rmse = rmse(test_predictions, test_y_scaled)
print("Best alpha:", best_alpha)
print("Test RMSE:", test_rmse)
