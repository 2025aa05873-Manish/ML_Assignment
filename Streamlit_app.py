# Assume last column is target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode target if categorical
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# One-hot encode categorical features
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

