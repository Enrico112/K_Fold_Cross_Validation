from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

# define get score func
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# get dataset
digits = load_digits()

# split train and test dfs
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=.2)

# the scores  change every time we run train_test_split because dfs are randomly shuffled
# as a solution, K Fold
kf = KFold(n_splits=10)
kf
# kf.splits produces different n_splits (number selected above)
# (data is = range(10) for illustration only)
for train_index, test_index in kf.split(range(10)):
    print(train_index, test_index)

# models to test
models = [LogisticRegression(), SVC(), RandomForestClassifier()]
scores = []

# iterate over K fold splits
for train_index, test_index in kf.split(digits.data):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                       digits.target[train_index], digits.target[test_index]
    # iterate over models, print and store get_score
    for model in models:
        scores.append(get_score(model, X_train, X_test, y_train, y_test))
        print(get_score(model, X_train, X_test, y_train, y_test))


# cross val score does the swame thing as the first loop, default cv = 5
cross_val_score(LogisticRegression(), digits.data, digits.target)

# Stratified K Fold: divides classification categories uniformly across splits
skf = StratifiedKFold(n_splits=10)