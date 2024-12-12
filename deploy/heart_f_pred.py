### PREPROCESS ###
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
### PREPROCESS ###

### MODELING ###
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
### MODELING ###

# EXTRAS #
from sklearn.metrics import accuracy_score, classification_report

# Creating custom XGBoost model

class CustomXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = xgb.XGBClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __sklearn_tags__(self):
        return self.model.__sklearn_tags__()


# Loading the dataset
df = pd.read_csv('../data/heart.csv')

# Preprocess function
def preprocess(data):
    data = data.copy()

    # one-hot encoding the categorical columns
    encoder = OneHotEncoder(drop='first', sparse_output=False)  # drop first: elkerüli a multikollinearitást
    encoded_columns = encoder.fit_transform(data[['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope']])

    # creating a new dataframe with the encoded columns
    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(
        ['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope']))

    # dropping the original columns
    data = data.drop(['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope'], axis=1)
    # concatenating the original and encoded dataframes
    data = pd.concat([data, encoded_df], axis=1)

    # splitting the data to train and test
    x = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']

    x_train, x_test, y_traing, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)


    # scaling the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_traing, y_test

# Preprocessing the data
x_train, x_test, y_train, y_test = preprocess(df)


# all the models I will use
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gaussion Naive Bayes': GaussianNB(),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
    'XGBoost': CustomXGBClassifier(),
}

# applying the models and getting the results
for name, model in models.items():
    pipeline = make_pipeline(model)

    # fitting the model
    pipeline.fit(x_train, y_train)

    # predicting the test data
    y_pred = pipeline.predict(x_test)

    # calculating the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name}: {accuracy}')
    print(classification_report(y_test, y_pred))
    print("-" * 55)
