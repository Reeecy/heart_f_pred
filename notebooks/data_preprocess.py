import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Loading the dataset
df = pd.read_csv('../data/heart.csv')


# Preprocess method
def preprocess(data=None):

    data = pd.read_csv('../data/heart.csv')
    print(f'Original data shape: {data.head()}')
    data = data.copy()

    # one-hot encoding the categorical columns
    encoder = OneHotEncoder(drop='first', sparse_output=False)  # drop first: drop the first column to avoid multicollinearity
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
