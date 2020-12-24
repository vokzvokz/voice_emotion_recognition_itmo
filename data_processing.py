import numpy as np
from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def transform_data(X_train, X_test, y_train, y_test):
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))
    return X_train, X_test, y_train, y_test, lb


def split_df(df, test_size=0.2, transform=False):
    df = shuffle(df)
    X = np.array(df.iloc[:, 1:])
    y = np.array(df.iloc[:, 0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    if transform:
        return transform_data(X_train, X_test, y_train, y_test)
    else:
        return X_train, X_test, y_train, y_test
