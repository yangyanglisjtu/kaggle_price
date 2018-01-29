from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np

default_preprocessor = CountVectorizer().build_preprocessor()

def rmsle(Y, Y_pred):
    # Y and Y_red have already been in log scale.
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))

def build_preprocessor(field, full_df):
    field_idx = list(full_df.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])

def vector_data(full_df):
    print("Vectorizing data...")

    vectorizer = FeatureUnion([
        ('name', CountVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            preprocessor=build_preprocessor('name',full_df))),
        ('category_name', CountVectorizer(
            token_pattern='.+',
            preprocessor=build_preprocessor('category_name',full_df))),
        ('brand_name', CountVectorizer(
            token_pattern='.+',
            preprocessor=build_preprocessor('brand_name',full_df))),
        ('shipping', CountVectorizer(
            token_pattern='\d+',
            preprocessor=build_preprocessor('shipping',full_df))),
        ('item_condition_id', CountVectorizer(
            token_pattern='\d+',
            preprocessor=build_preprocessor('item_condition_id',full_df))),
        ('item_description', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=100000,
            preprocessor=build_preprocessor('item_description',full_df))),
    ])
    return vectorizer

def ridge_prediction(train_df, dev_df, test_df):
    # Concatenate train - dev - test data for easy to handle
    full_df = pd.concat([train_df, dev_df, test_df])

    # Convert data type to string
    full_df['shipping'] = full_df['shipping'].astype(str)
    full_df['item_condition_id'] = full_df['item_condition_id'].astype(str)

    vectorizer = vector_data(full_df)

    X = vectorizer.fit_transform(full_df.values)
    n_trains = train_df.shape[0]
    n_devs = dev_df.shape[0]

    X_train = X[:n_trains]
    X_dev = X[n_trains:n_trains + n_devs]
    X_test = X[n_trains + n_devs:]

    Y_train = train_df.target.values.reshape(-1, 1)
    Y_dev = dev_df.target.values.reshape(-1, 1)


    print("Fitting Ridge model on training examples...")
    ridge_model = Ridge(
        solver='auto', fit_intercept=True, alpha=0.5,
        max_iter=100, normalize=False, tol=0.05,
    )
    ridge_model.fit(X_train, Y_train)

    Y_dev_preds_ridge = ridge_model.predict(X_dev)
    Y_dev_preds_ridge = Y_dev_preds_ridge.reshape(-1, 1)
    print("RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_ridge))

    ridge_preds = ridge_model.predict(X_test)
    ridge_preds = np.expm1(ridge_preds)
    return ridge_preds

