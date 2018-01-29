import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Rnn import rnn_prediction
from RidgeModel import ridge_prediction
# load train and test data
train_df = pd.read_table('../data/train.tsv')
test_df = pd.read_table('../data/test.tsv')
print(train_df.shape, test_df.shape)

# prepare data for processing by rnn and ridge
# Handle missing data.
def fill_missing_values(df):
    df.category_name.fillna(value="Other", inplace=True)
    df.brand_name.fillna(value="missing", inplace=True)
    df.item_description.fillna(value="None", inplace=True)
    return df

train_df = fill_missing_values(train_df)
test_df = fill_missing_values(test_df)

# Scale target variable to log.
train_df["target"] = np.log1p(train_df.price)

# Split training examples into train/dev examples.
train_df, dev_df = train_test_split(train_df, random_state=347, train_size=0.99)

def aggregate_predicts(Y1, Y2):
    assert Y1.shape == Y2.shape
    ratio = 0.63
    return Y1 * ratio + Y2 * (1.0 - ratio)

#preds = aggregate_predicts(Y_dev_preds_rnn, Y_dev_preds_ridge)
#print("RMSL error for RNN + Ridge on dev set:", rmsle(Y_dev, Y_dev_preds))


# preds = aggregate_predicts(rnn_preds, ridge_preds)
# submission = pd.DataFrame({
#         "test_id": test_df.test_id,
#         "price": preds.reshape(-1),
# })
# submission.to_csv("./rnn_ridge_submission.csv", index=False)

if __name__ == '__main__':
    rnn_prediction(train_df, dev_df, test_df)
    #ridge_prediction(train_df, dev_df, test_df)