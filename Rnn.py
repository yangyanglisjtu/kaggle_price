import numpy as np
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
import pandas as pd

MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75

def rmsle(Y, Y_pred):
    # Y and Y_red have already been in log scale.
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))

def get_keras_data(df):
    X = {
        'name': pad_sequences(df.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(df.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(df.brand_name),
        'category_name': np.array(df.category_name),
        'item_condition': np.array(df.item_condition_id),
        'num_vars': np.array(df[["shipping"]]),
    }
    return X


def text2seq(full_df, n_trains, n_devs):
    # Calculate number of train/dev/test examples.
    #n_tests = test_df.shape[0]

    print("Processing categorical data...")
    le = LabelEncoder()

    le.fit(full_df.category_name)
    full_df.category_name = le.transform(full_df.category_name)

    le.fit(full_df.brand_name)
    full_df.brand_name = le.transform(full_df.brand_name)

    del le

    print("Transforming text data to sequences...")
    raw_text = np.hstack([full_df.item_description.str.lower(), full_df.name.str.lower()])

    print("   Fitting tokenizer...")
    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_text)

    print("   Transforming text to sequences...")
    full_df['seq_item_description'] = tok_raw.texts_to_sequences(full_df.item_description.str.lower())
    full_df['seq_name'] = tok_raw.texts_to_sequences(full_df.name.str.lower())

    del tok_raw

    train = full_df[:n_trains]
    dev = full_df[n_trains:n_trains + n_devs]
    test = full_df[n_trains + n_devs:]

    X_train = get_keras_data(train)
    X_dev = get_keras_data(dev)
    X_test = get_keras_data(test)

    return X_train, X_dev, X_test

def new_rnn_model(X_train, max_dic, lr=0.001, decay=0.0):
    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_name = Embedding(max_dic['MAX_TEXT'], 20)(name)
    emb_item_desc = Embedding(max_dic['MAX_TEXT'], 60)(item_desc)
    emb_brand_name = Embedding(max_dic['MAX_BRAND'], 10)(brand_name)
    emb_category_name = Embedding(max_dic['MAX_CATEGORY'], 10)(category_name)
    emb_item_condition = Embedding(max_dic['MAX_CONDITION'], 5)(item_condition)

    # rnn layers
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_name)

    # main layers
    main_l = concatenate([
        Flatten() (emb_brand_name)
        , Flatten() (emb_category_name)
        , Flatten() (emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , num_vars
    ])

    main_l = Dense(256)(main_l)
    main_l = Activation('elu')(main_l)
    #main_l = Dropout(0.3)(main_l)

    main_l = Dense(128)(main_l)
    main_l = Activation('elu')(main_l)
    #main_l = Dropout(0.3)(main_l)

    main_l = Dense(64)(main_l)
    main_l = Activation('elu')(main_l)
    #main_l = Dropout(0.3)(main_l)

    # the output layer.
    output = Dense(1, activation="linear") (main_l)

    model = Model([name, item_desc, brand_name , category_name, item_condition, num_vars], output)

    optimizer = Adam(lr=lr, decay=decay)
    model.compile(loss="mse", optimizer=optimizer)

    return model

#model = new_rnn_model()
#model.summary()
#del model

def rnn_prediction(train_df, dev_df, test_df):
    # Set hyper parameters for the model.
    n_trains = train_df.shape[0]
    n_devs = dev_df.shape[0]
    # n_tests = test_df.shape[0]
    BATCH_SIZE = 1024
    epochs = 2
    # Calculate learning rate decay.
    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    steps = int(n_trains / BATCH_SIZE) * epochs
    #lr_init, lr_fin = 0.007, 0.0005
    #lr_init, lr_fin = 0.01, 0.001
    #lr_init, lr_fin = 0.1, 0.01
    full_df = pd.concat([train_df, dev_df, test_df])
    X_train, X_dev, X_test = text2seq(full_df, n_trains, n_devs)
    max_dic = {}
    max_dic['MAX_TEXT'] = np.max([
        np.max(full_df.seq_name.max()),
        np.max(full_df.seq_item_description.max()),
    ]) + 4
    max_dic['MAX_CATEGORY'] = np.max(full_df.category_name.max()) + 1
    max_dic['MAX_BRAND'] = np.max(full_df.brand_name.max()) + 1
    max_dic['MAX_CONDITION'] = np.max(full_df.item_condition_id.max()) + 1

    for lr_init in np.arange(0.016, 0.1, 0.005):
        lr_fin = lr_init / 10
        lr_decay = exp_decay(lr_init, lr_fin, steps)
        rnn_model = new_rnn_model(X_train, max_dic, lr=lr_init, decay=lr_decay)

        Y_train = train_df.target.values.reshape(-1, 1)
        Y_dev = dev_df.target.values.reshape(-1, 1)
        print("Fitting RNN model to training examples...")
        rnn_model.fit(
                X_train, Y_train, epochs=epochs, batch_size=BATCH_SIZE,
                validation_data=(X_dev, Y_dev), verbose=2,
        )

        print("Evaluating the model on validation data...")
        Y_dev_preds_rnn = rnn_model.predict(X_dev, batch_size=BATCH_SIZE)
        with open('result.txt', 'a') as file:
            file.write(" RMSLE error:" + str(rmsle(Y_dev, Y_dev_preds_rnn)) + ' lr_init:' + str(lr_init) + '\n')
    #print(" RMSLE error:", rmsle(Y_dev, Y_dev_preds_rnn))

        #rnn_preds = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
        #rnn_preds = np.expm1(rnn_preds)
    #return rnn_preds
    return None