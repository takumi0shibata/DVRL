class PAESConfig:
    DROPOUT = 0.5
    CNN_FILTERS = 100
    CNN_KERNEL_SIZE = 5
    LSTM_UNITS = 100
    EMBEDDING_DIM = 50
    FEATURES_PATH = 'data/hand_crafted_v3.csv'
    READABILITY_PATH = 'data/allreadability.pickle'
    EPOCHS = 50
    BATCH_SIZE = 64