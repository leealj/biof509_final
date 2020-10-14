import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


class sctype:
    def __init__(self, csv, test_size, random_state):
        df = pd.read_csv(csv)
        train, test = train_test_split(
            df, test_size=test_size, random_state=random_state)
        self.train_labels = train.iloc[]
        self.train_data = train.iloc[]
        self.test_labels = test.iloc[]
        self.test_data = test.iloc[]
