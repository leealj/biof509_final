import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


class sctype:
    def __init__(self, csv, test_size, random_state):
        self.df = pd.read_csv(csv)
        # this data-importing step will be updated once I figure out the best way to
        # accomplish it.

    def split(self, test_size, random_state):
        train, test = train_test_split(
            self.df, test_size=test_size, random_state=random_state)
        self.train_labels = train.iloc[]
        self.train_data = train.iloc[]
        self.test_labels = test.iloc[]
        self.test_data = test.iloc[]
        # the indices will also be updated once I know exactly what the data will
        # look like once it has been properly imported
