import numpy as np
import pandas as pd
import scipy.io as io
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# path = path to the 10Xv2 data directory containing genes.tsv, barcodes.tsv, and matrix.mtx files
# Note: need to import cell type annotations file manually since annotation files do not
# have any set format between datasets. Set index names to cell barcode IDs in the annotation dataframe
# and have the celltype annotations as the first column.


class sctype:
    def __init__(self, path):
        self.genes = pd.read_csv(
            path + 'genes.tsv', sep='\t', header=None).iloc[:, 1]
        self.barcodes = pd.read_csv(
            path + 'barcodes.tsv', sep='\t', header=None).iloc[:, 0]
        expression = io.mmread(path + 'matrix.mtx')

        data = pd.DataFrame.sparse.from_spmatrix(
            data=expression, index=self.genes, columns=self.barcodes)
        data = data.fillna(0)
        gene_counts = data.sum(axis=1)
        data = data[gene_counts != 0]

        self.data = data.transpose()
        self.dropouts = gene_counts[gene_counts == 0].index


# This preprocess step will divide the gene counts by the total counts per cell, multiply by scale_factor
# and perform log1p; output is normalized logCPM data.
# Scale will center the per gene expression data around 0 and adjust to unit variance.
# Though, this doesn't seem to affect NN performance too much. Can feed self.data to the split method
# and go directly to training the NN without running this preprocessing step.

    def data_preprocess(self, normalize=True, scale_factor=10000, scale=True):
        if normalize:
            cell_counts = self.data.sum(axis=1)
            div = self.data.divide(other=cell_counts, axis='index')
            self.data_norm = np.log1p(div * scale_factor)
        if scale:
            if normalize:
                self.data_norm = (self.data_norm - np.mean(self.data_norm, axis=0))\
                    / (np.std(self.data_norm, axis=0))
            else:
                self.data_scaled = (self.data - np.mean(self.data, axis=0))\
                    / (np.std(self.data, axis=0))


# Labels input should be a dataframe with celltype annotations in the first column.
# This method will convert labels with one-hot encoding for use as labels in the NN.
# Must be done BEFORE splitting into train/test sets.

# The key to the one-hot encoded labels will be stored in self.key
# The original labels will be stored in self.labels

    def process_labels(self, labels, depth):
        all_labels = labels.copy()

        if depth == 1:
            all_labels[all_labels == ['CD4+/CD45RO+ Memory']] = 'CD4+ T'
            all_labels[all_labels == ['CD4+/CD25 T Reg']] = 'CD4+ T'
            all_labels[all_labels == ['CD4+ T Helper2']] = 'CD4+ T'
            all_labels[all_labels == ['CD4+/CD45RA+/CD25- Naive T']] = 'CD4+ T'
            all_labels[all_labels == ['CD8+ Cytotoxic T']] = 'CD8+ T'
            all_labels[all_labels == [
                'CD8+/CD45RA+ Naive Cytotoxic']] = 'CD8+ T'

            factorized = pd.factorize(all_labels.iloc[:, 0])
            self.labels = all_labels.iloc[:, 0]
            self.key = factorized[1]
            self.categorical_labels = to_categorical(
                factorized[0], len(factorized[1]))

        if depth == 2:
            factorized = pd.factorize(all_labels.iloc[:, 0])
            self.labels = all_labels.iloc[:, 0]
            self.key = factorized[1]
            self.categorical_labels = to_categorical(
                factorized[0], len(factorized[1]))


# NOTE: for SVM, it is better to use data that is NOT scaled; it will take too long to train.
# Only use self.data_norm to split if scale = False when calling self.data_preprocess().
# Otherwise, use data = self.data

    def split(self, data, labels, test_size, random_state):
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            data, labels, test_size=test_size,
            random_state=random_state, stratify=labels)

    def svm(self, iterations=1000, cv=4, method='sigmoid'):
        train_labels = self.to_vector(self.train_labels)

        clf = LinearSVC(max_iter=iterations)
        clf = CalibratedClassifierCV(clf, cv=cv, method=method)
        self.svmfit = clf.fit(self.train_data, train_labels)
        return self.svmfit

    def evaluate_svm(self, predictions, target):
        true_labels = self.to_vector(target)
        acc = sum(predictions == true_labels)/len(predictions)
        return acc

    def ann(self, epochs, batch_size):
        model = keras.Sequential()
        model.add(layers.Dense(200, activation='relu', name='layer1'))
        model.add(layers.Dense(100, activation='relu', name='layer2'))
        model.add(layers.Dense(len(self.key),
                               activation='softmax', name='outputlayer'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(self.train_data, self.train_labels,
                  epochs=epochs, batch_size=batch_size)
        self.ann_fit = model
        return self.ann_fit


# From the output of NN, select highest probabilities per cell as the celltype prediction

    def max_prob(self, probs):
        predictions = (probs == probs.max(axis=1)[:, None]).astype(int)
        return predictions


# Converts the binary class matrix (output from NN) back to a class vector with
# integeters representing the different classes. (essentially undoes to_categorical)
# The order of the labels are stored in self.key

    def to_vector(self, class_matrix):
        copy_matrix = class_matrix.copy()
        for i in range(len(copy_matrix[0])):
            copy_matrix[:, i] = copy_matrix[:, i] * (i+1)
        tmp = []
        for j in range(len(copy_matrix)):
            tmp.append(np.trim_zeros(copy_matrix[j]))
        class_vector = np.concatenate(tmp)
        return class_vector
