# scRNA-seq cell type predictions with SVM and ANN
Applied Machine Learning Final Project: 

The sctype() class gives you the option of training either an SVM or an ANN to make cell type predictions
with single cell RNA-seq data. This tool can theoretically work with any data generated from the 10X v2
pipeline that produces a directory with 3 files: genes.tsv, barcodes.tsv, and matrix.mtx.

The Zheng 2017 Fresh 68K PBMC dataset with annotations was used in the Jupyter Notebook demo
which can be downloaded from:
https://support.10xgenomics.com/single-cell-gene-expression/datasets

### The sctype object
Initializing an sctype object will require a path to the directory containing these 3 files and will 
import the data and store the raw counts in the .data attribute. 

NOTE: the annotation file needs to be loaded in manually with cell types placed in the first column.


## __Preprocessing workflow__:
1. Define path to annotation and data files (in format of 10X v2 data with genes.tsv, barcodes.tsv, and matrix.mtx files)
2. import and process annotation file to have barcodes as indices and cell type labels in first column
3. Create sctype object by calling __sctype(path)__ with path = path to the directory with data files.
4. Process the labels with the __.process_labels()__ method. This will set several attributes:
  - The .labels attribute will store the original cell type labels.
  - The .categorical_labels attribute will store the cell type labels (per cell) in the one-hot encoded format.
  - The .key attribute will store the unique cell types in order. 
5. Preprocess counts data with the __.data_preprocess()__ method. Setting normalize = True will take the raw counts from self.data
and divide each feature from each sample by the sum of all counts for that sample and calculate log1p after multiplying
by scale_factor (default = 10,000). This will result in the normalized logCPM expression stored in the .data_norm attribute.
Setting scale = True will center the logCPM expression data around 0. NOTE: This is not recommended if using SVM; it will
dramatically increase training time.
6. Split the data with the __.split()__ method to create separate train/test datasets with respective labels. Must explicitly
indicate which data to split and which labels to  use. If using one-hot encoded labels (needed for neural network), set 
labels = self.categorical_labels. If using normalized data (from .data_preprocess), set data = self.data_norm.
The test_size parameter will determine the size of the validation set and random_state can be set to produce consistent results
across different function calls. Refer to sklearn documentation for train_test_split() for more information on these two parameters.

## Training/testing the Artificial Neural Network
1. Train the NN with # of epochs and batch size provided as input to the self.ann() method.
2. Evaluate performance of NN
3. Generate cell type predictions with self.ann_fit.predict()
4. Convert prediction probabilities to cell type labels with self.max_prob followed by self.to_vector. This will provide a 
class vector with each integeter representing a cell type (actual cell type names stored in self.key)

## Train/testing the SVM
1. Train the SVM with cv set to number of folds for cross validation used to calibrate the probabilities returned from model.predict().
2. Generate cell type predictions with model.predict()
3. Evaluate performance of SVM with self.evaluate_svm() using the output from model.predict() and the true labels 
(from self.train_labels or self_test_labels) as the input.
4. Convert prediction probabilities to cell type labels with self.max_prob() followed by self.to_vector(). This will provide a class 
vector with each integeter representing a cell type (actual cell type names stored in self.key)

## Overview of __methods__ included in the sctype class:
- `__init__(path)`: will import 10X v2 data given a path to a directory with genes.tsv, barcodes.tsv, and matrix.mtx files. Automatically 
filters dropout genes (genes with 0 counts across all cells), and stores names of drop out genes in self.dropouts. Raw counts data
are stored in self.data
- `.data_preprocess(normalize = True, scale_factor = 10000, scale = True)`: Setting normalize = True will take the raw counts from self.data
and divide each feature from each sample by the sum of all counts for that sample and calculate log1p after multiplying
by scale_factor (default = 10,000). This will result in the normalized logCPM expression stored in the .data_norm attribute.
Setting scale = True will center the logCPM expression data around 0. __NOTE: Setting scale = True is not recommended if using SVM; it will
dramatically increase training time.__
- `.process_labels(labels, depth)`: given the name of the manually loaded annotations file with cell type labels in the first column. 
The 'depth' parameter = 1 or 2, where depth = 1 will all of the different CD4/CD8 subtypes to either CD4+ T or CD8+ T, while depth = 2
will retain all of the original cell type labels. 
This method will set several attributes:
   - The .labels attribute will store the original cell type labels.
   - The .categorical_labels attribute will store the cell type labels (per cell) in the one-hot encoded format.
   - The .key attribute will store the unique cell types in order. 
- `.split(data, labels, test_size, random_state)`: creates separate train/test datasets with labels. Must explicitly
indicate which data to split and which labels to use. If using one-hot encoded labels (needed for neural network), set 
labels = self.categorical_labels. If using normalized data (from .data_preprocess), set data = self.data_norm.
The test_size parameter will determine the size of the validation set and random_state can be set to produce consistent results
across different function calls. Refer to sklearn documentation for train_test_split() for more information on these two parameters.
- `.svm(iterations = 1000, cv = 4, method = 'sigmoid')`: Fits a linear SVM with data and labels stored in the .train_data and .train_labels
attributes that should have been set after calling the .split() method. The cv parameter refers to the number of folds for the 
CalibratedClassifierCV() wrapper to calibrate the probabilities returned from the predictions, while the method parameter specifies the 
method to calibrate the probabilities. 
- `.evaluate_svm(predictions, target)`: Calculates accuracy of the predictions generated from SVM. The target parameter should be the
original cell type labels stored in self.test_labels or self.train_labels.
- `.ann(epochs, batch_size)`: Trains the artificial neural network given the number of training epochs and batch size. Will use the data and 
labels stored in the self.train_data and self.train_labels attributes.
- `.max_prob(probs)`: Since the ANN will provide prediction probabilities across all classes, this method will take the probability matrix
and choose the highest probabilities per cell as the predicted cell type label.
- `.to_vector(class_matrix)`: This method, given a binary class matrix (such as the one generated from .process_labels() and stored in
self.categorical_labels) and convert it to a class vector, effectively reversing the keras.util method to_categorical(). 
