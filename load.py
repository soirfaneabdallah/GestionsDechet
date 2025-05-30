import h5py
import numpy as np

def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) 
    y_train = np.array(train_dataset["Y_train"][:]) 

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) 
    y_test = np.array(test_dataset["Y_test"][:]) 
    
    return X_train, y_train, X_test, y_test

def load_datagans():
    train_dataset = h5py.File('datasets/trainset_gans.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) 
    y_train = np.array(train_dataset["Y_train"][:]) 

    test_dataset = h5py.File('datasets/testset_gans.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) 
    y_test = np.array(test_dataset["Y_test"][:]) 
    
    return X_train, y_train, X_test, y_test
def normalisation(X_train, X_test):
         X_train = X_train / X_train.max()
         X_test = X_test / X_train.max()
         return X_train,X_test