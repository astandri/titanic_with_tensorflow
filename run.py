from dataset_preparation import DatasetPreparation
from data_reader import DataReader
from model import Model

import numpy as np

def main():
	#Prepare dataset from csv to npz files
	#DatasetPreparation.prepare('train_preprocessed.csv','test_preprocessed.csv')
	
	#Read the dataset, create batches, and one hot encode the targets
	batch_size = 100
	train_data = DataReader('train.npz',batch_size)
	validation_data = DataReader('validation.npz')
	
	test_data = np.load('test.npz')

	m = Model(train_data,validation_data)
	m.train()
	
	m.test(test_data)	
	
  
if __name__== "__main__":
	main()
