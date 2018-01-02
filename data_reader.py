import numpy as np

class DataReader():
	def __init__(self, filename, batch_size = None):
		npz = np.load(filename)
		
		#set inputs and targets
		self.inputs, self.targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
		
		if batch_size is None:
			self.batch_size = self.inputs.shape[0]
		else:
			self.batch_size = batch_size
		
		self.curr_batch = 0
		self.batch_count = self.inputs.shape[0]//self.batch_size
	
	#create iterator for batching
	def __next__(self):
		#stop iteration if all batches have been used
		if self.curr_batch >= self.batch_count:
			self.curr_batch = 0
			raise StopIteration()
		
		#slicing dataset in batches to be load one after another by "next" function
		batch_slice  = slice(self.curr_batch * self.batch_size, (self.curr_batch + 1)* self.batch_size)
		
		inputs_batch = self.inputs[batch_slice]
		targets_batch = self.targets[batch_slice]
		self.curr_batch += 1
		
		#apply one-hot encode to targets, 0 --> 0
		classes_num = 2 #ADJUST FOR OTHER CLASSIFICATION PROBLEM
		targets_one_hot = np.zeros((targets_batch.shape[0], classes_num)) #create array of zeros [0,0,...,num of class]
		targets_one_hot[range(targets_batch.shape[0]), targets_batch] = 1 #using indexing, set the element 0 to [1,0], 1 to [0,1]
		
		return inputs_batch, targets_one_hot
	
	#tells python that we're defining an iterable
	def __iter__(self):
		return self
		
		