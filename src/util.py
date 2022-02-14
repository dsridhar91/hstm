import numpy as np

def split_train_test_indices(num_docs, prop_to_split=0.5, seed=42):
	np.random.seed(seed)
	indices = np.arange(num_docs)
	np.random.shuffle(indices)
	num_sample = int(num_docs * prop_to_split)
	train_indices = indices[:num_sample]
	test_indices = indices[num_sample:]
	return train_indices, test_indices

def cross_val_splits(num_docs, num_splits=10, seed=42):
	np.random.seed(seed)
	indices = np.arange(num_docs)
	np.random.shuffle(indices)

	split_size = (num_docs // num_splits)
	split_indices = [indices[i*split_size:(i+1)*split_size] for i in range(num_splits-1)]
	split_indices.append(indices[(num_splits-1)*split_size:])
	return split_indices

def get_cv_split_assignments(num_docs, num_splits=10, seed=42):
	np.random.seed(seed)
	indices = np.arange(num_docs)
	np.random.shuffle(indices)

	split_size = (num_docs // num_splits)
	split_indices = [indices[i*split_size:(i+1)*split_size] for i in range(num_splits-1)]
	split_indices.append(indices[(num_splits-1)*split_size:])

	split_assignment = np.zeros(num_docs)
	for (s_idx, inds) in enumerate(split_indices):
		split_assignment[inds]=s_idx

	return split_assignment
