from __future__ import division
from __future__ import print_function
import math
import operator
import time
import random
import copy
import sys
import ast
import csv
import os.path
from collections import Counter


class csvdataStructure():

	def __init__(self, classifier):
		self.items = []
		self.attributes = []
		self.attr_types = []
		self.classifier = classifier
		self.class_index = None


def get_data(dataset, csvfile, datatypes):
	
	file_object = open(csvfile)
	file_content = file_object.read()
	splitted_data = file_content.splitlines()
	dataset.items = [rows.split(',') for rows in splitted_data]
	dataset.attributes = dataset.items.pop(0)
	dataset.attr_types = datatypes.split(',')


def data_preprocess(dataset):

	class_values = [example[dataset.class_index] for example in dataset.items]
	class_mode = Counter(class_values)
	class_mode = class_mode.most_common(1)[0][0]
						 
	for attr_index in range(len(dataset.attributes)):

		class_zero_datasets = filter(lambda x: x[dataset.class_index] == '0', dataset.items)
		class_zero_values = [item[attr_index] for item in class_zero_datasets]  
					
		class_one_datasets = filter(lambda x: x[dataset.class_index] == '1', dataset.items)
		class_one_values = [item[attr_index] for item in class_one_datasets]
				
		values = Counter(class_zero_values)
		value_counts = values.most_common()
		
		mode0 = values.most_common(1)[0][0]

		values = Counter(class_one_values)
		mode1 = values.most_common(1)[0][0]

		mode_01 = [mode0, mode1]

		attr_modes = [0]*len(dataset.attributes)
		attr_modes[attr_index] = mode_01

		for item in dataset.items:
			for x in range(len(dataset.items[0])):
				if dataset.attributes[x] == 'True':
					item[x] = float(item[x])





class decisiontreeNode():
	
	def __init__(self, leafnode, classification, attributeSplitIndexVal, attributeSplittedValue, parent, upperChild, lowerChild, tree_height):
		
	
		self.parent = parent
		self.upperChild = None
		self.lowerChild = None
		self.tree_height = None
		self.leafnode = True
		self.classification = None
		self.attributeSplit = None
		self.attributeSplitIndexVal = None
		self.attributeSplittedValue = None


def gain(dataset, entropy, val, attr_index):

	classifier = dataset.attributes[attr_index]
	attr_entropy = 0
	total_examples = len(dataset.items);
	gain_upper_dataset = csvdataStructure(classifier)
	gain_lower_dataset = csvdataStructure(classifier)
	gain_upper_dataset.attributes = dataset.attributes
	gain_lower_dataset.attributes = dataset.attributes
	gain_upper_dataset.attr_types = dataset.attr_types
	gain_lower_dataset.attr_types = dataset.attr_types

	for example in dataset.items:
		if (example[attr_index] >= val):
			gain_upper_dataset.items.append(example)
		elif (example[attr_index] < val):
			gain_lower_dataset.items.append(example)

	if (len(gain_upper_dataset.items) == 0 or len(gain_lower_dataset.items) == 0):
		return -1

	attr_entropy += get_entropy(gain_upper_dataset, classifier)*len(gain_upper_dataset.items)/total_examples
	attr_entropy += get_entropy(gain_lower_dataset, classifier)*len(gain_lower_dataset.items)/total_examples

	return entropy - attr_entropy


def get_entropy(dataset, classifier):

	ones = one_count(dataset.items, dataset.attributes, classifier)
	total_examples = len(dataset.items);
	zeros = total_examples - ones

	entropy = 0
	p = ones / total_examples
	if (p != 0):
		entropy += p * math.log(p, 2)
	p = zeros/total_examples
	if (p != 0):
		entropy += p * math.log(p, 2)

	entropy = -entropy

	return entropy


def variance_impurity(dataset, classifier):

	ones = one_count(dataset.items, dataset.attributes, classifier)
	total_examples = len(dataset.items);
	zeros = total_examples - ones

	entropy = 0
	entropy += (zeros/total_examples) * (ones/total_examples)

	return entropy


def variance_impurity_calc_gain(dataset, entropy, val, attr_index):
	
	classifier = dataset.attributes[attr_index]
	attr_entropy = 0
	total_examples = len(dataset.items);
	gain_upper_dataset = csvdataStructure(classifier)
	gain_lower_dataset = csvdataStructure(classifier)
	gain_upper_dataset.attributes = dataset.attributes
	gain_lower_dataset.attributes = dataset.attributes
	gain_upper_dataset.attr_types = dataset.attr_types
	gain_lower_dataset.attr_types = dataset.attr_types
	for example in dataset.items:
		if (example[attr_index] >= val):
			gain_upper_dataset.items.append(example)
		elif (example[attr_index] < val):
			gain_lower_dataset.items.append(example)

	if (len(gain_upper_dataset.items) == 0 or len(gain_lower_dataset.items) == 0):
		return -1

	attr_entropy += variance_impurity(gain_upper_dataset, classifier)*len(gain_upper_dataset.items)/total_examples
	attr_entropy += variance_impurity(gain_lower_dataset, classifier)*len(gain_lower_dataset.items)/total_examples

	return entropy - attr_entropy
	

def one_count(instances, attributes, classifier):

	count = 0
	class_index = None

	for a in range(len(attributes)):
		if attributes[a] == classifier:
			class_index = a
		else:
			class_index = len(attributes) - 1
	for i in instances:
		if i[class_index] == "1":
			count += 1
	return count


def leaf_classifier(dataset, classifier):
	ones = one_count(dataset.items, dataset.attributes, classifier)
	total = len(dataset.items)
	zeroes = total - ones
	if (ones >= zeroes):
		return 1
	else:

		return 0



def information_gain_compute_tree(dataset, parent_node, classifier, attribute_list):
	

	node = decisiontreeNode(True, None, None, None, parent_node, None, None, 0)
	
	if (parent_node == None):
		node.tree_height = 0
	else:
		node.tree_height = node.parent.tree_height + 1

	ones = one_count(dataset.items, dataset.attributes, classifier)
	if (len(dataset.items) == ones):
		node.classification = 1
		node.leafnode = True
		return node
	
	elif (ones == 0):
		node.classification = 0
		node.leafnode = True
		return node
	
	else:
		node.leafnode = False
	
	attr_to_split = None
	max_gain = 0
	split_val = None 
	min_gain = 0.01
	dataset_entropy = get_entropy(dataset, classifier)
	
	for attr_index in attribute_list:	

		if (dataset.attributes[attr_index] != classifier):
			local_max_gain = 0
			local_split_val = None
			attr_value_list = [example[attr_index] for example in dataset.items]
			attr_value_list = list(set(attr_value_list))
            
			for val in attr_value_list:
				local_gain = gain(dataset, dataset_entropy, val, attr_index)
  
				if (local_gain > local_max_gain):
					local_max_gain = local_gain
					local_split_val = val

			if (local_max_gain > max_gain):
				max_gain = local_max_gain
				split_val = local_split_val
				attr_to_split = attr_index

	if (split_val is None or attr_to_split is None):
		node.leafnode = True
		node.classification = leaf_classifier(dataset, classifier)
		return node
	elif (max_gain <= min_gain or node.tree_height > 10):
		node.leafnode = True
		node.classification = leaf_classifier(dataset, classifier)
		return node

	node.attributeSplitIndexVal = attr_to_split
	node.attributeSplit = dataset.attributes[attr_to_split]
	node.attributeSplittedValue = split_val
	
	upper_dataset = csvdataStructure(classifier)
	lower_dataset = csvdataStructure(classifier)

	upper_dataset.attributes = dataset.attributes
	lower_dataset.attributes = dataset.attributes
	upper_dataset.attr_types = dataset.attr_types
	lower_dataset.attr_types = dataset.attr_types

	for example in dataset.items:
		if (attr_to_split is not None and example[attr_to_split] >= split_val):
			upper_dataset.items.append(example)
		elif (attr_to_split is not None):
			lower_dataset.items.append(example)

	
	if(attr_to_split is not None):
		attribute_split_index=dataset.attributes.index(str(node.attributeSplit))
		attribute_list = [item for item in attribute_list if item != attribute_split_index]
    

	node.upperChild = information_gain_compute_tree(upper_dataset, node, classifier, attribute_list)
	node.lowerChild = information_gain_compute_tree(lower_dataset, node, classifier, attribute_list)

	return node




def variance_impurity_compute_tree(dataset, parent_node, classifier, attribute_list):
	
	node = decisiontreeNode(True, None, None, None, parent_node, None, None, 0)
	
	if (parent_node == None):
		node.tree_height = 0
	else:
		node.tree_height = node.parent.tree_height + 1

	ones = one_count(dataset.items, dataset.attributes, classifier)
	
	if (len(dataset.items) == ones):
		node.classification = 1
		node.leafnode = True
		return node
	
	elif (ones == 0):
		node.classification = 0
		node.leafnode = True
		return node
	
	else:
		node.leafnode = False
	
	attr_to_split = None
	max_gain = 0
	split_val = None 
	min_gain = 0.001
	dataset_entropy = variance_impurity(dataset, classifier)
	
	
	for attr_index in attribute_list:

		if (dataset.attributes[attr_index] != classifier):
			local_max_gain = 0
			local_split_val = None
			attr_value_list = [example[attr_index] for example in dataset.items]
			attr_value_list = list(set(attr_value_list))

			for val in attr_value_list:
				local_gain = variance_impurity_calc_gain(dataset, dataset_entropy, val, attr_index)
  
				if (local_gain > local_max_gain):
					local_max_gain = local_gain
					local_split_val = val

			if (local_max_gain > max_gain):
				max_gain = local_max_gain
				split_val = local_split_val
				attr_to_split = attr_index

	if (split_val is None or attr_to_split is None):
		node.leafnode = True
		node.classification = leaf_classifier(dataset, classifier)
		return node		

	elif (max_gain <= min_gain or node.tree_height > 10):

		node.leafnode = True
		node.classification = leaf_classifier(dataset, classifier)
		return node


	node.attributeSplitIndexVal = attr_to_split
	node.attributeSplit = dataset.attributes[attr_to_split]
	node.attributeSplittedValue = split_val
	
	upper_dataset = csvdataStructure(classifier)
	lower_dataset = csvdataStructure(classifier)

	upper_dataset.attributes = dataset.attributes
	lower_dataset.attributes = dataset.attributes
	upper_dataset.attr_types = dataset.attr_types
	lower_dataset.attr_types = dataset.attr_types

	for example in dataset.items:
		if (attr_to_split is not None and example[attr_to_split] >= split_val):
			upper_dataset.items.append(example)
		elif (attr_to_split is not None):
			lower_dataset.items.append(example)


	if(attr_to_split is not None):
		attribute_split_index=dataset.attributes.index(str(node.attributeSplit))
		attribute_list = [item for item in attribute_list if item != attribute_split_index]		

	node.upperChild = variance_impurity_compute_tree(upper_dataset, node, classifier, attribute_list)
	node.lowerChild = variance_impurity_compute_tree(lower_dataset, node, classifier, attribute_list)

	return node




def validate_tree(node, dataset):
	total = len(dataset.items)
	correct = 0
	for example in dataset.items:
		correct += validate_datarow(node, example)
	return correct/total




def validate_datarow(node, example):
	if (node.leafnode == True):
		projected = node.classification
		actual = int(example[-1])
		if (projected == actual): 
			return 1
		else:
			return 0
	value = example[node.attributeSplitIndexVal]
	if (value >= node.attributeSplittedValue):
		return validate_datarow(node.upperChild, example)
	else:
		return validate_datarow(node.lowerChild, example)




def test_datarow(example, node, class_index):
	if (node.leafnode == True):
		return node.classification
	else:
		if (example[node.attributeSplitIndexVal] >= node.attributeSplittedValue):
			return test_datarow(example, node.upperChild, class_index)
		else:
			return test_datarow(example, node.lowerChild, class_index)




def prune_tree(root, node, dataset, best_score):
	
	if (node.leafnode == True):

		classification = node.classification

		if node.parent is not None:
			node.parent.leafnode = True
			node.parent.classification = node.classification

		if (node.tree_height < 10):
			new_score = validate_tree(root, dataset)
		else:
			new_score = 0
  

		if (new_score >= best_score):
			return new_score
		else:
			node.parent.leafnode = False
			node.parent.classification = None
			return best_score

	else:

		new_score = prune_tree(root, node.upperChild, dataset, best_score)

		if (node.leafnode == True):
			return new_score

		new_score = prune_tree(root, node.lowerChild, dataset, new_score)

		if (node.leafnode == True):
			return new_score

		return new_score



def print_tree(node, dataset):
	
	if(node is None):
		return

	if(node.leafnode == True):
		print(node.classification)
		return
			
	else:
		print(str(dataset.attributes[node.attributeSplitIndexVal]) + " = " + str(node.attributeSplittedValue) + " : ")
		print_tree(node.upperChild, dataset)
		print(str(dataset.attributes[node.attributeSplitIndexVal]) + " = 0 : ")
		print_tree(node.lowerChild, dataset)
		return





if __name__ == "__main__":

	args = str(sys.argv)
	args = ast.literal_eval(args)

	if (len(args) < 6):
		print( "You have input less than the minimum number of arguments.")
		print("Usage: .\program <training-set> <validation-set> <test-set> <to-print> to-print:{yes,no} <prune> prune:{yes, no}")

	elif (args[1][-4:] != ".csv" and args[2][-4:] != ".csv" and args[3][-4:] != ".csv"):
		print( "Your training, validation and test files must be a .csv!")

	else:
		training_data = args[1]
		validation_data = args[2]
		test_data = args[3]
		to_print = args[4].split(":")
		to_prune = args[5].split(":")

		dataset = csvdataStructure("")
		datatypes = "true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false"
		get_data(dataset, training_data, datatypes)
		classifier = dataset.attributes[-1]
		dataset.classifier = classifier
		for a in range(len(dataset.attributes)):
			if dataset.attributes[a] == dataset.classifier:
				dataset.class_index = a
			else:
				dataset.class_index = range(len(dataset.attributes))[-1]
				
		unprocessed = copy.deepcopy(dataset)
		data_preprocess(dataset)

		attribute_list=dataset.attributes
		attribute_index_list=[]
		
		for index,element in enumerate(attribute_list):
			attribute_index_list.append(index)
		
		root = information_gain_compute_tree(dataset, None, classifier,attribute_index_list) 
		vi_root = variance_impurity_compute_tree(dataset, None, classifier,attribute_index_list)

		if(str(to_print[1]) == "yes"):
			print("Tree generated using information gain:")
			print("\n")
			print_tree(root, dataset)
			print( "\n")
			print("Tree generated using variance impurity:")
			print("\n")
			print_tree(vi_root, dataset)
			print("\n")

		train_best_score = validate_tree(root, dataset)
		#print( "Initial (pre-pruning) training set score using information gain heuristic: " + str(100*train_best_score) +"%")

		vi_train_best_score = validate_tree(vi_root, dataset)
		vi_all_train_ex_score = copy.deepcopy(vi_train_best_score)
		#print( "Initial (pre-pruning) training set score using variance impurity heuristic: " + str(100*vi_train_best_score) +"%")

		if(str(to_prune[1]) == "yes"):
			train_post_prune_accuracy = prune_tree(root, root, dataset, train_best_score)
			#print( "Post-pruning score on training set using information gain heuristic: " + str(100*train_post_prune_accuracy) + "%")

			vi_train_post_prune_accuracy = prune_tree(vi_root, vi_root, dataset, vi_train_best_score)
			#print( "Post-pruning score on training set using variance impurity heuristic: " + str(100*vi_train_post_prune_accuracy) + "%")

		validateset = csvdataStructure(classifier)
		get_data(validateset, validation_data, datatypes)
		for a in range(len(dataset.attributes)):
			if validateset.attributes[a] == validateset.classifier:
				validateset.class_index = a
			else:
				validateset.class_index = range(len(validateset.attributes))[-1]
		data_preprocess(validateset)
		best_score = validate_tree(root, validateset)
		#print( "Initial (pre-pruning) validation set score using information gain heuristic: " + str(100*best_score) +"%")

		vi_best_score = validate_tree(vi_root, validateset)
		#print( "Initial (pre-pruning) validation set score using variance impurity heuristic: " + str(100*vi_best_score) +"%")

		if(str(to_prune[1]) == "yes"):
			post_prune_accuracy = prune_tree(root, root, validateset, best_score)
			#print( "Post-pruning score on validation set using information gain heuristic: " + str(100*post_prune_accuracy) + "%")

			vi_post_prune_accuracy = prune_tree(vi_root, vi_root, validateset, vi_best_score)
			#print( "Post-pruning score on validation set using variance impurity heuristic: " + str(100*vi_post_prune_accuracy) + "%")

		testset = csvdataStructure(classifier)
		get_data(testset, test_data, datatypes)
		for a in range(len(dataset.attributes)):
			if testset.attributes[a] == testset.classifier:
				testset.class_index = a
			else:
				testset.class_index = range(len(testset.attributes))[-1]

		for example in testset.items:
			example[testset.class_index] = '0'
		testset.items[0][testset.class_index] = '1'
		testset.items[1][testset.class_index] = '1'
		data_preprocess(testset)

		test_best_score = validate_tree(root, testset)
		#print( "Initial (pre-pruning) test set score using information gain heuristic: " + str(100*test_best_score) +"%")

		vi_test_best_score = validate_tree(vi_root, testset)
		#print( "Initial (pre-pruning) test set score using variance impurity heuristic: " + str(100*vi_test_best_score) +"%")

		if(str(to_prune[1]) == "yes"):
			test_post_prune_accuracy = prune_tree(root, root, testset, test_best_score)
			#print( "Post-pruning score on validation set using information gain heuristic: " + str(100*test_post_prune_accuracy) + "%")

			vi_test_post_prune_accuracy = prune_tree(vi_root, vi_root, testset, vi_test_best_score)
			#print( "Post-pruning score on validation set using variance impurity heuristic: " + str(100*vi_test_post_prune_accuracy) + "%")

		dataset_name = training_data.split("/")


		fname = "results.txt"

		ret_val = os.path.isfile(fname)

		if(ret_val == True):
			text_file = open(fname, "a")
		else:
			text_file = open(fname, "w")

		print(dataset_name[-2], file=text_file)

		print("H1 NP Training {}".format(train_best_score), file=text_file)
		print("H1 NP Validation {}".format(best_score), file=text_file)
		print("H1 NP Test {}".format(test_best_score), file=text_file)

		if(str(to_prune[1]) == "yes") :
			print("H1 P Training {}".format(train_post_prune_accuracy), file=text_file)
			print("H1 P Validation {}".format(post_prune_accuracy), file=text_file)
			print("H1 P Test {}".format(test_post_prune_accuracy), file=text_file)

		print("H2 NP Training {}".format(vi_train_best_score), file=text_file)
		print("H2 NP Validation {}".format(vi_best_score), file=text_file)
		print("H2 NP Test {}".format(vi_test_best_score), file=text_file)

		if(str(to_prune[1]) == "yes") :
			print("H2 P Training {}".format(vi_train_post_prune_accuracy), file=text_file)
			print("H2 P Validation {}".format(vi_post_prune_accuracy), file=text_file)
			print("H2 P Test {}".format(vi_test_post_prune_accuracy), file=text_file)

		text_file.close()

		print( "Task complete. Results outputted to results.txt")


