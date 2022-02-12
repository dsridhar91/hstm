from model import adjusted_hstm, topic_model, supervised_lda as slda
from model.model_trainer import ModelTrainer
from evaluation.evaluator import Evaluator
# from evaluation.eval_topics import get_perplexity, get_normalized_pmi, get_topics_from_model, get_supervised_topics_from_model, shuffle_topics
import util
import argparse
import sys
import os
import numpy as np
from data.dataset import TextResponseDataset
import torch
from torch.utils.data import DataLoader
from absl import flags
from absl import app

def main(argv):
	
	proc_file = FLAGS.procfile
	pretraining_file = FLAGS.pretraining_file

	base_dataset = FLAGS.data
	if FLAGS.data == 'framing_corpus':
		base_dataset = FLAGS.data + '_' + FLAGS.framing_topic

	base_pretraining = '_pretraining.npz'
	if FLAGS.pretrained_prodlda:
		base_pretraining = '_prodlda_pretraining.npz'

	if pretraining_file == "":
		pretraining_file = '../dat/proc/' + base_dataset + base_pretraining
	if proc_file == "":
		proc_file = '../dat/proc/' + base_dataset + '_proc.npz'

	num_topics = FLAGS.num_topics
	if FLAGS.data == 'amazon':
		num_topics = 30
	elif FLAGS.data == 'yelp':
		num_topics = 30
	elif FLAGS.data == 'amazon_binary':
		num_topics = 20
	elif FLAGS.data == 'framing_corpus':
		num_topics = 10
	
	label_is_bool = False
	if FLAGS.data in TextResponseDataset.CLASSIFICATION_SETTINGS:
		label_is_bool = True

	print("Running model", FLAGS.model, '..'*20)

	if FLAGS.pretrained or FLAGS.pretrained_prodlda or FLAGS.model == 'hstm-all-2stage':
		array = np.load(pretraining_file)
		beta = np.log(array['beta']).T
	else:
		beta=None

	if FLAGS.model == 'hstm-all-2stage':
		text_dataset = TextResponseDataset(FLAGS.data, FLAGS.datafile, proc_file, pretrained_theta=theta_pretrained)
	else:
		text_dataset = TextResponseDataset(FLAGS.data, FLAGS.datafile, proc_file)

	text_dataset.process_dataset()
	text_dataset.preprocessing()

	total_docs = text_dataset.get_full_size()

	if FLAGS.train_test_mode:
		tr_indices = np.arange(0, FLAGS.train_size)
		te_indices = np.arange(FLAGS.train_size+1, n_docs)
	else:
		split_indices = util.cross_val_splits(total_docs, num_splits=FLAGS.num_folds)
		all_indices = np.arange(total_docs)
	
		te_indices = split_indices[FLAGS.split]
		tr_indices = np.setdiff1d(all_indices, te_indices)
	
	text_dataset.assign_splits(tr_indices, te_indices)

	train_params = {'batch_size': FLAGS.batch_size,
                'shuffle': True,
                'num_workers': 0
                }
	training_dataloader = DataLoader(text_dataset, **train_params)
	vocab_size = text_dataset.get_vocab_size()
	n_docs = len(text_dataset)
	

	if FLAGS.model == 'prodlda':
		model = topic_model.TopicModel(num_topics, vocab_size, n_docs, 
			beta_init=beta, 
			C=FLAGS.C)
	elif FLAGS.model == 'slda':
		model = slda.SupervisedLDA(num_topics, vocab_size, n_docs,
			label_is_bool=label_is_bool,
			beta_init=beta,
			predict_with_z=True)
	else:
		model = adjusted_hstm.HeterogeneousSupervisedTopicModel(num_topics, vocab_size, n_docs, 
		beta_init=beta, 
		label_is_bool=label_is_bool, 
		C_weights=FLAGS.C, 
		C_topics=FLAGS.C_topics,
		response_model=FLAGS.model)

	trainer = ModelTrainer(model, 
		model_name=FLAGS.model, 
		use_pretrained=(FLAGS.pretrained or FLAGS.pretrained_prodlda), 
		do_pretraining_stage=FLAGS.do_pretraining_stage, 
		do_finetuning=FLAGS.do_finetuning,
		save=FLAGS.save,
		load=FLAGS.load,
		model_file=FLAGS.model_file)
	
	trainer.train(training_dataloader, epochs=FLAGS.epochs, extra_epochs=FLAGS.extra_epochs)

	test_nll = trainer.evaluate_heldout_nll(text_dataset.te_counts, 
		theta=text_dataset.te_pretrained_theta)

	print("Held out neg. log likelihood:", test_nll)
	
	if FLAGS.model != 'prodlda':
		test_err, predictions = trainer.evaluate_heldout_prediction(text_dataset.te_counts, 
			text_dataset.te_labels,
			theta=text_dataset.te_pretrained_theta)

		test_mse = perplexity = npmi = shuffle_loss = 0.0
		test_auc = test_accuracy = test_log_loss = 0.0

		if label_is_bool:
			print("AUC on test set:", test_err[0], "Log loss:", test_err[1], "Accuracy:", test_err[2])
			test_auc = test_err[0]
			test_log_loss = test_err[1]
			test_accuracy = test_err[2]
		else:
			print("MSE on test set:", test_err)
			test_mse = test_err

	evaluator = Evaluator(model, 
		text_dataset.vocab, 
		text_dataset.te_counts, 
		text_dataset.te_labels, 
		text_dataset.te_docs,
		model_name=FLAGS.model)

	npmi_evaluator = Evaluator(model, 
		text_dataset.vocab, 
		text_dataset.counts, 
		text_dataset.labels, 
		text_dataset.docs,
		model_name=FLAGS.model)

	perplexity = evaluator.get_perplexity()
	npmi = npmi_evaluator.get_normalized_pmi_df()

	print("Perplexity:", perplexity)
	print("NPMI:", npmi)

	topics_str = evaluator.visualize_topics(format_pretty=True, num_words=7)

	if FLAGS.model in {'hstm-all', 'stm+bow', 'hstm-nobeta'}:
		bow_str = evaluator.visualize_word_weights(num_words=7)

	if FLAGS.model in {'hstm', 'hstm-all', 'hstm-nobeta'}:
		pos_topics_str = evaluator.visualize_supervised_topics(normalize=True, pos_topics=True, format_pretty=True, compare_to_bow=False, num_words=7)
		neg_topics_str = evaluator.visualize_supervised_topics(normalize=True, pos_topics=False, format_pretty=True, compare_to_bow=False, num_words=7)

		pos_npmi = npmi_evaluator.get_normalized_pmi_df(topics_to_use='pos')
		neg_npmi = npmi_evaluator.get_normalized_pmi_df(topics_to_use='neg')
		print("NPMI for positive topics and negative topics:", pos_npmi, neg_npmi)

	os.makedirs(FLAGS.outdir, exist_ok=True)
	np.save(os.path.join(FLAGS.outdir, FLAGS.model + '.result.' + 'split'+str(FLAGS.split) + '.setting=' + str((FLAGS.C,FLAGS.C_topics))), np.array([test_mse, test_auc, test_log_loss, test_accuracy, perplexity, npmi, shuffle_loss]))

	if FLAGS.print_latex:
		latex = evaluator.get_latex_for_topics(normalize=True,num_words=7, num_topics=5)
		print('\n', latex)

		print('\n', bow_str)


if __name__ == '__main__':
	FLAGS = flags.FLAGS
	flags.DEFINE_string('model', 'hstm-all', "type of response model.")
	flags.DEFINE_string("datafile", "", "path to file if using raw data files.")
	flags.DEFINE_string("procfile", "", "path to file for processed data.")
	flags.DEFINE_string("pretraining_file", "", "path to pretrained data.")
	flags.DEFINE_string("data", "amazon", "name of text corpus.")
	flags.DEFINE_string("framing_topic", "immigration", "only useful when using framing corpus; specifies which topic of articles to use.")
	flags.DEFINE_string("outdir", "../out/", "directory to which to write outputs.")
	flags.DEFINE_string("model_file", "../out/model/hstm-all.amazon", "file where model is saved.")

	flags.DEFINE_float("C", 1e-6, "l1 penalty for BoW weights and base rates.")
	flags.DEFINE_float("C_topics", 1e-6, "l1 penalty for gammas.")
	
	flags.DEFINE_integer("train_size", 10000, "number of samples to set aside for training split (only valid if train/test setting is used)")
	flags.DEFINE_integer("num_topics", 50, "number of topics to use.")
	flags.DEFINE_integer("batch_size", 512, "batch size to use in training.")
	flags.DEFINE_integer("split", 0, "split to use as the test data in cross-fold validation.")
	flags.DEFINE_integer("num_folds", 10, "number of splits for cross-fold validation (i.e. K in K-fold CV).")
	flags.DEFINE_integer("epochs", 10, "number of epochs for training.")
	flags.DEFINE_integer("extra_epochs", 10, "number of extra epochs to train supervised model.")
	
	flags.DEFINE_boolean("train_test_mode", False, "flag to use to run a train/test experiment instead of cross validation (default).")
	flags.DEFINE_boolean("pretrained", False, "flag to use pretrained LDA topics or not.")
	flags.DEFINE_boolean("pretrained_prodlda", False, "flag to use pretrained ProdLDA topics or not.")
	flags.DEFINE_boolean("do_pretraining_stage", False, "flag to run sgd steps for topic model only.")
	flags.DEFINE_boolean("do_finetuning", False, "flag to run sgd steps for response model only.")
	flags.DEFINE_boolean("save", False, "flag to save model.")
	flags.DEFINE_boolean("load", False, "flag to load saved model.")
	flags.DEFINE_boolean("print_latex", False, "flag to print latex for tables.")
	
	app.run(main)