import pickle
import numpy as np
import pandas as pd
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

# ADDED AS FEATURES.BUILD_FEATURES WAS GIVING PROBLEMS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from data.preprocess_data import *
from os.path import abspath

FINAL_VECT = abspath("../../models/final_count_vect.pkl")
FINAL_TFIDF = abspath("../../models/final_tf_transformer.pkl")
FINAL_MODEL = abspath("../../models/final_model.pkl")

final_stopwords_list = list(fr_stop) + list(en_stop)

def generateNGramValues(X_train):
	count_vect = CountVectorizer(ngram_range=(1, 3), # n-gram range can be changed here
		preprocessor=preprocess, lowercase=True)
	X_train_counts = count_vect.fit_transform(X_train)
	return [X_train_counts, count_vect]

def generateTFIDFValues(X_train_counts):
	tf_transformer = TfidfTransformer(norm='l2', use_idf=True, # "l2" can be changed to "l1" for difference in normalization
		smooth_idf=True, sublinear_tf=False)
	X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
	return [X_train_tfidf, tf_transformer]
# ADDED UNTIL HERE

def getdata(df_train):
	x_train,y_train=df_train['tweet'],df_train['class2'] # CHANGE THESE TO "class2" or "class" TO SEE EACH TASK 
	return x_train,y_train

if __name__ == '__main__':
	print("Reading training data...")
	df_train = pd.read_csv("../data/datasets/fr-pr-tr/fr_pr_tr_train.csv", index_col=False,
		lineterminator='\n')
	X_train, Y_train = getdata(df_train)
	X_train = df_train['tweet']
	Y_train = df_train['class2'] # CHANGE THESE TO "class2" or "class" TO SEE EACH TASK 
	print(df_train)

	print("Building features...")
	print("Generating n-gram values...")
	print("Generating TFIDF values...")
	X_train_counts, count_vect = generateNGramValues(X_train)
	X_train_tfidf, tf_transformer = generateTFIDFValues(X_train_counts)

	print("Training model...")
	log_regression = LogisticRegression(C=100, class_weight='balanced',
		solver='liblinear', penalty='l2', max_iter=100, multi_class='ovr')
	clf = log_regression.fit(X_train_tfidf, Y_train)

	
	# clf = RandomForestClassifier(class_weight='balanced')
	# clf.fit(X_train_tfidf, Y_train)

	# from sklearn.neural_network import MLPClassifier
	# clf = MLPClassifier()
	# clf.fit(X_train_tfidf, Y_train)

	# clf = KNeighborsClassifier(n_neighbors=3)
	# clf.fit(X_train_tfidf, Y_train)
	
	
	# from sklearn.svm import SVC
	# clf = SVC(gamma='auto')
	# clf.fit(X_train_tfidf, Y_train)
	
	# from sklearn.tree import DecisionTreeClassifier
	# clf = DecisionTreeClassifier(random_state=0)
	# clf.fit(X_train_tfidf, Y_train)

	print("Model trained.")
	
	# Save features and models for predicting
	with open(FINAL_VECT, 'wb') as final_count_vect:
		pickle.dump(count_vect, final_count_vect, pickle.HIGHEST_PROTOCOL)
	with open(FINAL_TFIDF, 'wb') as final_tf_transformer:
		pickle.dump(tf_transformer, final_tf_transformer, pickle.HIGHEST_PROTOCOL)
	with open(FINAL_MODEL, 'wb') as final_model:
		pickle.dump(clf, final_model, pickle.HIGHEST_PROTOCOL)
