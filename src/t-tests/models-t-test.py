from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from mlxtend.evaluate import paired_ttest_5x2cv

import numpy as np
import pandas as pd
from scipy import stats

# STOP WORDS
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop

# MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# ADDED AS FEATURES.BUILD_FEATURES WAS GIVING PROBLEMS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from data.preprocess_data import *

import numpy as np
import pandas as pd

# 
final_stopwords_list = list(fr_stop) + list(en_stop)

def generateNGramValues(X_train):
	count_vect = CountVectorizer(ngram_range=(1, 3),
		preprocessor=preprocess, lowercase=True)
	X_train_counts = count_vect.fit_transform(X_train)
	return [X_train_counts, count_vect]

def generateTFIDFValues(X_train_counts):
	tf_transformer = TfidfTransformer(norm='l2', use_idf=True,
		smooth_idf=True, sublinear_tf=False)
	X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
	return [X_train_tfidf, tf_transformer]
# ADDED UNTIL HERE

def getdata(df_train):
	x_train,y_train=df_train['tweet'],df_train['class2']
	return x_train,y_train

df_train = pd.read_csv("../data/datasets/en/en_train.csv", # CHANGE PATH TO TRAIN OTHER DATA 
	index_col=False, lineterminator='\n')
X_train, Y_train = getdata(df_train)
X_train = df_train['tweet']
Y_train = df_train['class'] # CHANGE TO "class2" FOR SECOND TASK 

X_train_counts, count_vect = generateNGramValues(X_train)
X_train_tfidf, tf_transformer = generateTFIDFValues(X_train_counts)
# 





# TRAIN MODEL 1
def generateNGramValues(X_train):
	count_vect = CountVectorizer(ngram_range=(1, 3),
		preprocessor=preprocess, lowercase=True)
	X_train_counts = count_vect.fit_transform(X_train)
	return [X_train_counts, count_vect]

def generateTFIDFValues(X_train_counts):
	tf_transformer = TfidfTransformer(norm='l2', use_idf=True,
		smooth_idf=True, sublinear_tf=False)
	X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
	return [X_train_tfidf, tf_transformer]

def getdata(df_train):
	x_train,y_train=df_train['tweet'],df_train['class2']
	return x_train,y_train

df_train = pd.read_csv("../data/datasets/en-fr/en_fr_train.csv", # CHANGE PATH TO TRAIN OTHER DATA
	index_col=False, lineterminator='\n')
X_train, Y_train = getdata(df_train)
X_train = df_train['tweet']
Y_train = df_train['class2']

X_train_counts, count_vect = generateNGramValues(X_train)
X_train_tfidf, tf_transformer = generateTFIDFValues(X_train_counts)
log_regression = LogisticRegression(C=100, class_weight='balanced',
		solver='liblinear', penalty='l2', max_iter=100, multi_class='ovr')
clf1 = log_regression.fit(X_train_tfidf, Y_train)


# TEST & EVALUATION
df_test = pd.read_csv("../data/datasets/en-fr/en_fr_test.csv", # CHANGE PATH TO TEST OTHER DATA
    index_col=False, lineterminator='\n')
X_test = df_test['tweet']
Y_test = df_test['class2']

obj = [count_vect, tf_transformer, clf1]
input_counts = obj[0].transform(X_test)
input_tfidf = obj[1].transform(input_counts)


cv1 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores1 = cross_val_score(clf1, input_tfidf, Y_test, scoring='accuracy', cv=cv1, n_jobs=-1)
print ('LogisticRegression Mean Accuracy: %.3f (%.3f)' % (mean(scores1), std(scores1)))



# third
clf3 = RandomForestClassifier(random_state=0)
clf3.fit(X_train_tfidf, Y_train)
# clf3 = DecisionTreeClassifier(random_state=0)
# clf3.fit(X_train_tfidf, Y_train)
# cv2 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

df_test = pd.read_csv("../data/datasets/en-fr/en_fr_test.csv",
    index_col=False, lineterminator='\n')
X_test = df_test['tweet']
Y_test = df_test['class2']

obj = [count_vect, tf_transformer, clf3]
input_counts = obj[0].transform(X_test)
input_tfidf = obj[1].transform(input_counts)

cv1 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores3 = cross_val_score(clf3, input_tfidf, Y_test, scoring='accuracy', cv=cv1, n_jobs=-1)

print('RandomForestClassifier Mean Accuracy: %.3f (%.3f)' % (mean(scores3), std(scores3)))

# check if difference between algorithms is real
t, p = paired_ttest_5x2cv(estimator1=clf1, estimator2=clf3, X=input_tfidf, y=Y_test, scoring='accuracy', random_seed=1)
# summarize
print('P-value: %.3f, t-Statistic: %.3f' % (p, t))
# interpret the result
if p <= 0.05:
	print('Difference between mean performance is probably real')
else:
	print('Algorithms probably have the same performance')

print (stats.ttest_ind(scores1, scores3, equal_var=False))
