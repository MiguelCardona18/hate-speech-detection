from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind, sem, t
import numpy as np
import pandas as pd

# Other models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# ADDED FOR FEATURES
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from data.preprocess_data import *




# CREATING MODEL 1
def generateNGramValues(X_train):
	count_vect = CountVectorizer(ngram_range=(1, 3), # N-GRAM RANGE CHANGED HERE
		preprocessor=preprocess, lowercase=True)
	X_train_counts = count_vect.fit_transform(X_train)
	return [X_train_counts, count_vect]

def generateTFIDFValues(X_train_counts):
	tf_transformer = TfidfTransformer(norm='l2', use_idf=True,
		smooth_idf=True, sublinear_tf=False)
	X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
	return [X_train_tfidf, tf_transformer]

def getdata(df_train):
	x_train,y_train=df_train['tweet'],df_train['class']
	return x_train,y_train

df_train = pd.read_csv("../data/datasets/en-fr/en_fr_train.csv", # THIS PATH CAN BE CHANGED TO USE DIFFERENT DATA
	index_col=False, lineterminator='\n')
X_train, Y_train = getdata(df_train)
X_train = df_train['tweet']
Y_train = df_train['class']

X_train_counts, count_vect = generateNGramValues(X_train)
X_train_tfidf, tf_transformer = generateTFIDFValues(X_train_counts)

log_regression = LogisticRegression(C=100, class_weight='balanced',
		solver='liblinear', penalty='l2', max_iter=100, multi_class='ovr')
clf1 = log_regression.fit(X_train_tfidf, Y_train)


# EVALUATION OF MODEL 1
df_test = pd.read_csv("../data/datasets/en-fr/en_fr_test.csv", # THIS PATH CAN BE CHANGED TO USE DIFFERENT DATA
    index_col=False, lineterminator='\n')
X_test = df_test['tweet']
Y_test = df_test['class']

obj = [count_vect, tf_transformer, clf1]
input_counts = obj[0].transform(X_test)
input_tfidf = obj[1].transform(input_counts)

cv1 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores1 = cross_val_score(clf1, input_tfidf, Y_test, scoring='accuracy', cv=cv1, n_jobs=-1)
print ('LogisticRegression with n-gram range 1-3 Mean Accuracy: %.3f (%.3f)' % (mean(scores1), std(scores1)))


#### SECOND MODEL ####


# CREATING MODEL 2
def generateNGramValues2(X_train):
	count_vect = CountVectorizer(ngram_range=(1, 4), # N-GRAM RANGE CHANGED HERE
		preprocessor=preprocess, lowercase=True)
	X_train_counts = count_vect.fit_transform(X_train)
	return [X_train_counts, count_vect]

def generateTFIDFValues2(X_train_counts):
	tf_transformer = TfidfTransformer(norm='l2', use_idf=True,
		smooth_idf=True, sublinear_tf=False)
	X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
	return [X_train_tfidf, tf_transformer]

X_train_counts2, count_vect = generateNGramValues2(X_train)
X_train_tfidf2, tf_transformer = generateTFIDFValues2(X_train_counts2)

log_regression2 = LogisticRegression(C=100, class_weight='balanced',
		solver='liblinear', penalty='l2', max_iter=100, multi_class='ovr')
clf2 = log_regression2.fit(X_train_tfidf2, Y_train)

df_test = pd.read_csv("../data/datasets/en-fr/en_fr_test.csv", # THIS PATH CAN BE CHANGED TO USE DIFFERENT DATA
    index_col=False, lineterminator='\n')
X_test = df_test['tweet']
Y_test = df_test['class']

obj = [count_vect, tf_transformer, clf2]
input_counts = obj[0].transform(X_test)
input_tfidf = obj[1].transform(input_counts)

cv1 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores2 = cross_val_score(clf2, input_tfidf, Y_test, scoring='accuracy', cv=cv1, n_jobs=-1)
print ('LogisticRegression2 Mean Accuracy: %.3f (%.3f)' % (mean(scores2), std(scores2)))


print(scores1) 
print(scores2)
stat, p = ttest_ind(scores1, scores2)
print('t=%.3f, p=%.3f' % (stat, p))



# Calculating t-test
def independent_ttest(data1, data2, alpha):
	# means
	mean1, mean2 = mean(data1), mean(data2)
	# standard errors
	se1, se2 = sem(data1), sem(data2)
	# standard error on the difference between  samples
	sed = np.sqrt(se1**2.0 + se2**2.0)
	# t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = len(data1) + len(data2) - 2
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	#  p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p

t_stat, df, cv, p = independent_ttest(scores1, scores2, 0.05)
print('t=%.3f, df=%.3f, cv=%.3f, p=%.5f' % (t_stat, df, cv, p))