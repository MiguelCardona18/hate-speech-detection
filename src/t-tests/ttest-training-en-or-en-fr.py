from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind, sem, t
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ADDED AS FEATURES.BUILD_FEATURES WAS GIVING PROBLEMS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from data.preprocess_data import *
from visualization.visualize import *
# 

# Required functions

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



# MODEL 1
# Getting Train data
df_train = pd.read_csv("../data/datasets/en-fr/en_fr_train.csv", index_col=False,
    lineterminator='\n')
X_train, Y_train = getdata(df_train)
X_train = df_train['tweet']
Y_train = df_train['class2']

X_train_counts, count_vect = generateNGramValues(X_train)
X_train_tfidf, tf_transformer = generateTFIDFValues(X_train_counts)
log_regression = LogisticRegression(C=100, class_weight='balanced',
		solver='liblinear', penalty='l2', max_iter=100, multi_class='ovr')
clf1 = log_regression.fit(X_train_tfidf, Y_train)

# Getting Test data
df_test = pd.read_csv("../data/datasets/en-fr/en_fr_test.csv",
    index_col=False, lineterminator='\n')
X_test = df_test['tweet']
Y_test = df_test['class2']

obj = [count_vect, tf_transformer, clf1]
input_counts = obj[0].transform(X_test)
input_tfidf = obj[1].transform(input_counts)

cv1 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores1 = cross_val_score(clf1, input_tfidf, Y_test, scoring='accuracy', cv=cv1, n_jobs=-1)
print ('Trained in English & French Mean Accuracy: %.3f (%.3f)' % (mean(scores1), std(scores1)))






# MODEL 2
# Getting Train data
df_train2 = pd.read_csv("../data/datasets/en/en_train.csv", index_col=False,
    lineterminator='\n')
X_train2, Y_train2 = getdata(df_train2)
X_train2 = df_train2['tweet']
Y_train2 = df_train2['class2']

X_train_counts2, count_vect = generateNGramValues(X_train2)
X_train_tfidf2, tf_transformer = generateTFIDFValues(X_train_counts2)
log_regression2 = LogisticRegression(C=100, class_weight='balanced',
		solver='liblinear', penalty='l2', max_iter=100, multi_class='ovr')
clf2 = log_regression2.fit(X_train_tfidf2, Y_train2)

# Getting Test data
df_test = pd.read_csv("../data/datasets/en-fr/en_fr_test.csv",
    index_col=False, lineterminator='\n')
X_test = df_test['tweet']
Y_test = df_test['class2']

obj = [count_vect, tf_transformer, clf2]
input_counts = obj[0].transform(X_test)
input_tfidf = obj[1].transform(input_counts)

cv2 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores2 = cross_val_score(clf2, input_tfidf, Y_test, scoring='accuracy', cv=cv2, n_jobs=-1)

print('Trained in English Mean Accuracy: %.3f (%.3f)' % (mean(scores2), std(scores2)))
print(scores1)
print(scores2)


stat, p = ttest_ind(scores1, scores2)
print('t=%.3f, p=%.3f' % (stat, p))



# function for calculating the t-test for two independent samples
def independent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = mean(data1), mean(data2)
	# calculate standard errors
	se1, se2 = sem(data1), sem(data2)
	# standard error on the difference between the samples
	sed = np.sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = len(data1) + len(data2) - 2
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p

t_stat, df, cv, p = independent_ttest(scores1, scores2, 0.05)
print('t=%.3f, df=%.3f, cv=%.3f, p=%.5f' % (t_stat, df, cv, p))
