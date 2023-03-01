import pickle
import numpy as np
import pandas as pd
from models.predict_model import predictResult
from sklearn import metrics
# from visualization.visualize import *
import itertools
import matplotlib.pyplot as plt
from os.path import abspath

FINAL_VECT = abspath("../../models/final_count_vect.pkl")
FINAL_TFIDF = abspath("../../models/final_tf_transformer.pkl")
FINAL_MODEL = abspath("../../models/final_model.pkl")

# Added from visualization as it was giving problems
def draw_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
	fig = plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.3f'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	plt.show()

if __name__ == '__main__':
	print("Reading testing data...")
	df_test = pd.read_csv("../data/datasets/fr-pr-tr/fr_pr_tr_test.csv", # CHANGE PATH TO TEST OTHER DATA
		index_col=False, lineterminator='\n')
	X_test = df_test['tweet']
	Y_test = df_test['class2'] # CHANGE THESE TO "class2" or "class" TO SEE EACH TASK 

	class_support = df_test.groupby('class2').size() # CHANGE THESE TO "class2" or "class" TO SEE EACH TASK 
	class_support = np.array(list(class_support))

	with open(FINAL_VECT, 'rb') as final_count_vect:
		count_vect = pickle.load(final_count_vect)
	with open(FINAL_TFIDF, 'rb') as final_tf_transformer:
		tf_transformer = pickle.load(final_tf_transformer)
	with open(FINAL_MODEL, 'rb') as final_model:
		lr_clf = pickle.load(final_model)

	obj = [count_vect, tf_transformer, lr_clf]
	print("Evaluating...")
	predicted = predictResult(X_test, obj)
	print("Accuracy: ", np.mean(predicted == Y_test))
	print("Classification report:\n", metrics.classification_report(Y_test, predicted))
	cm = metrics.confusion_matrix(Y_test, predicted)
	cm = (cm.T/class_support).T

	# CHANGE THESE TO SEE EACH TASK 

	# draw_confusion_matrix(cm, classes=['Hateful', 'Offensive', 'Clean'])
	draw_confusion_matrix(cm, classes=['Offensive', 'Clean'])
