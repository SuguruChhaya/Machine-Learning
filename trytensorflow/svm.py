#Soon get into a neural network and stuff.
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import  metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()
#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#1 is benign.
#print(x_train, y_train)

classes  = ["malignant", "benign"]
#For later printing.
#Try to use hyperplane to divide data in straight stuff.
#So many different hyperplanes can exist.
#Largest distance we can pick.
#Important to have a greater margin to separate classes.

#use kernels sometimes -> turn into form that we can draw.
#Repeat process if it doesn't divide nicely. Hope it changes.
#Soft margin: some outlier points can exist in outside.

#C is the amount on the margin.
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)
#K nearest works OK to.

