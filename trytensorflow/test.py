import tensorflow
from tensorflow import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from sklearn.utils import shuffle
import pickle
from matplotlib import style

#Maybe this might cause errors because there were 2 environments. Check tomorrow again.
#Is is because I was in a wrong environment or what?
#There seems to be 2 environments or something?
#*trytensorflow is just my project name so shouldn't have any problem...

#*Another stupid keras error.
#AttributeError: module 'keras.utils.generic_utils' has no attribute 'populate_dict_with_module_objects'

#*The problem is that I don't even know which version of tensorflow I am installing.
#*Let's GOOOO random stuff off the internet saves the day: https://github.com/keras-team/keras/releases

#*Data is separated by semicolons.
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#*Really similar to sentdex's method. I don't know who will explain it better though...

#*Grab the first 5 elements.

predict = "G3"
#Label is G3.

#*Returns a new data frame that does not include predict.
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#*Split some for training and some for testing.
#*Don't wanna just memorize.
#*testing 10% of the video.


#*Try to make the model better and better.

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
'''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)
    
    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            #Create file.
            pickle.dump(linear, f)
'''
#*Basically with statements are used for dealing with files.


pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


#*All I have to do for linear regression.


#*Give coefficients for all 5 variables.
#print(f"Co: {linear.coef_}")
#print(f"Inercept {linear.intercept_}")

predictions = linear.predict(x_test)

#*We could do some comparisons showing the predicted vs actual.
for x in range(len(predictions)):
    #y_test[x] is the actual y-value.
    print(predictions[x], x_test[x], y_test[x])

#*Wanna save the most accurate model instead of retraining it all the time
#X label is gonna be G1 in this case. 
p = "studytime"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

#*K nearest neighbor: classification algo. 

