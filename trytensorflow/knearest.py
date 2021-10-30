import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
#print(data.head())

#*Convert non-numerical data to numberical data.
#Encode into appropriate object.
le = preprocessing.LabelEncoder()
#*Get all the "buying" column data and put them into list and then do conversion.
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["persons"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

#*Create tuple objects into list/
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#Looks for grouping of algorithms. k standads of amount of neighbors closest to the question point. Pick k to be an odd where we always have a winner. 
#Have to pick the correct k value. 
#have to find the distance between all points. 
#Cannot save data points because heavy. 

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

predictions = model.predict(x_test)

#Could create an array to match 0, 1, 2, 3
#names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predictions)):
    #y_test[x] is the actual y-value.
    print(predictions[x], x_test[x], y_test[x])
    n = model.kneighbors([x_test[x]], 9, True)
    print(n)


