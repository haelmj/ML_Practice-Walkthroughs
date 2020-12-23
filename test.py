import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# load data set into program
data = pd.read_csv("student-mat.csv", sep=";")
# filter out some columns
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


x_test, x_train, y_test, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

