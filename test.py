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