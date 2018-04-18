import pandas as pd
import numpy as np
from sklearn.svm import SVR


student_df = pd.read_csv("../Project/california-kindergarten-immunization-rates/StudentData.csv")
pertusis_df = pd.read_csv("../Project/california-kindergarten-immunization-rates/pertusisRates2010_2015.csv")
infant_df = pd.read_csv("../Project/california-kindergarten-immunization-rates/InfantData.csv")
geo_df = pd.read_csv("../Project/california-kindergarten-immunization-rates/geoData.csv")
#print student_df.loc[np.where(np.logical_and(student_df['year'] == 2000, student_df['COUNTY'] == 'YUBA'))]
counties =  student_df.COUNTY.unique()
years = student_df.year.unique()

print infant_df