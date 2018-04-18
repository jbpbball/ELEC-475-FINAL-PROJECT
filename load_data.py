import pandas as pd
import numpy as np
from sklearn.svm import SVR


student_df = pd.read_csv("../Project/california-kindergarten-immunization-rates/StudentData.csv")
pertusis_df = pd.read_csv("../Project/california-kindergarten-immunization-rates/pertusisRates2010_2015.csv")
#print student_df.loc[np.where(np.logical_and(student_df['year'] == 2000, student_df['COUNTY'] == 'YUBA'))]
counties =  student_df.COUNTY.unique()
years = student_df.year.unique()

# Adding a "-1" because Alpine was giving us issues
features = np.zeros((len(counties), 3))
p_rates = []

for i, county in enumerate(counties):
	# Alpine was giving us problems
	if county == "CALIFORNIA":
		continue
	print county

	# try:
	n = student_df.loc[np.where(np.logical_and(student_df[
	'year'] == 2009, student_df['COUNTY'] == county))]['n'].sum()
	features[i,0] = student_df.loc[np.where(np.logical_and(student_df[
	'year'] == 2009,student_df['COUNTY'] == county))]['nMMR'].sum()/n
	features[i,1] = student_df.loc[np.where(np.logical_and(student_df[
	'year'] == 2009, student_df['COUNTY'] == county))]['nDTP'].sum()/n
	features[i,2] = student_df.loc[np.where(np.logical_and(student_df[
	'year'] == 2009, student_df['COUNTY'] == county))]['nPolio'].sum()/n

	p_rates.append(pertusis_df.loc[i]['Rate2010'])

	# except:
	# 	print "Divide by zero error at ", i, county
	# 	continue

	
clf = SVR()
clf.fit(features, p_rates)

print clf.score(features, p_rates)