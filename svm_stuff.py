# Import all the stuff we need
import pandas as pd
import numpy as np
import sklearn
import sklearn.svm

# Define a few constants:

# Threshold for determining outbreak
thrs = 10

# Function to help with visualization
def show_more_cols_head(data, num_cols=100):
    # Store the previous number of columns
    prev_maxcols = pd.options.display.max_columns
    # Set the number of columns to the max temporarily
    pd.options.display.max_columns = num_cols
    # Show the stuff
    print data.head()
    # Change it back to the previous value
    pd.options.display.max_columns = prev_maxcols
    # Done
    return

# Create data frames from CSVs
student_df = pd.read_csv("../Project/california-kindergarten-immunization-rates/StudentData.csv")
pertusis_df = pd.read_csv("../Project/california-kindergarten-immunization-rates/pertusisRates2010_2015.csv")
infant_df = pd.read_csv("../Project/california-kindergarten-immunization-rates/InfantData.csv")
geo_df = pd.read_csv("../Project/california-kindergarten-immunization-rates/geoData.csv")
# Get collection of the counties and years
counties =  student_df.COUNTY.unique()
years = student_df.year.unique()

# Get the differences between years and the sign of the amount (so we can see if it was trending positive or negative)
pertusis_df['2011_diff'] = pertusis_df['Rate2011'] - pertusis_df['Rate2010']
pertusis_df['2011_sgn'] = np.sign(pertusis_df['2011_diff'] - thrs)
pertusis_df['2012_diff'] = pertusis_df['Rate2012'] - pertusis_df['Rate2011']
pertusis_df['2012_sgn'] = np.sign(pertusis_df['2012_diff'] - thrs)
pertusis_df['2013_diff'] = pertusis_df['Rate2013'] - pertusis_df['Rate2012']
pertusis_df['2013_sgn'] = np.sign(pertusis_df['2013_diff'] - thrs)
pertusis_df['2014_diff'] = pertusis_df['Rate2014'] - pertusis_df['Rate2013']
pertusis_df['2014_sgn'] = np.sign(pertusis_df['2014_diff'] - thrs)

# Combine infant and pertusis data
concat = pd.merge(pertusis_df, infant_df, left_on='county', right_on='COUNTY')
# Delete new county data so we only have one
concat = concat.drop(['COUNTY'], axis=1)
# Display part of the data to see what's up
# show_more_cols_head(concat)

# Gather X and Y data
# For now, X holds the difference in rates from 2011-2012 and 2012-2013
# We need to grab the stuff as a matrix
X = concat[['Cases2012', 'Rate2012', 'Cases2013', 'Rate2013']].as_matrix()
# Y is predicting whether there was an outbreak in the number of pertussis cases
y = np.ravel(concat['2014_sgn'].as_matrix())

# printX

# Create SVM model
svm_model = sklearn.svm.SVC(kernel='poly', degree=3)

# Choose number of folds for kfolds:
n_fld = 5
kf = sklearn.model_selection.KFold(n_fld)
sklearn.model_selection.KFold(n_fld, random_state=None, shuffle=False)


cf = np.zeros([2, 2])
# # Confusion matrix stuff
for train_index, test_index in kf.split(X):
    # For each fold, split  X and y into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Try fitting the SVM model on the data
    svm_model.fit(X_train, y_train)
    # Predict using the test data, and compare to the actual values
    y_pred = svm_model.predict(X_test)
    cf = cf + sklearn.metrics.confusion_matrix(y_test, y_pred)


print cf

# Compute accuracy
acc = cf.trace()/cf.sum()
print acc