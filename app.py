import streamlit as st 
import pandas as pd 
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import sys
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt
from dmba import regressionSummary

st.title('Data mining')
st.markdown(' step 1: Importing packages. We import all the necessary packages')
st.code("""import streamlit as st 
import pandas as pd 
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import sys
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt
from dmba import regressionSummary """)

st.markdown("step 2 : importing the data")
data_df = pd.read_csv('https://raw.githubusercontent.com/ashish-cell/BADM-211-FA21/main/Data/Credit_Quant.csv')
st.code("""data_df = pd.read_csv('https://raw.githubusercontent.com/ashish-cell/BADM-211-FA21/main/Data/Credit_Quant.csv')""")
st.write('The dataframe looks like: ')
st.dataframe(data_df)

st.markdown("step 3 : Inspect the dataframe")
st.write("The shape of the dataframe")
st.code("""data_df.shape""")
st.write(data_df.shape)

st.write("The top ten rows of the dataframe")
st.code("""data_df.head(10)""")
st.write(data_df.head(10))

st.write("The bottom ten rows of the dataframe")
st.code("""data_df.tail(10)""")
st.write(data_df.tail(10))

st.write("The columns of the dataframe")
st.code("""data_df.columns.to_list()""")
st.write(data_df.columns.to_list())

st.write("The datatypes of the dataframe")
st.code("""data_df.dtypes""")
df_types =  pd.DataFrame(data_df.dtypes, columns=['Data Type'])
st.dataframe(df_types.astype(str))

st.write("The info() method prints information about the DataFrame. The information contains the number of columns, column labels, column data types, memory usage, range index, and the number of cells in each column (non-null values). Note: the info() method actually prints the info.")
st.code("""data_df.info()""")
info = data_df.info()
st.write("""<class 'pandas.core.frame.DataFrame'>
RangeIndex: 310 entries, 0 to 309
Data columns (total 11 columns):
     Column             Non-Null Count  Dtype  
 0   Personal Income    310 non-null    float64
 1   Credit Limit       310 non-null    int64  
 2   Credit Rating      310 non-null    int64  
 3   Number of Cards    310 non-null    int64  
 4   Age                310 non-null    int64  
 5   Education Years    310 non-null    int64  
 6   Gender             310 non-null    object 
 7   Student            310 non-null    object 
 8   Married            310 non-null    object 
 9   Ethnicity          310 non-null    object 
 10   Credit Balance    310 non-null    float64
dtypes: float64(2), int64(5), object(4)
memory usage: 26.8+ KB""")

st.markdown("Step 4 : Data preprocessing")
st.write("Data preprocessing and cleaning is an important aspect of data analysis.Strip leading and trailing spaces (that is, spaces at the beginning and end of a column name) and replace any remaining spaces with an underscore _. We do this by creating a modified copy of columns and assigning it to the columns field of the dataframe. I do this in two steps below, but I show how to do this in one step in commented code.")
new_names = [s.strip().replace(' ', '_') for s in data_df.columns]
data_df.columns = new_names
st.code("""new_names = [s.strip().replace(' ', '_') for s in data_df.columns]
data_df.columns = new_names
data_df.columns""")
st.write("""Index(['Personal_Income', 'Credit_Limit', 'Credit_Rating', 'Number_of_Cards',
       'Age', 'Education_Years', 'Gender', 'Student', 'Married', 'Ethnicity',
       'Credit_Balance'],
      dtype='object')""")

st.write('Breaking down the code')
d = {'animal type ': ['dog', 'cat', 'bird'],'age in years': [1, 2, 3],'size':['6', '8', '10'],'city of residence': ['miami', 'chicago', 'london']}
df = pd.DataFrame(data = d)
df
st.code("""d = {'animal type ': ['dog', 'cat', 'bird'],'age in years': [1, 2, 3],'size':['6', '8', '10'],'city of residence': ['miami', 'chicago', 'london']}
df = pd.DataFrame(data = d)
df""")
st.write(df)

st.markdown("""Accessing subsets of the data
Pandas uses two methods to access rows and columns in a data frame: loc and iloc. DataCamp discussed both methods. Let's practice them here.

The loc method is label-based, meaning that you access rows and columns using their labels. (Sometimes those labels are integers!) The iloc method on the other hand only allows using integer numbers. To specify a range of rows use the slice notation, e.g. 0:9.

Remember, Python uses 0-indexing, which means that indices start at 0 and not at 1. This might be different from other languages you're familiar with, e.g. R.

Let's show the first four rows of the data frame. First we'll do it using loc, then using iloc.""")
st.code("""data_df.loc[0:3]  # for loc, the second index in the slice is inclusive """)
st.write(data_df.loc[0:3])

st.markdown("using iloc()")
st.code("""data_df.iloc[0:4]""")
st.write(data_df.iloc[0:4])

st.markdown("show the first ten rows of the first column. Notice that we can do this using either dot notation or bracket notation,")
st.code("""data_df['Personal_Income'].iloc[0:10]

""")
st.write(data_df['Personal_Income'].iloc[0:10])

st.markdown("Step 5 : Descriptive statistics.Use the describe method to print a number of common statistics. Below, apply it to a single column..")
st.code("""data_df['Personal_Income'].describe()""")
st.write(data_df['Personal_Income'].describe())

st.markdown("Now apply it to an entire dataframe. Note that when applying it the entire dataframe, it will only display results for numeric variables.")
st.code("""data_df.describe()""")
st.write(data_df.describe())

st.markdown("""Dummy Coding in Python

In Pandas, the get_dummies method is used to convert categorical variables into dummy variables, which are equal to 1 or 0.

The new dummy variable's will be a combination of the original variable name and value. For example, Student becomes Student_Yes and Student_No; the Student variable will no longer exist.

The parameter prefix_sep indicates what symbol should to separate the original variable name and value. In the above example, and in virtually all of the examples you will see in this class, we will use an underscore. That is, prefix_sep = '-'.

The parameter drop_first = True is used for linear models (regression and classification) that cannot function with a 100% collinearity between variables. For example, Student_Yes and Student_No are perfectly collinear; when Student_Yes = 1, Student_No = 0 and vice versa. There are models, such as decision trees, that can operate with perfectly collinear variables. In those cases, you would set drop_first = False.""")
st.code("""data_df = pd.get_dummies(data_df, prefix_sep='_', drop_first=True)

data_df.columns""")
data_df = pd.get_dummies(data_df, prefix_sep='_', drop_first=True)

st.write(data_df.columns)

st.markdown("""Split predictors and outcome variable.
In Python, before modeling we need to spearate the predictor and outcome variables. All the predictors are kept as a dataframe "X" and the outcome variable as a Pandas Series "y".

Create an object to hold just the predictors and another one to hold just the outcome variable.""")
st.code("""X = data_df.drop('Credit_Balance', axis=1) # all variables EXCEPT Credit_Balance

y = data_df['Credit_Balance'] # just the outcome variable""")
X = data_df.drop('Credit_Balance', axis=1) # all variables EXCEPT Credit_Balance

y = data_df['Credit_Balance'] # just the outcome variable

st.markdown("""Create a holdout dataset
We need to split the dataset into training (60%) and validation (40%) sets.
A couple of notes: 1) The terminology here can get confusing: we sometimes refer to the validation set as a test set or, as we used in the header, a holdout set. 2) We'll be taking about a more sophisticated version of this called cross-validation. But let's start here to get our bearings.

Below, randomly sample 60% of the dataset into a training dataset, where train_X holds the predictors and train_y holds the outcome variable. The remaining 40% serve as a validation set, where test_X holds the predictors and test_y holds the outcome variable.""")
st.code("""from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.40, random_state=12)

print('Training   : ', train_X.shape)
print('Validation : ', test_X.shape)""")
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.40, random_state=12)

st.write('Training   : ', train_X.shape)
st.write('Validation : ', test_X.shape)



st.markdown("train_X:")
st.code("train_X")
st.write(train_X)

st.markdown("test_X:")
st.code("test_X")
st.write(test_X)

st.markdown("train_y:")
st.code("train_y")
st.write(train_y)

st.markdown("test_y:")
st.code("test_y")
st.write(test_y)

st.markdown("""Step 6 : Create a linear regression model
Now, let's run a linear regression. We do this in two steps. 1) Load the linear regression algorithm into a model called model_lm 2) Fit the linear regression algorithm object to the training data, using the fit method, thus creating a model.""")
st.code("""# Step 1
model_lm = LinearRegression()  # This fucntion LinearRegression() is called from the SKLearn library we imported in the first cell of the notebook. 

# model_lm is the name we have given to LinearRegression Model. If you want you may change it. 

# Step 2
model_lm.fit(train_X, train_y) # This step is known as model fitting. To fit the model, you supply two arguments the set of predictors and the outcome variable from the trainign dataset""")
# Step 1
model_lm = LinearRegression()  # This fucntion LinearRegression() is called from the SKLearn library we imported in the first cell of the notebook. 

# model_lm is the name we have given to LinearRegression Model. If you want you may change it. 

# Step 2
model_lm.fit(train_X, train_y) # This step is known as model fitting. To fit the model, you supply two arguments the set of predictors and the outcome variable from the trainign dataset
st.write(model_lm)

st.markdown("We can print the coefficients in the model:")
st.code("""print('intercept:', model_lm.intercept_) # 
# print('slope:', model_lm.coef_)

print(pd.DataFrame({"variable_name": train_X.columns, "slope": model_lm.coef_}))""")
print('intercept:', model_lm.intercept_) # 
# print('slope:', model_lm.coef_)

print(pd.DataFrame({"variable_name": train_X.columns, "slope": model_lm.coef_}))
st.write(pd.DataFrame({"variable_name": train_X.columns, "slope": model_lm.coef_}))

st.markdown("""step 7: Check the performance results
Now let's check how well our model performed. What does this mean? We want to see how effective our model is at predicting the outcome variable for a new set of data. We do this by applying it to the test data.

In the code below, we compare the known outcome variables for the test data (test_y) to the predicted outcome variables for the test data (model_lm.predict(test_X)).""")
st.code("""# print performance measures of the training data
regressionSummary(test_y, model_lm.predict(test_X))""")

summary = pd.DataFrame(regressionSummary(test_y, model_lm.predict(test_X)))

st.write("""Regression statistics

                      Mean Error (ME) : 0.8009
       Root Mean Squared Error (RMSE) : 9.9218
            Mean Absolute Error (MAE) : 7.9308
          Mean Percentage Error (MPE) : 1.3929
Mean Absolute Percentage Error (MAPE) : 3.6940""")


