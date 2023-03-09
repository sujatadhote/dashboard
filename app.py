import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# make containers
header = st.container()
data_sets = st.container()
features = st.container()
model_training = st.container()

with header:
	st.title("Titanic")
	st.text("Work on titanic data")

with data_sets:
	st.header("Data sets")
	st.text("Work on titanic data sets")
	# import data
	df = sns.load_dataset('titanic')
	df = df.dropna()
	st.write(df.head(10))

	st.subheader("Total people")
	st.bar_chart(df['sex'].value_counts())

	#Other plot
	st.subheader("Class")
	st.bar_chart(df['class'].value_counts())
	# barplot
	st.bar_chart(df['age'].sample(10))


with features:
	st.title("Feature")
	st.text("Work on titanic features")
	st.markdown('1. **Feature 1:** Feature')

with model_training:
	st.title("Model training")
	st.text("Work on titanic data training")
	# making columns
	input, display = st.columns(2)

	#First column contain selection points
	max_depth = input.slider("How many people do you know", min_value=10, max_value=100, value=20, step=5)

# n_estimators
n_estimators = input.selectbox("How many tree should be there in a RF", option=[50,100,200,300,'No limit'])

# adding list of features
input_write(df.columns) 

# input feature from user
input_features = input.text_input('Which feature we shoud use')

# machine learning model

model = RandomForestRegressor(max_depth = max_depth, n_estimators=n_estimators)
# condition
if n_estimators == 'No limit':
   model = RandomForestRegressor(max_depth=max_depth)
else:
   model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators) 


#define x and y
x = df[[input_features]]
y = df[['fare']]

#fit our model
model.fit(x,y)
pred = model.predict(y)

#Display metrices

display.subheader("Mean absolute error of the model is: ")
display.write(mean_absolute_error(y,pred))
display.subheader("Mean squared error of the model is: ")
display.write(mean_squared_error(y,pred))
display.subheader("R squared score of the model is: ")
display.write(r2_score(y,pred))
 
	

