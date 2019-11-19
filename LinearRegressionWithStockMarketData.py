import pandas as pd 
import numpy as np


def read_goog_sp500_dataframe():
	"""Returns a dataframe with the results for Google and S&P 500"""
  
		# Point to where you've stored the CSV file on your local machine
	googFile = 'data/GOOG.csv'
	spFile = 'data/SP_500.csv'
	
	goog = pd.read_csv(googFile, sep=",", usecols=[0,5], names=['Date','Goog'], header=0)
	sp = pd.read_csv(spFile, sep=",", usecols=[0,5], names=['Date','SP500'], header=0)
	
		# We merge the sp data frame onto the goog data frame.
	goog['SP500'] = sp['SP500']
	
		# The date object is a string, format it as a date
	goog['Date'] = pd.to_datetime(goog['Date'], format='%Y-%m-%d')
	goog = goog.sort_values(['Date'], ascending=[True])
	returns = goog[[key for key in dict(goog.dtypes) if dict(goog.dtypes)[key] in ['float64', 'int64']]]\
		.pct_change()
	return returns


def read_goog_sp500_logistic_data():
	"""Returns a dataframe with the results for Google and 
	S&P 500 set up for logistic regression"""
	returns = read_goog_sp500_dataframe()
	
	returns['Intercept'] = 1
	
		# Leave out the first row since it will not have a prediction for UP/DOWN
		# Leave out the last row as it will not have a value for returns
		# Resultant dataframe with the S&P500 and intercept values of all 1s
	xData = np.array(returns[["SP500", "Intercept"]][1:-1])
	
	yData = (returns["Goog"] > 0)[1:-1]
	
	return (xData, yData)


def read_goog_sp500_data():
	"""Returns a tuple with 2 fields, the returns for Google and the S&P 500.
	Each of the returns are in the form of a 1D array"""
	
	returns = read_goog_sp500_dataframe()
	
		# Filter out the very first row which does not have any value for returns
	xData = np.array(returns["SP500"])[1:]
	yData = np.array(returns["Goog"])[1:]
	
	return (xData, yData)


def read_xom_oil_nasdaq_data():
	"""Returns a tuple with 3 fields, the returns for Exxon Mobil, Nasdaq and oil prices.
	Each of the returns are in the form of a 1D array"""
	
	def readFile(filename):
			# Only read in the date and price at columns 0 and 5
		data = pd.read_csv(filename, sep=",", usecols=[0, 5], names=['Date', 'Price'], header=0)
		
			# Sort the data in ascending order of date so returns can be calculated
		data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
		
		data = data.sort_values(['Date'], ascending=[True])
		
			# Exclude the date from the percentage change calculation
		returns = data[[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['float64', 'int64']]]              .pct_change()
		
			# Filter out the very first row which has no returns associated with it
		return np.array(returns["Price"])[1:]

	nasdaqData = readFile('data/NASDAQ.csv')
	oilData = readFile('data/USO.csv')
	xomData = readFile('data/XOM.csv')
  
	return (nasdaqData, oilData, xomData)


from sklearn import datasets, linear_model

xData, yData = read_goog_sp500_data()


	# Set up a linear model to represent this
googModel = linear_model.LinearRegression()
	# Creates an array for each row.
		# This separates all groups of data into its own array.
googModel.fit(xData.reshape(-1,1), yData.reshape(-1,1))


	# Find the coefficient and intercept of this linear model
print (googModel.coef_)
print (googModel.intercept_)


import tensorflow as tf


	# Model linear regression y = Wx + b
W = tf.Variable(tf.zeros([1, 1]), name="W")
b = tf.Variable(tf.zeros([1]), name="b")

	# First dimension is none because we don't know how many data points there will be.
x = tf.placeholder(tf.float32, [None, 1], name="x")


	# x will have many rows and 1 column and W is a 1x1 matrix
	# Number of columns of x == number of rows for W
Wx = tf.matmul(x, W)

	# y is the predicted value.
y = Wx + b

	# y_ is the actual value.
		# The training data labels.
y_ = tf.placeholder(tf.float32, [None, 1], name="y_")


	# The cost function represents the "mean square error".
		# How accurate our program is and will adjust optimizers according to this number.
	# The mean of the square of the difference between actual value and predicted value.
cost = tf.reduce_mean(tf.square(y_ - y))

cost_hist = tf.summary.histogram("cost", cost)


train_step_ftrl = tf.train.FtrlOptimizer(1).minimize(cost)


	# Total number of points for our x values
dataset_size = len(xData)


def trainWithMultiplePointsPerEpoch(steps, train_step, batch_size):
	init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		sess.run(init)
		
		for i in range(steps):
			
			if dataset_size == batch_size:
				batch_start_idx = 0
			elif dataset_size < batch_size:
				raise ValueError("dataset_size: %d, must be greater than batch_size: %d" % (dataset_size, batch_size))
			else:
				batch_start_idx = (i * batch_size) % (dataset_size)
				
			batch_end_idx = batch_start_idx + batch_size
				
				# Access the x and y values in batches
			batch_xs = xData[batch_start_idx : batch_end_idx]
			batch_ys = yData[batch_start_idx : batch_end_idx]
				
				# Reshape the 1-D arrays as 2D feature vectors with many rows and 1 column
			feed = { x: batch_xs.reshape(-1, 1), y_: batch_ys.reshape(-1, 1) }
				
			sess.run(train_step, feed_dict=feed)
						
					# Print result to screen for every 500 iterations
			if (i + 1) % 500 == 0:
				print ("After %d iteration:" % i)
				print ("W: %f" % sess.run(W))
				print ("b: %f" % sess.run(b))
				print ("cost: %f" % sess.run(cost, feed_dict=feed))


trainWithMultiplePointsPerEpoch(5000, train_step_ftrl, dataset_size)

