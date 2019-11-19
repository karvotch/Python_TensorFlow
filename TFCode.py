import tensorflow as tf

	# Not necessary to use 'dtype=', can just use 'tf.float32'.
	# A data type object (dtype) (an instance of numpy.dtype class) describes 
		# how the bytes in the fixed-size block of memory corresponding to an array item should be interpreted.
var_x = tf.Variable([1.5, 3.0, 6.0], dtype=tf.float32, name="var_x")
var_y = tf.Variable([10, 12.1, 33.7], dtype=tf.float32, name="var_y")
	# Initialized variables here.
init = tf.global_variables_initializer()

PH_x = tf.placeholder(tf.float32, name="PH_x")
PH_y = tf.placeholder(tf.float32, name="PH_y")

sum_var_x = tf.reduce_sum(var_x, name="sum_var_x")
sum_var_y = tf.reduce_sum(var_y, name="sum_var_y")
sum_PH_x = tf.reduce_sum(PH_x, name="sum_PH_x")
sum_PH_y = tf.reduce_sum(PH_y, name="sum_PH_y")

sum_var_x_y = tf.reduce_sum((sum_var_x, sum_var_y), name="sum_var_x_y")
sum_PH_x_y = tf.reduce_sum((sum_PH_x, sum_PH_y), name="sum_PH_x_y")

div_var_x_y = tf.div(var_x, var_y, name="div_var_x_y")
div_PH_x_y = tf.div(PH_x, PH_y, name="div_PH_x_y")
div_total = tf.div(sum_var_x_y, sum_PH_x_y, name="div_total")


x_PH = [13.3, 69.0, 99.1]
y_PH = [27.45, 48.13, 13.9]

with tf.Session() as sess:
		# Must run the global variable initializer.
	sess.run(init)
	
		# Must be 'feed_dict', cannot be another term.
	print("(Var) x + y = ", sess.run(sum_var_x_y))
	print("(PH) x + y = ", sess.run(sum_PH_x_y, feed_dict = {PH_x: x_PH, PH_y: y_PH}))
	print("(Var) x / y = ", sess.run(div_var_x_y))
	print("(PH) x / y = ", sess.run(div_PH_x_y, feed_dict = {PH_x: x_PH, PH_y: y_PH}))
	print("(Var, PH) (x + y) / (x + y) = ", sess.run(div_total, feed_dict = {PH_x: x_PH, PH_y: y_PH}))
	

	# Writing info to tensorboard.
writer = tf.summary.FileWriter('./TFCode', sess.graph)
writer.close()