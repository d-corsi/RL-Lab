import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
import tensorflow as tf; import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from DangerousGridWorld import GridWorld


def mse( network, dataset_input, target ):
	"""
	Compute the MSE loss function

	"""
	
	# Compute the predicted value, over time this value should
	# looks more like to the expected output (i.e., target)
	predicted_value = network( dataset_input )
	
	# Compute MSE between the predicted value and the expected labels
	mse = tf.math.square(predicted_value - target)
	mse = tf.math.reduce_mean(mse)
	
	# Return the averaged values for computational optimization
	return mse


def objective( x, y):
	"""
	Implements the following simple 2-variables function to optimize:
		2x^2 + 2xy + 2y^2 - 6x

	"""

	return 2*x**2 + 2*x*y + 2*y**2 - 6*x


def findMinimum( objective_function, n_iter=5000 ):
	"""
	Function that find the assignements to the variables that minimize the objective function,
	exploiting TensorFlow.

	Args:
		objective_function: the objective function to minimize
		n_iter: rnumber of iteration for the gradient descent process
		
	Returns:
		x: the best assignement for variable 'x'
		y: the best assignement for variable 'y'

	"""
	
	x = tf.Variable(0.0, name='x')
	y = tf.Variable(0.0, name='y')
	optimizer = tf.keras.optimizers.SGD( learning_rate=0.001 )
	#
	# YOUR CODE HERE!
	#
	return x.numpy(), y.numpy()


def createDNN( nInputs, nOutputs, nLayer, nNodes ):
	"""
	Function that generates a neural network with the given requirements.

	Args:
		nInputs: number of input nodes
		nOutputs: number of output nodes
		nLayer: number of hidden layers
		nNodes: number nodes in the hidden layers
		
	Returns:
		model: the generated tensorflow model

	"""
	
	# Initialize the neural network
	model = Sequential()
	#
	# YOUR CODE HERE!
	#
	return model


def collect_random_trajectories( env, num_episodes=10 ):
	"""
	Function that collect a dataset from the environment with an iterative
	interaction process

	Args:
		env: the environment in the gym-like format on which collect the data
		num_episodes: number of episodes to perform in the environment
		
	Returns:
		memory_buffer: an array with the collected data

	"""

	memory_buffer = []

	for _ in range(num_episodes):
		state = env.random_initial_state()
		#
		# YOUR CODE HERE!
		#
		
	return np.array(memory_buffer)


def trainDNN( model, memory_buffer, epoch=20 ):

	"""
	Function that perform the gradient descent training loop based on the data collected;
	the objective is to generate a neural network able to predict the reward of a state 
	given in input.

	Args:
		model: the initial model before the training phase
		memory_buffer: an array with the collected data
		epoch: number of gradient descent iteration
		
	Returns:
		model: the trained model

	"""

	# Preprocess data
	dataset_input = np.vstack(memory_buffer[:, 2])
	target = np.vstack(memory_buffer[:, 3])

	#
	# YOUR CODE HERE!
	#
	return model


def main():
	print( "\n************************************************" )
	print( "*  Welcome to the seventh lesson of the RL-Lab!  *" )
	print( "*        (Tensorflow and Neural Networks)        *" )
	print( "**************************************************" )

	# PART 1) Non Linear Optimization
	x, y = findMinimum( objective )
	print( f"\nA) The global minimum of the function: '2x^2 + 2xy + 2y^2 - 6x' is:")
	print( f"\t<x:{round(x, 2)}, y:{round(y, 2)}> with value {round(objective(x, y), 2)}")

	# PART 2) Creating a Deep Neural Network
	print( "\nB) Showing the deep neural network structure:")
	dnn_model = createDNN( nInputs=1, nOutputs=1, nLayer=2, nNodes=8 )
	dnn_model.summary()

	# PART 3) A Standard DRL Loop
	print( "\nC) Collect a dataset from the interaction with the environment")
	env = GridWorld()
	memory_buffer = collect_random_trajectories( env, num_episodes=10 )
	inp = np.array([[0], [48]])

	# PART 4) Train the DNN to predict the reward of given the state
	print( "\nD) Training a DNN to predict the reward of a state:")
	
	out = dnn_model( inp ).numpy()
	print( "Pre Training Reward Prediction: " )
	print( f"\tstate {inp[0][0]} => reward: {out[0][0]} ")
	print( f"\tstate {inp[1][0]} => reward: {out[1][0]} ")

	dnn_model = trainDNN( dnn_model, memory_buffer, epoch=1000 )

	out = dnn_model( inp ).numpy()
	print( "Post Training Reward Prediction: " )
	print( f"\tstate {inp[0][0]} => reward: {out[0][0]} ")
	print( f"\tstate {inp[1][0]} => reward: {out[1][0]} ")


if __name__ == "__main__":
	main()	
