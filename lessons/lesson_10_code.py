import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections


# TODO: implement the following functions as in the previous lessons
# Notice that the value function has only one output with a linear activation
# function in the last layer
def createDNN( nInputs, nOutputs, nLayer, nNodes, last_activation ): raise NotImplementedError
def training_loop( env, actor_net, critic_net, updateRule, frequency=10, episodes=100 ): raise NotImplementedError


def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99 ):

	"""
	Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
	and for the critic network (or value function)

	"""
	
	#TODO: implement the update rule for the critic (value function)
	for _ in range(10):
		# Shuffle the memory buffer
		np.random.shuffle( memory_buffer )
		#TODO: extract the information from the buffer
		# Tape for the critic
		with tf.GradientTape() as critic_tape:
			#TODO: Compute the target and the MSE between the current prediction
			# and the expected advantage 
			#TODO: Perform the actual gradient-descent process
			raise NotImplementedError

	#TODO: implement the update rule for the actor (policy function)
	#TODO: extract the information from the buffer for the policy update
	# Tape for the actor
	with tf.GradientTape() as actor_tape:
		#TODO: compute the log-prob of the current trajectory and 
		# the objective function, notice that:
		# the REINFORCE objective is the sum of the logprob (i.e., the probability of the trajectory)
		# multiplied by advantage
		#TODO: compute the final objective to optimize, is the average between all the considered trajectories
		raise NotImplementedError
	

def main(): 
	print( "\n*************************************************" )
	print( "*  Welcome to the tenth lesson of the RL-Lab!   *" )
	print( "*                    (A2C)                      *" )
	print( "*************************************************\n" )

	_training_steps = 2500

	env = gymnasium.make( "CartPole-v1" )
	actor_net = createDNN( 4, 2, nLayer=2, nNodes=32, last_activation="softmax")
	critic_net = createDNN( 4, 1, nLayer=2, nNodes=32, last_activation="linear")
	rewards_naive = training_loop( env, actor_net, critic_net, A2C, episodes=_training_steps  )

	t = np.arange(0, _training_steps)
	plt.plot(t, rewards_naive, label="A2C", linewidth=3)
	plt.xlabel( "epsiodes", fontsize=16)
	plt.ylabel( "reward", fontsize=16)
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()	
