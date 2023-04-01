import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections


def createDNN( nInputs, nOutputs, nLayer, nNodes ): 
	raise NotImplementedError


def training_loop( env, neural_net, updateRule, frequency=10, episodes=100 ):
	"""
	Main loop of the reinforcement learning algorithm. Execute the actions and interact
	with the environment to collect the experience for the training.

	Args:
		env: gymnasium environment for the training
		neural_net: the model to train 
		updateRule: external function for the training of the neural network
		
	Returns:
		averaged_rewards: array with the averaged rewards obtained

	"""

	#TODO: initialize the optimizer 
	optimizer = None 
	rewards_list, reward_queue = [], collections.deque( maxlen=100 )
	memory_buffer = []
	for ep in range(episodes):

		#TODO: reset the environment and obtain the initial state
		state = None 
		ep_reward = 0
		while True:

			#TODO: select the action to perform
			action = None 

			#TODO: Perform the action, store the data in the memory buffer and update the reward
			memory_buffer.append( None )
			ep_reward += None

			#TODO: exit condition for the episode
			if False: break

			#TODO: update the current state
			state = None

		#TODO: Perform the actual training every 'frequency' episodes
		updateRule( neural_net, memory_buffer, optimizer )

		# Update the reward list to return
		reward_queue.append( ep_reward )
		rewards_list.append( np.mean(reward_queue) )
		print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f})" )

	# Close the enviornment and return the rewards list
	env.close()
	return rewards_list



def REINFORCE_naive( neural_net, memory_buffer, optimizer ):
	"""
	Main update rule for the REINFORCE process, the naive implementation of the policy-gradient theorem.

	"""

	#TODO: Setup the tape
		#TODO: Initialize the array for the objectives, one for each episode considered
		#TODO: Iterate over all the trajectories considered
			#TODO: Extract the information from the buffer (for the considered episode)
			#TODO: Compute the log-prob of the current trajectory
			#TODO: Implement the update rule, notice that the REINFORCE objective 
			# is the sum of the logprob (i.e., the probability of the trajectory)
			# multiplied by the sum of the reward

		#TODO: Compute the final final objective to optimize

	raise NotImplemented


def REINFORCE_rw2go( neural_net, memory_buffer, optimizer ):
	"""
	Main update rule for the REINFORCE process, with the addition of the reward-to-go trick,

	"""

	raise NotImplementedError


def main():
	print( "\n*************************************************" )
	print( "*  Welcome to the ninth lesson of the RL-Lab!   *" )
	print( "*                 (REINFORCE)                   *" )
	print( "*************************************************\n" )

	_training_steps = 1500
	env = gymnasium.make( "CartPole-v1" )

	# Training A)
	neural_net = createDNN( 4, 2, nLayer=2, nNodes=32)
	rewards_naive = training_loop( env, neural_net, REINFORCE_naive, episodes=_training_steps  )
	print()

	# Training B)
	neural_net = createDNN( 4, 2, nLayer=2, nNodes=32)
	rewards_rw2go = training_loop( env, neural_net, REINFORCE_rw2go, episodes=_training_steps  )

	# Plot
	t = np.arange(0, _training_steps)
	plt.plot(t, rewards_naive, label="naive", linewidth=3)
	plt.plot(t, rewards_rw2go, label="reward to go", linewidth=3)
	plt.xlabel( "epsiodes", fontsize=16)
	plt.ylabel( "reward", fontsize=16)
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()	
