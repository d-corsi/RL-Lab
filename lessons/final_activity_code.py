import warnings; warnings.filterwarnings("ignore")
import sys, os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from point_discrete import PointNavigationDiscrete
import gymnasium, collections

# TODO: implement the following functions as in the previous lessons
def createDNN( nInputs, nOutputs, nLayer, nNodes, last_activation ): raise NotImplementedError
def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99, observation_number=None ): raise NotImplementedError

def training_loop( env, actor_net, critic_net, updateRule, frequency=10, episodes=100 ):
	"""
	Main loop of the reinforcement learning algorithm. Execute the actions and interact
	with the environment to collect the experience for the training.

	"""

	#TODO: initialize the optimizer 
	optimizer = None 
	rewards_list, reward_queue = [], collections.deque( maxlen=100 )
	success_list, success_queue = [], collections.deque( maxlen=100 )
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
		success_queue.append( info["goal_reached"] )
		success_list.append( np.mean(success_queue) )
		print( f"episode {ep:4d}: reward: {ep_reward:5.2f} (averaged: {np.mean(reward_queue):5.2f}), success rate ({int(np.mean(success_queue)*100):3d}/100)" )

	# Close the enviornment and return the rewards list
	env.close()
	return success_list



class OverrideReward( gymnasium.wrappers.NormalizeReward ):


	def step(self, action):
		observation, reward, terminated, truncated, info = self.env.step( action )

		# Extract the information from the observations
		old_heading, old_distance, old_lidars = self.previous_observation[0], self.previous_observation[1], self.previous_observation[2:]
		heading, distance, lidars = observation[0], observation[1], observation[2:]

		# Exploting useful flags
		goal_reached = bool(info["goal_reached"])
		collision = bool(info["collision"])
		
		# Override the reward function
		# here!

		return observation, reward, terminated, truncated, info
	

def main(): 
	print( "\n*****************************************************" )
	print( "*    Welcome to the final activity of the RL-Lab    *" )
	print( "*                                                   *" )
	print( "*****************************************************\n" )

	_training_steps = 1000
	
	# Load the environment and override the reward function
	env = PointNavigationDiscrete( ) #optional: render_mode="human"
	env = OverrideReward(env)

	# Create the networks and perform the actual training
	actor_net = createDNN( None, None, nLayer=None, nNodes=None, last_activation=None )
	critic_net = createDNN( None, None, nLayer=None, nNodes=None, last_activation=None )
	success_training = training_loop( env, actor_net, critic_net, None, frequency=None, episodes=_training_steps  )

	# Save the trained neural network
	actor_net.save( "TrainedPolicy.h5" )

	# Plot the results
	t = np.arange(0, _training_steps)
	plt.plot(t, success_training, label="A2C", linewidth=3)
	plt.xlabel( "epsiodes", fontsize=16)
	plt.ylabel( "success", fontsize=16)
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()	