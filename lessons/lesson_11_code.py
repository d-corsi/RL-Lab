import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections


# TODO: implement the following functions as in the previous lessons
def createDNN( nInputs, nOutputs, nLayer, nNodes, last_activation ): raise NotImplementedError
def training_loop( env, actor_net, critic_net, updateRule, frequency=10, episodes=100 ): raise NotImplementedError
def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99, observation_number=None ): raise NotImplementedError


# TODO: implement the following class
class OverrideReward( gymnasium.wrappers.NormalizeReward ):
	"""
	Gymansium wrapper useful to update the reward function of the environment

	"""

	def step(self, action):
		previous_observation = np.array(self.env.state, dtype=np.float32)
		observation, reward, terminated, truncated, info = self.env.step( action )
		
		#TODO: extract the information from the observations
		#TODO: override the reward function before the return

		return observation, reward, terminated, truncated, info
	

def main(): 
	print( "\n***************************************************" )
	print( "*  Welcome to the eleventh lesson of the RL-Lab!  *" )
	print( "*                 (DRL in Practice)               *" )
	print( "***************************************************\n" )

	_training_steps = 2000

	# Crete the environment and add the wrapper for the custom reward function
	gymnasium.envs.register(
		id='MountainCarMyVersion-v0',
		entry_point='gymnasium.envs.classic_control:MountainCarEnv',
		max_episode_steps=1000
	)
	env = gymnasium.make( "MountainCarMyVersion-v0" )
	env = OverrideReward(env)
		
	# Create the networks and perform the actual training
	actor_net = createDNN( None, None, nLayer=None, nNodes=None, last_activation=None )
	critic_net = createDNN( None, None, nLayer=None, nNodes=None, last_activation=None )
	rewards_training, ep_lengths = training_loop( env, actor_net, critic_net, None, frequency=None, episodes=_training_steps  )

	# Save the trained neural network
	actor_net.save( "MountainCarActor.h5" )

	# Plot the results
	t = np.arange(0, _training_steps)
	plt.plot(t, ep_lengths, label="A2C", linewidth=3)
	plt.xlabel( "epsiodes", fontsize=16)
	plt.ylabel( "length", fontsize=16)
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()	