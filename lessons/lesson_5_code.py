import os, sys, numpy
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def epsilon_greedy(q, state, epsilon):
	"""
	Epsilon-greedy action selection function
	
	Args:
		q: q table
		state: agent's current state
		epsilon: epsilon parameter
	
	Returns:
		action id
	"""
	if numpy.random.random() < epsilon:
		return numpy.random.choice(q.shape[1])
	return q[state].argmax()


def dynaQ( environment, maxiters=250, n=10, eps=0.3, alfa=0.3, gamma=0.99 ):
	"""
	Implements the DynaQ algorithm
	
	Args:
		environment: OpenAI Gym environment
		maxiters: timeout for the iterations
		n: steps for the planning phase
		eps: random value for the eps-greedy policy (probability of random action)
		alfa: step size for the Q-Table update
		gamma: gamma value, the discount factor for the Bellman equation
		
	Returns:
		policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
	"""	

	Q = numpy.zeros((environment.observation_space, environment.action_space))
	M = numpy.array([[[None, None] for _ in range(environment.action_space)] for _ in range(environment.observation_space)])
	#
	# YOUR CODE HERE!
	#
	policy = Q.argmax(axis=1) 
	return policy



def main():
	print( "\n************************************************" )
	print( "*   Welcome to the fifth lesson of the RL-Lab!   *" )
	print( "*                  (Dyna-Q)                      *" )
	print( "**************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld( deterministic=True )
	env.render()

	print( "\n6) Dyna-Q" )
	dq_policy_n00 = dynaQ( env, n=0  )
	dq_policy_n25 = dynaQ( env, n=25 )
	dq_policy_n50 = dynaQ( env, n=50 )

	env.render_policy( dq_policy_n50 )
	print()
	print( f"\tExpected reward with n=0 :", env.evaluate_policy(dq_policy_n00) )
	print( f"\tExpected reward with n=25:", env.evaluate_policy(dq_policy_n25) )
	print( f"\tExpected reward with n=50:", env.evaluate_policy(dq_policy_n50) )
	
	

if __name__ == "__main__":
	main()