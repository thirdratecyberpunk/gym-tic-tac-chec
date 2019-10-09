from gym.envs.registration import register

register(
	id='tic-tac-chec',
	entry_point='gym_hive.envs:TTCEnv',
)
