from gym.envs.registration import register

register(
	id='tic_tac_chec',
	entry_point='gym_tic_tac_chec.envs:TTCEnv',
)
