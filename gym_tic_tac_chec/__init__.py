from gym.envs.registration import register

register(
	id='TTCVsSelf-v0',
	entry_point='gym_tic_tac_chec.envs:TTCEnv',
)
