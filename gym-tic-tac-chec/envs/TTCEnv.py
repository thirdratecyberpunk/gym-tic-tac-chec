import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

uniDict = {
"p": "♙", "r": "♖", "n": "♘", "b": "♗",
"P": "♟", "R": "♜", "N": "♞", "B": "♝",
".": "."
}

pieces_to_ids = {
"p" : 1, "r": 2, "n": 3, "b": 4,
"." : 0,
"P" : -1, "R": -2, "N": -3, "B": -4
}

sign = lambda x: (1, -1)[x < 0]

"""
AGENT POLICY
------------
"""
def make_random_policy(np_random):
	def random_policy(state):
		opp_player = -1
		moves = TTCEnv.get_possible_moves(state, opp_player)
		# No moves left
		if len(moves) == 0:
			return "resign"
		else:
			return np.random.choice(moves)
	return random_policy

class TTCEnv(gym.Env):
    """
    Class representing game information.
    """
    WHITE = 1
    BLACK = -1
    #metadata = {"render.modes": ["human"]}

    def __init__(self, player_color=1, opponent="random", log=True):
        # space containing all board states
        self.observation_space = spaces.Box(-4, 4, (4,4))
        # space containing all possible actions
        # 4 by 4 board with 4 possible pieces each
        self.action_space = spaces.Discrete(16 * 4)
        # defines the player as the given colour
        self.player = player_colour
        # defines opponent strategy
        self.opponent = opponent
        # resets and builds state
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        """
        resets random seed for random player
        """
        self.np_random, seed = seeding.np_random(seed)
        if isinstance(self.opponent, str):
            if self.opponent == "random":
                self.opponent_policy = make_random_policy(self.np_random)
            elif self.opponent == "none":
                self.opponent_policy = None
            else
                raise error.Error("Unrecognised opponent policy {}".format(self.opponent))
        else:
            self.opponent_policy = self.opponent
        return [seed]

    def _step(self, action):
		"""
		Run one timestep of the environment's dynamics. When end of episode
		is reached, reset() should be called to reset the environment's internal state.

		Input
		-----
		action : an action provided by the environment

		Outputs
		-------
		(observation, reward, done, info)
		observation : agent's observation of the current environment
		reward [Float] : amount of reward due to the previous action
		done : a boolean, indicating whether the episode has ended
		info : a dictionary containing other diagnostic information from the previous action
		"""
        # assert that the given action is valid
        assert self.action_space.contains(action),"INVALID ACTION {}".format(action)
        # return the state if the game is won
        if self.done:
            return self.state, 0., True, {'state': self.state}
        # player makes a move
        self.state, reward, self.done = self.player_move(
            self.current_player, self.state, action,
            render=self.log, render_msg= 'Player {}'.format(self.current_player)
        )
        # return the state if the game is won
        if self.done:
            return self.state, 0., True, {'state': self.state}

    	# player vs. player game
    	if not self.opponent_policy:
    		# +1 step
    		if self.current_player == -1:
    			self.state['on_move'] += 1
    		self.current_player *= -1
    		return self.state, reward, self.done, {'state': self.state}

    	# Bot Opponent play
    	#
    	else:
    		opp_move = self.opponent_policy(self.state)
    		opp_action = TTCEnv.move_to_actions(opp_move)

    		# make move
    		self.state, opp_reward, self.done = self.player_move(
    			-1, self.state, opp_action,
    			render=self.log, render_msg='Opponent')

    		total_reward = reward - opp_reward
    		self.state['on_move'] += 1
    		return self.state, total_reward, self.done, {'state': self.state}


    def _reset(self):
		"""
		Resets the state of the environment, returning an initial observation.
		Outputs -> observation : the initial observation of the space. (Initial reward is assumed to be 0.)
		"""
        # clears state
        self.state = {}
        self.done = False
        self.current_player = 1
        self.saved_states = {}
		pieces = np.linspace(1,4,4, dtype=int)
		# material that needs to be placed on the board
        self.state['captured'] = {1: [1,2,3,4], -1: [-1,-2,-3,-4]}
		# number of pieces remaining to place
		self.state['initial_placements'] = {1: 3, -1: 3}
		# direction that a pawn can move in
		self.state['pawn_direction'] = {1: 1, -1: -1}
		# current move number
		self.state['on_move'] = 1

		# board
		board = [["."] * 4] * 4
		self.state['board'] = np.array([[pieces_to_ids[x] for x in row] for row in board])
		self.state['prev_board'] = copy(self.state['board'])
		return self.state


	def _render(self, mode='human', close=False):
		return TTCEnv.render_board(self.state, mode=mode, close=close)

	@staticmethod
	def render_board(state, mode='human', close=False):
		"""
		Render the playing board
		"""
		board = state['board']
		outfile = StringIO() if mode == 'ansi' else sys.stdout

		outfile.write('    ')
		outfile.write('-' * 25)
		outfile.write('\n')

		for i in range(7,-1,-1):
			outfile.write(' {} | '.format(i+1))
			for j in range(7,-1,-1):
				piece = TTCEnv.ids_to_pieces[board[i,j]]
				figure = uniDict[piece[0]]
				outfile.write(' {} '.format(figure))
			outfile.write('|\n')
		outfile.write('    ')
		outfile.write('-' * 25)
		outfile.write('\n      a  b  c  d ')
		outfile.write('\n')
		outfile.write('\n')

		if mode != 'human':
			return outfile

	@staticmethod
	def render_moves(state, piece_id, moves, mode='human'):
		"""
		Render the possible moves that a piece can take
		"""
		board = state['board']
		moves_pos = [m['new_pos'] for m in moves if m['piece_id']==piece_id]

		outfile = StringIO() if mode == 'ansi' else sys.stdout
		outfile.write('    ')
		outfile.write('-' * 25)
		outfile.write('\n')

		for i in range(7,-1,-1):
			outfile.write(' {} | '.format(i+1))
			for j in range(7,-1,-1):
				piece = TTCEnv.ids_to_pieces[board[i,j]]
				figure = uniDict[piece[0]]

				# check moves + piece
				if board[i,j] == piece_id:
					outfile.write('<{}>'.format(figure))
				elif moves_pos and any(np.equal(moves_pos,[i,j]).all(1)):
					if piece == '.':
						if piece_id == TTCEnv.CASTLE_MOVE_ID:
							outfile.write('0-0')
						else:
							outfile.write(' X ')
					else:
						outfile.write('+{}+'.format(figure))
				else:
					outfile.write(' {} '.format(figure))
			outfile.write('|\n')
		outfile.write('    ')
		outfile.write('-' * 25)
		outfile.write('\n      a  b  c  d ')
		outfile.write('\n')
		outfile.write('\n')

		if mode != 'human':
			return outfile

	@staticmethod
	def encode_current_state(state, saved_states):
		board = state['board']
		new_saved_states = copy(saved_states)
		pieces_encoding = { '.': 0, 'p': 1, 'b': 2, 'n': 3, 'r': 4}
		for i in range(8):
			for j in range(8):
				piece_id = board[i][j]
				player = sign(piece_id)
				piece_type = TTCEnv.ids_to_pieces[piece_id][0].lower()
				piece_encode = pieces_encoding[piece_type]
				if piece_encode != 0:
					piece_encode += 3*(1-player)
				# hex encoding
				encoding += hex(piece_encode)[2:]
		if encoding in new_saved_states:
			new_saved_states[encoding] += 1
		else:
			new_saved_states[encoding] = 1
		return new_saved_states

	@staticmethod
	def resign_action():
		return 8**2 * 16 + 3

	@staticmethod
	def has_resigned(action):
		return action == TTCEnv.resign_action()

	#TODO: check if there is a draw condition for tic tac chec
	@staticmethod
	def is_a_draw(state):
		return False

	@staticmethod
	def move_to_actions(move):
		"""
		Encode move into action
		"""
		if move == "resign":
			return TTCEnv.resign_action()
		else:
			piece_id = move['piece_id']
			new_pos = move['new_pos']
			return 16*(abs(piece_id) - 1) + (new_pos[0]*4 + new_pos[1]).item()

	@staticmethod
	def action_to_move(action, player):
		"""
		Decode move from action
		"""
		square = action % 16
		column = square % 4
		row = (square - column) // 8
		piece_id = (action - square) // 64 + 1
		return {
			'piece_id': piece_id * player,
			'new_pos': np.array([int(row), int(column)]),
		}

	@staticmethod
	def next_state(state, move, player):
		"""
		Return the next state given a move
		-------
		(next_state, previous_piece, reward)
		"""
		new_state = copy(state)
		new_state['prev_board'] = copy(state['board'])

		board = copy(new_state['board'])
		new_pos = move['new_pos']
		piece_id = move['piece_id']
		reward = 0

		# find old position
		try:
			old_pos = np.array([x[0] for x in np.where(board == piece_id)])
		except:
			print('piece_id', piece_id)
			print(board)
			raise Exception()
		r, c = old_pos[0], old_pos[1]
		board[r, c] = 0

		# replace new position
		new_pos = np.array(new_pos)
		r, c = new_pos
		prev_piece = board[r, c]
		board[r, c] = piece_id

		# check for pawn reaching the end of the board
		# if pawns reach the end, their direction is inverted
		# TODO: check if this logic breaks when placing a pawn on the back row
		# for the first time
		if TTCEnv.ids_to_pieces[piece_id][0].lower() == 'p':
			if ((new_pos[0] == 4|| new_pos[0] == 0):
				new_state['pawn_direction'][player] *= -1

		new_state['board'] = board
		return new_state, prev_piece, reward

	@staticmethod
	def get_possible_actions(state, player):
		moves = TTCEnv.get_possible_moves(state, player)
		return [TTCEnv.move_to_actions(m) for m in moves]

	@staticmethod
	def get_possible_moves(state, player, attack=False):
		"""
		Returns a list of numpy tuples
		piece_id, position, new_position
		"""
		board = state['board']
		total_moves =  []
		# add all moves that involve placing a piece onto the board
		for piece_id in state.captured:
			moves = get_empty_squares(state, player)
			for move in moves:
				total_moves.append({
				'piece_id': piece_id,
				'new_pos': move,
				'type': 'placement'
				})
		# add all moves that involve moving a piece on the board
		for position, piece_id in np.denumerate(board):
			# for all pieces that are not blank and have player alignment
			if piece_id != 0 and sign(piece_id) == sign(player):
				if piece_type == 'r':
					moves = TTCEnv.rook_actions(state, position, player, attack=attack)
				elif piece_type == 'b':
					moves = TTCEnv.bishop_actions(state, position, player, attack=attack)
				elif piece_type == 'n':
					moves = TTCEnv.knight_actions(state, position, player, attack=attack)
				elif piece_type == 'p':
					moves = TTCEnv.pawn_actions(state, position, player, attack=attack)
				elif piece_type == '.':
					moves = []
					continue
				else:
					raise Exception("ERROR - inexistent piece type ")
			for move in moves:
				total_moves.append({
				'piece_id': piece_id,
				'pos': position,
				'new_pos': move,
				'type': 'move'
				})
			else:
				continue
			return total_moves

	@staticmethod
	def get_empty_squares(state, player):
		"""
		Returns all empty tiles that could have a piece placed on them
		"""
		empty_squares = []
		for index, x in np.ndenumerate(state['board']):
			if x == ".":
				empty_squares.add(index)
		return empty_squares

	@staticmethod
	def rook_moves(state, position, player):
		"""
		ROOK ACTIONS
		------------
		"""
		pos = np.array(position)
		go_to = []

		for i in [-1, +1]:
			step = np.array([i, 0])
			go_to += TTCEnv.iterative_steps(state, player, pos, step, attack=attack)

		for j in [-1, +1]:
			step = np.array([0, j])
			go_to += TTCEnv.iterative_steps(state, player, pos, step, attack=attack)

		return go_to

	@staticmethod
	def bishop_actions(state, position, player, attack=False):
		"""
		BISHOP ACTIONS
		--------------
		"""
		pos = np.array(position)
		go_to = []

		for i in [-1, +1]:
			for j in [-1, +1]:
				step = np.array([i, j])
				go_to += TTCEnv.iterative_steps(state, player, pos, step, attack=attack)
		return go_to

	@staticmethod
	def iterative_steps(state, player, position, step, attack=False):
		"""
		Used to calculate Bishop and Rook moves
		"""
		go_to = []
		pos = np.array(position)
		step = np.array(step)
		k = 1
		while True:
			move = pos + k*step
			if attack:
				add_bool, stop_bool = TTCEnv.attacking_move(state, move, player)
				if add_bool:
					go_to.append(move)
				if stop_bool:
					break
				else:
					k += 1
			else:
				add_bool, stop_bool = TTCEnv.playable_move(state, move, player)
				if add_bool:
					go_to.append(move)
				if stop_bool:
					break
				else:
					k += 1
		return go_to

	@staticmethod
	def knight_actions(state, position, player, attack=False):
		"""
		KNIGHT ACTIONS
		--------------
		"""
		go_to = []
		pos = np.array(position)
		moves = [pos + np.array([v,h]) for v in [-2, +2] for h in [-1, +1]]
		moves += [pos + np.array([v,h]) for v in [-1, +1] for h in [-2, +2]]

		# filter:
		for m in moves:
			if attack:
				add_bool, __ = TTCEnv.attacking_move(state, m, player)
				if add_bool:
					go_to.append(m)
			else:
				add_bool, __ = TTCEnv.playable_move(state, m, player)
				if add_bool:
					go_to.append(m)
		return go_to

	@staticmethod
	def pawn_actions(state, position, player, attack=False):
		"""
		PAWN ACTIONS
		------------
		"""
		board = state['board']
		pos = np.array(position)
		go_to = []

		attack_moves = [
			pos + np.array([1, -1])*player,
			pos + np.array([1, +1])*player,
		]

		step_1 = np.array([1, 0]) * player
		step_2 = np.array([2, 0]) * player

		if attack:
			return [m for m in attack_moves if TTCEnv.pos_is_in_board(m)]

		else:
			# moves only to empty squares
			# no double move on first placement
			if board[pos[0]+1*player, pos[1]] == 0:
				go_to.append(pos + step_1)

			# attacks only opponent's pieces
			for m in reversed(attack_moves):
				if not TTCEnv.pos_is_in_board(m):
					continue
				elif TTCEnv.is_own_piece(board, m, player):
					continue
				elif TTCEnv.is_opponent_piece(board, m, player):
					go_to.append(m)
					attack_moves.pop()
					continue
				elif board[m[0], m[1]] == 0:
					continue
				else:
					raise Exception("ERROR - PAWN ATTACK MOVES")
			return go_to

	@staticmethod
	def playable_move(state, move, player):
		"""
		return squares to which a piece can move
		- empty squares
		- opponent pieces (excluding king)
		=> return [<bool> add_move, <bool> break]
		"""
		board = state['board']
		if not TTCEnv.pos_is_in_board(move):
			return False, True
		elif TTCEnv.is_own_piece(board, move, player):
			return False, True
		elif TTCEnv.is_opponent_piece(board, move, player):
			return True, True
		elif board[move[0], move[1]] == 0: # empty square
			return True, False
		else:
			raise Exception('MOVEMENT ERROR \n{} \n{} \n{}'.format(board, move, player))

	@staticmethod
	def attacking_move(state, move, player):
		"""
		return squares that are attacked or defended
		- empty squares
		- opponent pieces (opponent king is ignored)
		- own pieces
		=> return [<bool> add_move, <bool> break]
		"""
		board = state['board']
		if not TTCEnv.pos_is_in_board(move):
			return False, True
		elif TTCEnv.is_own_piece(board, move, player):
			return True, True
		elif TTCEnv.is_opponent_piece(board, move, player):
			return True, True
		elif board[move[0], move[1]] == 0: # empty square
			return True, False
		else:
			raise Exception('ATTACKING ERROR \n{} \n{} \n{}'.format(board, move, player))

	"""
	- flatten board
	- find move in movelist
	"""
	@staticmethod
	def move_in_list(move, move_list):
		move_list_flat = [TTCEnv.flatten_position(m) for m in move_list]
		move_flat = TTCEnv.flatten_position(move)
		return move_flat in move_list_flat
	@staticmethod
	def flatten_position(position):
		x, y = position[0], position[1]
		return x + y*4
	@staticmethod
	def boardise_position(position):
		x = position % 4
		y = (position - x)//4
		return x, y

	@staticmethod
	def pos_is_in_board(pos):
		return not (pos[0] < 0 or pos[0] > 4 or pos[1] < 0 or pos[1] > 4)

	@staticmethod
	def squares_attacked(state, player):
		opponent_moves = TTCEnv.get_possible_moves(state, -player, attack=True)
		attacked_pos = [m['new_pos'] for m in opponent_moves]
		return attacked_pos

	"""
	Player Pieces
	"""
	@staticmethod
	def is_own_piece(board, position, player):
		return TTCEnv.is_player_piece(board, position, player)
	@staticmethod
	def is_opponent_piece(board, position, player):
		return TTCEnv.is_player_piece(board, position, -player)
	@staticmethod
	def is_player_piece(board, position, player):
		x, y = position
		return  (board[x,y] != 0 and sign(board[x,y]) == player)

	@staticmethod
	def convert_coords(move):
		"""
		TODO: include board context:
		- disambiguation
		- capture of opponent's piece marked with 'x'
		"""
		piece = ChessEnv.ids_to_pieces[move['piece_id']]
		old_pos = move['pos']
		new_pos = move['new_pos']
		alpha = 'abcd'
		if piece[0].lower() != 'p':
			piece = piece[0].upper()
		else:
			piece = ''
		return '{}{}{}-{}{}'.format(piece,
			alpha[old_pos[1]], old_pos[0]+1,
			alpha[new_pos[1]], new_pos[0]+1)
