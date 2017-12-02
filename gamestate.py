import copy


class GameState(object):
    def __init__(self):
        # Begin with an empty game board
        self.board = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']]

    # GameState needs to be hashable so that it can be used as a unique graph
    # node in NetworkX
    def __key(self):
        return self.__str__()

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):

        output = ''
        for row in range(16):
            for col in range(16):
                contents = self.board[row][col]
                if col < 15:
                    output += '{}'.format(contents)
                else:
                    output += '{}\n'.format(contents)

        output = output.replace(' ', '.')

        return output

    def turn(self):
        """
        Returns the player whose turn it is: 'X' or 'O'
        """
        num_X = 0
        num_O = 0
        for row in range(16):
            for col in range(16):
                if self.board[row][col] == 'X':
                    num_X += 1
                elif self.board[row][col] == 'O':
                    num_O += 1
        if num_X == num_O:
            return 'X'
        else:
            return 'O'

    def move(self, row, col):

        #print('Move: {} moves to ({}, {})'.format(self.turn(), row, col))
        self.board[row][col] = self.turn()
        #print('{}'.format(self))

    def legal_moves(self):

        # Check if terminal state
        if self.winner() is not None:
            return []

        possible_moves = []
        for row in range(16):
            for col in range(16):
                if self.board[row][col] == ' ':
                    possible_moves.append((row, col))
        return possible_moves

    def transition_function(self, row, col):

        # Verify that the specified action is legal
        assert (row, col) in self.legal_moves()

        # First, make a copy of the current state
        new_state = copy.deepcopy(self)

        # Then, apply the action to produce the new state
        new_state.move(row, col)

        return new_state

    def winner(self):

        for player in ['X', 'O']:
            # case -
            for i in range(16):
                for j in range(12):
                    option = [ self.board[i][j+0],
                               self.board[i][j+1],
                               self.board[i][j+2],
                               self.board[i][j+3],
                               self.board[i][j+4]]

                    if all(marker == player for marker in option):
                        return player

            # case |
            for i in range(16):
                for j in range(12):
                    option = [self.board[j+0][i],
                              self.board[j+1][i],
                              self.board[j+2][i],
                              self.board[j+3][i],
                              self.board[j+4][i]]

                    if all(marker == player for marker in option):
                        return player

            # case \
            for i in range(12):
                for j in range(12):
                    option = [ self.board[i+0][j+0],
                               self.board[i+1][j+1],
                               self.board[i+2][j+2],
                               self.board[i+3][j+3],
                               self.board[i+4][j+4]]

                    if all(marker == player for marker in option):
                        return player

            # case /
            for i in range(12):
                for j in range(12):
                    option = [self.board[i+4][j+0],
                              self.board[i+3][j+1],
                              self.board[i+2][j+2],
                              self.board[i+1][j+3],
                              self.board[i+0][j+4]]

                    if all(marker == player for marker in option):
                        return player

        # Check for ties, defined as a board arrangement in which there are no
        # open board positions left and there are no winners (note that the
        # tie is not being detected ahead of time, as could potentially be
        # done)
        accum = 0
        for row in range(16):
            for col in range(16):
                if self.board[row][col] == ' ':
                    accum += 1
        if accum == 0:
            return 'Tie'

        return None

