from flask import Flask, request
import numpy as np
from keras.models import load_model
import pickle
from gamestate import GameState
from policies import MCTSPolicy

app = Flask(__name__)

def predict(state, empty=0, me=1, you=-1, model_name='gomoku.hdf5'):
    x = np.zeros((1, 16, 16, 1))
    for i in range(16):
        for j in range(16):
            if state[i*16 + j] == me:
                x[0, i, j, 0] = 1
            elif state[i*16 + j] == you:
                x[0, i, j, 0] = -1

    model = load_model(model_name)
    y = model.predict(x)
    y = np.reshape(y, (16, 16))

    ret = []
    for i in range(16):
        for j in range(16):
            if y[i, j] >= 0.01 and x[0, i, j, 0] == empty:
                ret.append(((i, j), y[i, j]))
    if len(ret) < 5:
        for i in range(16):
            for j in range(16):
                if x[0, i, j, 0] == empty:
                    ret.append(((i, j), y[i, j]))

    ret.sort(key=lambda t: t[1], reverse=True)
    if len(ret) > 10: ret = ret[:10]
    return [t[0] for t in ret]

@app.route('/', methods=['POST',])
def f():
  if request.method == 'POST':
    data = request.get_json()
    print(data)
    state = []
    me_moves = []
    you_moves = []
    your_move = None

    for i in range(256):
      state.append(int(data['Inputs']['input1'][0][str(i+1)]))
      if int(data['Inputs']['input1'][0][str(i+1)]) == 1:
        me_moves.append((i//16, i%16))
      if int(data['Inputs']['input1'][0][str(i+1)]) == -1:
        you_moves.append((i//16, i%16))

    if len(me_moves) == 0:
      game = GameState()
      origin = None
      if len(you_moves) > 0: your_move = you_moves[0]
      if len(you_moves) > 1: return 'Error: you_moves > 1'
    else:
      game = pickle.load(open('game.pkl', 'rb'))
      origin = pickle.load(open('origin.pkl', 'rb'))
      pre_state = pickle.load(open('state.pkl', 'rb'))
      your_move = None
      for i in range(16):
        for j in range(16):
          if state[i*16+j] != pre_state[i*16+j] and state[i*16+j] == -1:
            if your_move != None: return 'Error: your_move not none'
            your_move = (i, j)

    if your_move != None:
      print('Moved', your_move)
      game.move(*your_move)

    player_icon = None
    if len(me_moves) == len(you_moves):
      player_icon = 'X'
    elif len(me_moves) == (len(you_moves)-1):
      player_icon = 'O'
    else:
      return 'Error: no player icon'

    recom_moves = predict(state, model_name='gomoku_nas.hdf5')
    print(recom_moves)
    print(origin)
    print(game)

    tar, tree = MCTSPolicy(player=player_icon).move(game, recom_moves, 100, origin)
    game.move(*tar)
    print(game)
    pickle.dump(game, open('game.pkl', 'wb'))
    pickle.dump(tree, open('origin.pkl', 'wb'))
    state[tar[0]*16+tar[1]] = 1
    pickle.dump(state, open('state.pkl', 'wb'))
    return str(tar[0]*16+tar[1])
