import numpy as np
from keras.models import load_model

def predict(state, empty=0, me=1, you=2, model_name='gomoku.hdf5'):
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
        print('')

    ret.sort(key=lambda t: t[1], reverse=True)
    return [t[0] for t in ret]

if __name__ == '__main__':
    state = [0 for _ in range(16*16)]
    print(predict(state))

    state[8*16+8] = 2
    print(predict(state))

    state[7*16+8] = 1
    state[9*16+9] = 2
    print(predict(state))