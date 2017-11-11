import numpy as np
from keras.models import load_model

def predict(state, empty=0, me=1, you=2, model_name='gomoku.hdf5'):
    if min(state) == 0 and max(state) == 0: return [(8, 8)]
    x = np.zeros((1, 17, 17, 1))
    for i in range(17):
        for j in range(17):
            if state[i*17 + j] == me:
                x[0, i, j, 0] = 1
            elif state[i*17 + j] == you:
                x[0, i, j, 0] = -1

    model = load_model(model_name)
    y = model.predict(x)
    y = np.reshape(y, (17, 17))

    ret = []
    for i in range(17):
        for j in range(17):
            if y[i, j] > 0 and x[0, i, j, 0] == empty:
                ret.append(((i, j), y[i, j]))

    ret.sort(key=lambda t: t[1], reverse=True)
    return [t[0] for t in ret]

if __name__ == '__main__':
    state = [0 for _ in range(17*17)]
    print(predict(state))

    state[8*17+8] = 2
    print(predict(state))