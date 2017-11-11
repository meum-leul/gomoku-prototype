from glob import glob
import numpy as np
from multiprocessing import Pool


def findGame(line):
    line = line.strip()
    for tok in line.split(","):
        # Nearly always start with middle point
        if tok.find("88") == 0:
            return tok
    return None


def checkGame(line):
    if len(line) == 0: return False
    if len(line) % 2 == 1: return False
    for c in line:
        if not c in "1234567890ABCDEF":
            return False
    return True


def numpyGame(line):
    x = np.zeros(shape=(1, 17, 17), dtype="int8")
    y = np.zeros(shape=(1, 17, 17), dtype="int8")
    y[0,8,8] = 1
    for i in range(len(line) // 2 - 1):
        currBoard = -np.copy(x[i,:,:] + y[i,:,:])
        nextBoard = np.zeros(shape=(17, 17), dtype="int8")
        nextBoard[int(line[i*2+2], 16), int(line[i*2+3], 16)] = 1
        x = np.concatenate((x, currBoard[np.newaxis,:,:]))
        y = np.concatenate((y, nextBoard[np.newaxis,:,:]))
    return (x, y)


def augmentData(arr):
    ret = None
    for i in range(4):
        if type(ret) == type(None): ret = np.copy(arr)
        else: ret = np.concatenate((ret, arr))
        ret = np.concatenate((ret, np.flip(arr, axis=1)))
        ret = np.concatenate((ret, np.flip(arr, axis=2)))
        ret = np.concatenate((ret, np.flip(np.flip(arr, axis=2), axis=1)))
        arr = np.rot90(arr, axes=(1, 2))
    return ret


def proc(tok):
    x, y = numpyGame(tok)
    x, y = augmentData(x), augmentData(y)
    return x, y


if __name__ == '__main__':
    '''
    # Preprocess with BDT (Raw data from RenjuNet dataset), output: cleaned.txt
    total = []
    for name in glob("dataset/*"):
        print("Processing [ %s ] ..." % name)
        with open(name, "r", encoding="UTF-8") as f:
            for line in f.readlines():
                ret = findGame(line)
                if ret: total.append(ret)
    with open("cleaned.txt", "w") as f:
        f.write(','.join(total))
    '''

    with open("cleaned.txt", "r") as f:
        total = f.readline().split(",")
        print("Checking data ...")
        for tok in range(len([tok for tok in total if not checkGame(tok)])):
            print("Error: wrong data [ %s ]" % tok)
        total = [tok for tok in total if checkGame(tok)]
        print("Total %d data ..." % len(total))

        # Multiprocess version
        for idx in range(100000, len(total), 10000):
            if idx + 10000 >= len(total):
                total_xy = Pool(8).map(proc, total[idx:])
            else:
                total_xy = Pool(8).map(proc, total[idx:idx+10000])
            total_x = [tup[0] for tup in total_xy]
            total_y = [tup[1] for tup in total_xy]
            np.save("x_%d.npy" % idx, np.concatenate(total_x))
            np.save("y_%d.npy" % idx, np.concatenate(total_y))

        '''
        # Single process version
        total_x, total_y = numpyGame(total[0])
        for i in range(1, len(total)):
            if (i * 100 // len(total)) != ((i-1) * 100 // len(total)):
                print("%d %% done" % (i * 100 // len(total)))
            if i % 10000 == 0:
                np.save("x.npy", total_x)
                np.save("y.npy", total_y)
                exit(1)
            x, y = numpyGame(total[i])
            x, y = augmentData(x), augmentData(y)
            total_x, total_y = np.concatenate((total_x, x)), np.concatenate((total_y, y))
        '''
