import numpy as np
import os
import hashlib


def diverse_density():
    pass

def most_visited(dirname):
    # first visitation frequency
    # count up the number of times a state is visited in one game
    # by matching up the state jsons for each of the (10?) runs
    # https://ai.stackexchange.com/questions/27500/why-is-it-that-the-state-visitation-frequency-equals-the-sum-of-state-visitation
    # dirname points to directory of jsons from a game

    onlyfiles = [f for f in os.listdir(dirname)
                if os.path.isfile(os.path.join(dirname,f)) ]

    files = {}
    for filename in onlyfiles:
        filehash = hashlib.md5(open(os.path.join(dirname, filename), 'rb')
                                .read()).hexdigest()
        try:
            files[filehash].append(filename)
        except KeyError:
            files[filehash] = [filename]

    counts = {key: len(value) for key, value in files.items()}
    print(counts)
