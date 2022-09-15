import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
from deep_rl import *


def plot_atari():
    plotter = Plotter()
    games = [
        'AmidarNoFrameskip-v4',
    ]

    patterns = [
        'remark_option_critic',
    ]

    labels = [
        'OC',
    ]

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=100,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TRAIN,
                       root='./data/benchmark/atari',
                       interpolation=0,
                       window=100,
                       )

    # plt.show()
    plt.tight_layout()
    plt.savefig('images/Amidar.png', bbox_inches='tight')


if __name__ == '__main__':
    mkdir('images')
    plot_atari()