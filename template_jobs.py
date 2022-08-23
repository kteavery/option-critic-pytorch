from examples import *


def batch_atari():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    games = [
        'AmidarNoFrameskip-v4',
        # 'AlienNoFrameskip-v4',
        # 'DemonAttackNoFrameskip-v4',
        # 'MsPacmanNoFrameskip-v4'
    ]

    algos = [
        option_critic_pixel,
    ]

    params = []

    for game in games:
        for r in range(1):
            for algo in algos:
                params.append([algo, dict(game=game, run=r, remark=algo.__name__)])

    algo, param = params[cf.i]
    algo(**param)
    exit()



if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()

    select_device(0)
    batch_atari()
