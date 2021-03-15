import gym_bridge.analyze as analyze
import numpy as np

def test_IMP():
    """Tests whether difference in score is correctly converted to IMPs."""
    scores = [[0, 10], [0, 40], [0, 80], [0, 120], [0, 160], [0, 210], [0, 260],
              [0, 310], [0, 360], [0, 420], [0, 490], [0, 590], [0, 740],
              [0, 890], [0, 1090], [0, 1290], [0, 1490], [0, 1740], [0, 1990],
              [0,2240], [0, 2490], [0, 2990], [0, 3490], [0, 3990], [0, 1000000],
              [40,50], [200,100], [400,0], [3000,1500], [6000,100], [500000, 0],
              [4500, 1000], [1200,1100], [400,350], [670, 640]]

    imps = np.concatenate([np.arange(25) * -1, 0, 3, 9, 17, 24, 24, 23, 3, 2, 1],
                           axis=None)

    for score, imp in zip(scores, imps):
        assert imp == analyze.IMP_difference(score[0], score[1])
        assert -imp == analyze.IMP_difference(score[1], score[0])
