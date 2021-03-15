from gym_bridge.state import Contract, State
import gym_bridge.analyze as analyze

def test_scoring():
    """Tests whether raw score is computed correctly for various contracts etc."""
    contracts = [Contract(0, False, False),
                 Contract(5, False, False),
                 Contract(15, False, False),
                 Contract(15, True, False),
                 Contract(25, True, True),
                 Contract(30, True, False),
                 Contract(24, True, True),
                 Contract(34, True, False),
                 Contract(21, False, False),
                 Contract(17, True, True)]
    vulns = [[0,0], [0,0], [1,0], [1,0], [1,1], [1,1], [1,0], [1,0], [0,0], [0,1]]
    declarers = [0,1,2,3,0,1,2,3,0,1]
    tricks_taken = [7,8,10,11,13,13,12,13,12,12, 6,6,6,8,6,10,8,8,8,6]

    scores = [70, 90, 130, 610, 2230, 2330, 1640, 1790, 420, 1880,
              -50, -100, -400, -300, -3400, -800, -1600, -1100, -150, -2200]
    state = State()
    for i in range(len(tricks_taken)):
        state.declarer = declarers[i%10]
        state.contract = contracts[i%10]
        state.vulnerability = vulns[i%10]
        tricks = tricks_taken[i]
        score = analyze.score(state.contract, state.declarer,
                              state.vulnerability, tricks)
        assert score == scores[i]
