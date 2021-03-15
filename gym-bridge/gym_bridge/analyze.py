import numpy as np
from gym_bridge.state import Contract

# Constants used throughout these functions to transform raw into serialize
CARDMAP = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7,
           'T':8, 'J':9, 'Q':10, 'K':11, 'A':12}
REV_CARDMAP = { 0:'2', 1:'3', 2:'4', 3:'5', 4:'6', 5:'7', 6:'8',
                7:'9', 8:'T', 9:'J', 10:'Q', 11:'K', 12:'A' }
BIDMAP = { 'C':0, 'c':0, 'D':1, 'd':1, 'H':2, 'h':2,
           'S':3, 's':3, 'N':4, 'n':4, 'NT':4, 'nt':4 }
REV_BIDMAP = { 0:'C', 1:'D', 2:'H', 3:'S', 4:'N'}

def serialize_bids(history):
    '''
    Only used in tests/test_moves to convert from user-interpretable input.

    Encodes the bidding history to a 318-dimensional vector,
    where a 1 in the i-th entry denotes that the i-th bid in the possible
    maximum bidding sequence is called.
    See Competitive Bridge Bidding with Deep Neural Networks (Rong et al.)
    for details.

    args:
        history(str): bidding sequence

    returns:
        serialized(list(int)): 318-dimensional vector representing
                               bidding history
    '''
    serialized = np.zeros(318)
    prev_c_idx = 0 # Previous contract bid index
    curr_p_idx = 0 # Current pass index
    for bid in history:
        if len(bid) > 1: #contract bid
            curr_p_idx = 1 # Reset 'pass' offset
            rank = bid[0]
            suit = BIDMAP[bid[1]]
            # 9 bits: first bit is the contract bid, subsequent 8 are
            # p-p-d-p-p-r-p-p, with a 1 indicating bid was made.
            idx = 3 + (int(rank)-1)*45+suit*9 # offset 3: initial passes
            serialized[idx] = 1
            prev_c_idx = idx # Save this index to enter p/d/r 'bids'
        else: # pass/double/redouble bid
            if bid in 'pP': # pass
                if curr_p_idx > 0 and curr_p_idx % 3 == 0: # 3 subsequent passes (or 4 initally)
                    return serialized
                serialized[prev_c_idx + curr_p_idx] = 1
                curr_p_idx += 1
            elif bid in 'dD': # double
                serialized[prev_c_idx + 3] = 1
                curr_p_idx = 4
            else: # redouble
                serialized[prev_c_idx + 6] = 1
                curr_p_idx = 7
    return serialized

def deserialize_bids(X):
    '''
    Deserializes the bitstring X representing the bidding history. Useful for
    user-interpretable information printing.

    args:
        X(list(int)): 318-dimensional vector representing the bidding
                      history
    returns:
        bids(list(str)): List containing the bids that were made
    '''
    bids = []
    for i, b in enumerate(X):
        if b == 0: # No bid made
            continue
        else: # A bid was made
            if i < 3: # First three entries refer to initial passes
                bids.append('p')
            else: # "real" bidding starts now
                if i % 3 != 0: # Contract bids / doubles / redoubles % 3 == 0
                    bids.append('p')
                else:
                    if (i-3) % 9 == 0: # contract bid
                        suit = int((i-3)/9)%5
                        rank = int((i-3)/45)
                        bids.append("{}{}".format(str(rank+1),
                                                  REV_BIDMAP[suit]))
                    if (i-6) % 9 == 0: # double
                        bids.append('d')
                    if i % 9 == 0: # redouble
                        bids.append('r')
    return bids

def serialize_hands(pbn_hands):
    '''
    Converts pbn hands into a ((4,52)) numpy array.

    args:
        - pbn_hands(list(str)): spades.hearts.diamond.clubs x4

    returns:
        - encoded_hands(list(list(int)): one-hot encoding of all 4 hands.
    '''
    encoded_hands = np.zeros((4,52))
    hands = pbn_hands.split(' ')
    for i, hand in enumerate(hands):
        suits = hand.split('.')
        for suit_idx in range(len(suits)):
            for card in suits[suit_idx]:
                encoded_hands[i, CARDMAP[card]*4+3-suit_idx] = 1
    return encoded_hands

def deserialize_hands(encoded_hands):
    '''
    Deserializes a hand into expected format: needs to be string formatted
    spades.hearts.diamonds.clubs, with player hands seperated by a space,
    starting with N.
    E.G:'234.234.234.2345 678.678.678.6789 TJQ.TJQ.TJQ.TJQK 59KA.59KA.59KA.A'

    args:
        - encoded_hands(list(list(int))): 4x52 one-hot encodings of hands

    returns:
        - hands(str): See above.
    '''
    deserialized_hands = []
    for hand in encoded_hands:
        deserialized_hand = deserialize_hand(hand)
        deserialized_hands.append(deserialized_hand)

    hands = ' '.join(['.'.join([''.join(suit_hand) for suit_hand in single_hand])
                      for single_hand in deserialized_hands])
    return hands

def deserialize_hand(encoded_hand):
    "Deserializes a single encoded hand into s.h.d.c"
    deserialized_hand = [[] for i in range(4)]
    for idx in np.nonzero(encoded_hand)[0]:
        deserialized_hand[(3-idx)%4].append(REV_CARDMAP[idx//4])
    return deserialized_hand

def score(contract, declarer, vulnerability, tricks_13_taken):
    '''
    Computes the raw score given the contract, vulnerability and tricks taken.

    args:
        - state(State): contains state.contract, state.vulnerability
        - tricks_13_taken(int): number of tricks (of 13) taken by the declarer.

    returns:
        - score(int): raw score (unconverted to IMP).
    '''
    if vulnerability[declarer % 2] == 1:
        vulnerable = True
    else:
        vulnerable = False

    tricks_taken = tricks_13_taken - 6 # Tricks out of 7 is used in scoring.
    contract_suit = contract.bid % 5
    contract_rank = contract.bid // 5 + 1 # int {1,...,7}

    if tricks_taken >= contract_rank: # Contract was made
        SUIT_SCORES = [20,20,30,30,30]  # Scores per trick made: C,D,H,S,No turmp
        score = SUIT_SCORES[contract_suit] * contract_rank # First count contract tricks
        if contract_suit == 4: # First trick in NT is worth 40, not 30
            score += 10
        score *= np.power(2, (contract.doubled + contract.redoubled))

        if score < 100: # Bonusscores: partscore
            score += 50
        elif score >= 100: # gamescore
            score = score + 500 if vulnerable else score + 300
        if contract_rank == 6: # small slam
            score = score + 750 if vulnerable else score + 500
        elif contract_rank == 7: # grand slam
            score = score + 1500 if vulnerable else score + 1000
        score += 50 * (contract.doubled + contract.redoubled)

        if contract.redoubled: # Overtricks
            score += (tricks_taken - contract_rank) * (200 + 200 * vulnerable)
        elif contract.doubled: # Overtricks
            score += (tricks_taken - contract_rank) * (100 + 100 * vulnerable)
        else: # Overtricks
            score += (tricks_taken - contract_rank) * SUIT_SCORES[contract_suit]

    elif tricks_taken < contract_rank: # Defeated
        undertricks = contract_rank - tricks_taken
        if contract.redoubled:
            if vulnerable:
                score = undertricks * 600 - 200
            else:
                if undertricks < 3:
                    score = undertricks * 600 - (200 + 200 * undertricks)
                else:
                    score = undertricks * 600 - 800
        elif contract.doubled:
            if vulnerable:
                score = undertricks * 300 - 100
            else:
                if undertricks < 3:
                    score = undertricks * 300 - (100 + 100 * undertricks)
                else:
                    score = undertricks * 300 - 400
        else: # Not doubled nor redoubled
            score = undertricks * 100 if vulnerable else undertricks * 50
        score *= -1 # Negative score because defeataed

    return score

def IMP_difference(score_1, score_2):
    '''
    Converts the difference in score between the two equal deals into IMP.
    Positive when N/S did better than S/E, negative otherwise.

    args:
        score_1(int): score obtained with players in original pos. (1st game).
        score_2(int): score obtained with players rotated 1 clockwise (2nd).

    returns:
        - IMP(int): International Match Point difference between the scores.
    '''
    diff = np.absolute(score_1 - score_2)
    IMP_bins = [0,15,45,85,125,165,215,265,315,365,425,495,595,745,895,1095,
                1295,1495,1745,1995,2245,2495,2995,3495,3995,100000000000]
    IMP = np.digitize(diff, IMP_bins) - 1
    if score_2 > score_1:
        IMP *= -1
    return IMP


def compute_max_score(dds_table, all_declarers, vulnerability):
    '''
    Computes the maximally obtainable score for a partnership.

    args:
        - dds_table(list(list(int))): 4x5 declarer x suit DDS
        - all_declarers(list(int)): Contains the declarer for each suit
                                    for a specific partnership
        - vulnerability(list(int)): 2-bit vector. First bit is NS vuln,
                                    second is EW vuln.

    returns:
        - max_score(int): The maximally obtainable score.
        - max_entry(int): the index in the dds_table for the contract
                          corresponding to this score.
    '''
    max_score = -1000000
    for i, d in enumerate(all_declarers):
        dds_entry = dds_table[d,i]
        contract = (dds_entry - 7) * 5 + i
        while contract < 0:
            contract += 5
        curr_score = score(Contract(contract, False, False), d, vulnerability, dds_entry)
        if curr_score > max_score:
            max_entry = (d,i)
            max_score = curr_score
    return max_score, max_entry


def check_valid_bid_sequence(bidding_history):
    '''
    Tests whether bidding_history(list(str)) is increasing and valid.
    Only used for testing in tests/test_played_histories. Not used within code.

    args:
        - bidding_history(list(str)): User-interpretable ['1C', 'p', '1H', '6S'..]

    returns:
        - bool: whether the history is valid or not.
    '''
    bid_idx_counter = 0
    pass_counter = 0
    doubled = False
    redoubled = False
    for bid in bidding_history:
        if len(bid) > 1: #contract bid
            if bid_idx_counter > 0 and pass_counter == 3:
                return False # Can't contract after three passes, except initial
            rank = bid[0]
            suit = BIDMAP[bid[1]]
            temp_bid_idx_counter = 3 + (int(rank)-1)*45+suit*9 # offset 3
            if temp_bid_idx_counter < bid_idx_counter:
                print("faulty: ", bidding_history)
                return False
            bid_idx_counter = temp_bid_idx_counter
            pass_counter = 0 # Reset, since contract bid was made
            doubled = False
            redoubled = False
        else: # pass/double/redouble bid
            if bid in 'pP': # pass
                if bid_idx_counter > 0 and pass_counter == 3:
                    print("faulty: ", bidding_history) # 4 consecutive passes
                    return False
                pass_counter += 1
            elif bid in 'dD': # double
                if doubled: # Can't double a double
                    print("faulty: ", bidding_history)
                    return False
                doubled = True
                pass_counter = 0 # Reset pass counter
            else: # redouble
                if redoubled or not doubled:
                    print("faulty: ", bidding_history)
                    return False
                redoubled = True
                pass_counter = 0 # Reset pass
    return True
