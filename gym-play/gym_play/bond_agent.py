from gym_bridge.agents import Agent
import numpy as np

class BondAgent(Agent):
    "Opposition playing agent as described by the Bridge Bond."
    def __init__(self, action_space, handservation_space):
        super().__init__(action_space, handservation_space)

    def __repr__(self):
        return "BridgeBond Agent"

    def act(self, hands, leader_pos, own_pos, state, declarer):
        hand = hands[own_pos]
        hcps = self._get_hcp_per_suit(hand)
        suit_cards = self._get_cards_per_suit(hand)

        if leader_pos == own_pos:
            a = self._play_first(hcps, suit_cards, state)
        elif (leader_pos+1)%4 == own_pos:
            a = self._play_second(hands, state, suit_cards, own_pos, declarer)
        elif (leader_pos+2)%4 == own_pos:
            a = self._play_third(hands, state, suit_cards, own_pos, declarer)
        else:
            a = self._play_last(state, suit_cards, own_pos)
        return a

    def _play_first(self, hcps, suit_cards, state):
        "Honors > Ontkennen > 5+:5, 3-4:3, 1-2:1"
        if np.sum(hcps) == 0:
            suit = self._no_hcp(suit_cards, state)
        else:
            suit = np.argmax(hcps)
        honors = self._highest_honors(suit_cards[suit])
        if honors > -1: # Highest honors
            return self._parse_action(suit_cards[suit][honors], suit)
        elif len(suit_cards[suit]) >= 5:
            play = 4 # De vijfde
        elif len(suit_cards[suit]) >= 3:
            play = 2 # De derde
        else:
            play = 0 # De hoogste
        # 9+ Ontkent erboven
        while play > 0 and suit_cards[suit][play] >= 9 \
                       and suit_cards[suit][play-1] == suit_cards[suit][play]+1:
            play -= 1
        return self._parse_action(suit_cards[suit][play], suit)

    def _play_second(self, hands, state, suit_cards, own_pos, declarer):
        suit_lead = state.cards_this_trick[state.first_player] % 4
        holds_highest_card = self._highest_card(hands, suit_lead, suit_cards,
                                                state.cards_this_trick[(own_pos-1)],
                                                own_pos, declarer)

        if holds_highest_card is not None: # Claim trick in leader suit
            return self._parse_action(holds_highest_card, suit_lead)
        else:
            return self._play_remainder_against(state, suit_cards, suit_lead)

    def _play_third(self, hands, state, suit_cards, own_pos, declarer):
        "Claim if possible, conventions oth."
        best_card = state.cards_this_trick[state.trick_winner()]
        suit_lead = state.cards_this_trick[state.first_player] % 4
        if own_pos == (declarer+1)%4: # Dummy is last to play
            if state.trick_winner() != (own_pos-2) % 4: # Currently losing
                return self._play_before_dummy(hands, state, suit_cards, suit_lead,
                                               own_pos, best_card)
            else: # Winning
                return self._play_passive_before_dummy(hands, suit_cards, suit_lead,
                                                       state, own_pos, best_card)

        else: # Improvement; Beat dummy?
            if len(suit_cards[suit_lead]) == 0:
                if state.trump == 4 or len(suit_cards[state.trump]) == 0:
                    return self._follow_play_along_convention(suit_cards,
                                                              state, suit_lead)
                else:
                    return self._parse_action(suit_cards[state.trump][-1],
                                                         state.trump)
            else:
                if state.cards_this_trick[(own_pos-2)%4] // 4 <= 9 and \
                state.cards_this_trick[(own_pos-1)%4] // 4 <= 9:
                    return self._parse_action(suit_cards[suit_lead][0], suit_lead)
                else: # even: high->low, odd: low->high
                    if len(suit_cards[suit_lead]) % 2 == 0: #even play high->low
                        return self._parse_action(suit_cards[suit_lead][-2],
                                                  suit_lead) # Play one-to-lowest to signal even
                    else:
                        return self._parse_action(suit_cards[suit_lead][-1],
                                                  suit_lead)


    def _play_last(self, state, suit_cards, own_pos):
        "Claim when possible, follow conventions otherwise if not"
        best_card = state.cards_this_trick[state.trick_winner()]
        suit_lead = state.cards_this_trick[state.first_player] % 4
        if state.trick_winner() != (own_pos+2)%4: # We are not currently winning:
            for c in reversed(suit_cards[suit_lead]): # Claim trick with leading suit
                if c-2 > best_card//4 and (best_card%4 != state.trump \
                                           or suit_lead == state.trump):
                    return self._parse_action(c, suit_lead)
            if state.trump < 4 and len(suit_cards[suit_lead%4]) == 0:
                if len(suit_cards[state.trump]) > 0:
                    if best_card % 4 == state.trump:
                        for c in reversed(suit_cards[state.trump]): # Claim trick with leading suit
                            if c-2 > best_card//4:
                                return self._parse_action(c, state.trump)
                    else:
                        return self._parse_action(suit_cards[state.trump][-1], state.trump)
        return self._follow_convention(suit_cards, state, suit_lead)

    def _no_hcp(self, suit_cards, state):
        for suit in range(4):
            if suit == state.trump:
                continue
            if len(suit_cards[suit]) > 0:
                return suit
        return state.trump

    def _play_before_dummy(self, hands, state, suit_cards, suit_lead, own_pos, best_card):
        "Claims with leading suit or trump when possible, follows convention oth."
        dummy_hand = self._get_cards_per_suit(hands[(own_pos+1)%4])
        for c in reversed(suit_cards[suit_lead]): # Claim trick with leading suit
            if c-2 > best_card//4 \
            and (best_card%4 != state.trump or suit_lead == state.trump) \
            and not any(c2 > c for c2 in dummy_hand[suit_lead]): # Dummy not better
                return self._parse_action(c, suit_lead)
        if state.trump < 4 and len(suit_cards[suit_lead]) == 0: # Overtroeven
            if len(suit_cards[state.trump]) > 0:
                if len(dummy_hand[suit_lead]) == 0:
                    if len(dummy_hand[state.trump]) > 0:
                        for tc in reversed(suit_cards[state.trump]):
                            if tc > dummy_hand[state.trump][0]: # Claim the trick
                                return self._parse_action(tc, state.trump)
                        return self._parse_action(suit_cards[state.trump][-1],
                                                     state.trump)
                    else:
                        return self._follow_convention(suit_cards, state, suit_lead)
        return self._follow_convention(suit_cards, state, suit_lead)

    def _play_passive_before_dummy(self, hands, suit_cards, suit_lead, state, own_pos, best_card):
        "Only claim when necessary. (i.e. dummy holds better card than parter's)"
        dummy_hand = self._get_cards_per_suit(hands[(own_pos+1)%4])
        if len(dummy_hand[suit_lead]) > 0 and \
        dummy_hand[suit_lead][0]-2 > best_card // 4: # Can claim with suit_lead
            for c in reversed(suit_cards[suit_lead]):
                if c > dummy_hand[suit_lead][0]:
                    return self._parse_action(c, suit_lead)
            if len(suit_cards[suit_lead]) == 0 and state.trump < 4 and \
            len(suit_cards[state.trump]) > 0: # Can claim with trump
                return self._parse_action(suit_cards[state.trump][-1], state.trump)
        elif len(dummy_hand[suit_lead]) == 0 and state.trump < 4: # Highest troef
            if len(suit_cards[suit_lead]) == 0:
                if len(dummy_hand[state.trump]) > 0:
                    for c in reversed(suit_cards[state.trump]):
                        if c > dummy_hand[state.trump][0]:
                            return self._parse_action(c, state.trump)
                else:
                    if len(suit_cards[state.trump]) > 0:
                        return self._parse_action(suit_cards[state.trump][0], state.trump)
        if len(suit_cards[suit_lead]) > 0:
            return self._parse_action(suit_cards[suit_lead][-1], suit_lead)
        else:
            return self._follow_play_along_convention(suit_cards, state, suit_lead)

    def _play_remainder_against(self, state, suit_cards, suit_lead):
        "Troef in when possible, signal convention otherwise"
        if state.trump < 4 and len(suit_cards[suit_lead%4]) == 0: # Overtroeven
            if len(suit_cards[state.trump]) > 0:
                return self._parse_action(suit_cards[state.trump][-1],
                                                     state.trump)
        return self._follow_convention(suit_cards, state, suit_lead)

    def _follow_convention(self, suit_cards, state, suit_lead):
        if len(suit_cards[suit_lead]) > 0:
            return self._follow_suit_convention(suit_cards, suit_lead)
        else:
            return self._follow_play_along_convention(suit_cards, state,
                                                      suit_lead)

    def _follow_play_along_convention(self, suit_cards, state, suit_lead):
        "Play lowest card from longest remaining suit"
        assert len(suit_cards[suit_lead]) == 0
        length = 0
        for i in range(4):
            if i == state.trump:
                continue
            if len(suit_cards[i]) > length:
                longest_suit = i
                length = len(suit_cards[i])
        if length > 0:
            return self._parse_action(suit_cards[longest_suit][-1],
                                                 longest_suit)
        else:
            return self._parse_action(suit_cards[state.trump][-1], state.trump)

    def _follow_suit_convention(self, suit_cards, suit_lead):
        "Even = High->Low, Odd = Low->High. Ignore if Q+. Lowest of 2 honors"
        if len(suit_cards[suit_lead]) % 2 == 0: #even play high->low
            play = -2 # Play one-to-lowest to signal even
            if len(suit_cards[suit_lead]) == 2:
                play = 0
        elif len(suit_cards[suit_lead]) % 2 == 1: #odd: play low->high
            play = -1
        if len(suit_cards[suit_lead]) >= 2 and \
           suit_cards[suit_lead][play] >= 10 and \
           suit_cards[suit_lead][play] == suit_cards[suit_lead][play+1] + 1:
            play -= 1 # Play lowest of 2 honors
        if len(suit_cards[suit_lead]) == 2:
            if suit_cards[suit_lead][0] >= 12: # Don't signal Q+
                play = -1 # Just play lowest
        return self._parse_action(suit_cards[suit_lead][play], suit_lead)

    def _highest_card(self, hands, suit, suit_cards, lead_card, own_pos, declarer):
        "Returns highest card if this player holds it (accounting for dummy), None otherwise"
        best_card = None
        for c in reversed(suit_cards[suit]):
            if c-2 < lead_card // 4: # Worse
                 continue
            to_stop = False
            for player in range(4):
                if (own_pos-1)%4 == (declarer+2)%4 and player == (declarer+2)%4:
                    continue
                for pc in [x//4 for x in np.nonzero(hands[player])[0] \
                           if x%4==suit]:
                    if not to_stop:
                        if pc > c-2:
                            best_card = None
                            to_stop = True
                            break
                        best_card = c
        return best_card

    def _parse_action(self, card, suit):
        return (card-2) * 4 + suit

    def _get_hcp_per_suit(self, hand):
        "Ascending: [c, d, h, s]"
        hcp_per_suit = np.zeros([4,1])
        for c in np.nonzero(hand)[0]:
            if c >= 36: # High card
                hcp_per_suit[c%4] += (c-32) // 4
        return hcp_per_suit

    def _get_cards_per_suit(self, hand):
        "Sorted high to low, i.e. [[14, 2], [5, 3, 2], ...]"
        cards_per_suit = list()
        for i in range(4):
            current_suit = list()
            for j in reversed(range(13)):
                idx = 4*j+i
                if hand[idx] == 1:
                    current_suit.append(idx//4+2)
            cards_per_suit.append(current_suit)
        return cards_per_suit

    def _highest_honors(self, suit_cards):
        "Checks whether suit contains honors sequence, and returns highest if so"
        prev = 100
        for card_idx, card in enumerate(suit_cards):
            if prev < 10:
                break
            if card == prev-1:
                return card_idx-1
            prev = card
        return -1
