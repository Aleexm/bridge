class Bid():
    def __init__(self, player, bid, suit, rank):
        self.player = player
        self.bid = bid
        self.suit = suit
        self.rank = rank

    def __repr__(self):
        return "{}: {}".format(self.player, self.bid.__repr__())
