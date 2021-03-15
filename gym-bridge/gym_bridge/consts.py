from enum import Enum
class Bid_Enum(Enum):
    C1=0; D1=1; H1=2; S1=3; N1=4;
    C2=5; D2=6; H2=7; S2=8; N2=9;
    C3=10; D3=11; H3=12; S3=13; N3=14;
    C4=15; D4=16; H4=17; S4=18; N4=19;
    C5=20; D5=21; H5=22; S5=23; N5=24;
    C6=25; D6=26; H6=27; S6=28; N6=29;
    C7=30; D7=31; H7=32; S7=33; N7=34;
    P=35; D=36; R=37

    def __repr__(self):
        return self.name[::-1]

ACTION_MEANING = {
    0: "C1", 1: "D1", 2: "H1", 3: "S1", 4: "N1",
    5: "C2", 6: "D2", 7: "H2", 8: "S2", 9: "N2",
    10: "C3", 11: "D3", 12: "H3", 13: "S3", 14: "N3",
    15: "C4", 16: "D4", 17: "H4", 18: "S4", 19: "N4",
    20: "C5", 21: "D5", 22: "H5", 23: "S5", 24: "N5",
    25: "C6", 26: "D6", 27: "H6", 28: "S6", 29: "N6",
    30: "C7", 31: "D7", 32: "H7", 33: "S7", 34: "N7",
    35: "P", 36: "D", 37: "R"
}

class Position(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

class Suit(Enum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3
    NO_TRUMP = 4
