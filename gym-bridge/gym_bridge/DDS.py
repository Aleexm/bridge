import ctypes
from ctypes import POINTER, Structure, byref, c_char, c_int, c_uint
import os
import sys
import numpy as np
import gym_bridge.analyze as analyze

'''
Credit to anntzer, https://github.com/anntzer/redeal for showing how to ctypes.
Some functions (solve_pbn functionality) copied, calc_table_PBN (4x5 DDS table)
added.

This script loads in Bo Haglund's (http://privat.bahnhof.se/wb758135/index.html)
DDS solver library, and can compute Double Dummy scores for a given hand,
declarer, contract, etc.
'''

def calc_multiple_table_PBN(multiple_encoded_hands):
    '''
    Computes len(multiple_encoded_hands) DDS tables, one for each encoded hand.
    Wraps CalcAllTablesPBN.

    args:
        - multiple_encoded_hands(list(list(list(int)))): n x 4 x 52 ints.
    '''
    table_deal_pbns = []
    for encoded_hands in multiple_encoded_hands:
        hands = analyze.deserialize_hands(encoded_hands)
        c_table_deal_pbn = TableDealPBN.from_deal(hands)
        table_deal_pbns.append(c_table_deal_pbn)
    num_tables = len(multiple_encoded_hands)
    c_multiple_table_PBN =  TableDealsPBN(32, tuple(table_deal_pbns))
    tables_results = TablesRes()
    par_results = AllParResults()
    status = dll.CalcAllTablesPBN(c_multiple_table_PBN,
                                  -1,
                                  (c_int * 5)(0,0,0,0,0),
                                  byref(tables_results),
                                  byref(par_results))
    if status != 1:
        raise Exception("CalcAllTablesPBN({}, ...) failed with status {} ({}).".
                        format(hands, status, SolveBoardStatus[status]))
    processed_tables = np.zeros((len(multiple_encoded_hands), 4 ,5))
    for i, table in enumerate(tables_results.ddTableResults):
        # Get the values from c int array
        res_table = np.transpose(np.array(list([list(row)
                                                for row in table.table])))
        # Set ordering to C-D-H-S-NT for suits (player order is N-E-S-W).
        processed = np.c_[np.flip(res_table[:,:4],axis=1), res_table[:,-1]]
        processed_tables[i,:,:] = processed
    return processed_tables

def calc_table_PBN(encoded_hands):
    '''
    Computes the 4x5 Table for a deal. res[1][2] contains the number of
    tricks declarer E (1) can take with suit H (2). Wraps calcDDtablePBN.

    args:
        - encoded_hands(list(list(int))): 4x52 one-hot encodings of hands

    returns:
        - processed(list(list(int))): 4x5 Table w/ number of tricks taken
                                      for each (declarer, suit) combination.
                                      decl: nesw, suits: cdhsn
    '''
    hands = analyze.deserialize_hands(encoded_hands)
    c_table_deal_pbn = TableDealPBN.from_deal(hands)
    res_object = TableResults()
    status = dll.CalcDDtablePBN(c_table_deal_pbn, byref(res_object))
    if status != 1:
        raise Exception("calcDDtablePBN({}, ...) failed with status {} ({}).".
                        format(hands, status, SolveBoardStatus[status]))
    # Get the values from c int array
    res_table = np.transpose(np.array(list([list(row)
                                            for row in res_object.table])))
    # Set ordering to C-D-H-S-NT for suits (player order is N-E-S-W).
    processed = np.c_[np.flip(res_table[:,:4],axis=1), res_table[:,-1]]
    return processed

def solve_pbn(encoded_hands, encoded_contract, declarer):
    '''
    Return the number of tricks for declarer; wraps SolveBoardPBN.

    args:
        - encoded_hands(list(list(int))): 4x52 one-hot encodings of hands
        - encoded_contract(int): {0,1,...34} 2C<2D<...7S<7N.
        - declarer(int): 0:N, 1:E, 2:S, 3:W.

    returns:
        - tricks_taken(int): Number of tricks the declarer will win (out of 13).
    '''
    leader = (declarer + 1) % 4 # SolveBoardPBN expects leader, not declarer
    hands = analyze.deserialize_hands(encoded_hands)
    suit = encoded_contract % 5
    c_deal_pbn = DealPBN.from_deal(hands, suit, leader)
    futp = FutureTricks()
    status = dll.SolveBoardPBN(c_deal_pbn, -1, 1, 1, byref(futp), 0)
    if status != 1:
        raise Exception("SolveBoardPBN({}, ...) failed with status {} ({}).".
                        format(hands, status, SolveBoardStatus[status]))
    tricks_taken = 13 - futp.score[0]
    return tricks_taken

def to_c_suit(int):
    '''
    DDS Solver expects suits ordered as 0:S, 1:H, 2:D, 3:C, 4:N.
    I use 0:C, 1:D, 2:H, 3:S, 4:N.
    '''
    return {
        0: 3, 1: 2, 2: 1, 3: 0, 4: 4,
    }[int]

class TableDealPBN(Structure):
    """The ddTableDealPBN struct."""

    _fields_ = [
        ("cards", c_char * 80), # PBN-like format
    ]

    @classmethod
    def from_deal(cls, hands):
        return cls(cards = b"N:" + hands.encode('ascii'))


class TableDealsPBN(Structure):
    """The ddTableDealsPBN struct."""

    _fields_ = [
        ("noOfTables", c_int), # Number of DD table deals in structure.
        ("ddTableDealPBN", TableDealPBN * 32)
    ]
    # @classmethod
    # def from_data(cls, num_tables, PBNs):
    #     return cls(noOfTables = 10,
    #                ddTableDealPBN = (TableDealPBN * num_tables)
    #                (tuple(PBNs)))

class TableResults(Structure):
    """The ddTableResults struct."""

    _fields_ = [
        ("table", (c_int * 4) * 5) # 5x4 table of Suit x Declarer tricks
    ]

class TablesRes(Structure):
    """The ddTablesRes struct."""

    _fields_ = [
        ("noOfBoards", c_int), # The number of DD table deals in structure.
        ("ddTableResults", TableResults * 32)
    ]

    # @classmethod
    # def test(cls, num_tables):
    #     return cls(ddTableResults = (TableResults * num_tables))

class ParResults(Structure):
    """The parResults struct."""

    _fields_ = [
        ("parScore", (c_char * 16) * 2), # First index is NS/EW. Side encoding.
        ("parContractsString", (c_char * 32) * 2) # As above
    ]

class AllParResults(Structure):
    """The allParResults struct."""

    _fields_ = [
        ("parResults", ParResults*32)
    ]

class DealPBN(Structure):
    """The dealPBN struct."""

    _fields_ = [
        ("trump", c_int),  # 0=S, 1=H, 2=D, 3=C, 4=NT
        ("first", c_int),  # leader: 0=N, 1=E, 2=S, 3=W
        ("currentTrickSuit", c_int * 3),
        ("currentTrickRank", c_int * 3),  # 2-14, up to 3 cards; 0=unplayed
        ("remainCards", c_char * 80),  # PBN-like format
    ]

    @classmethod
    def from_deal(cls, hands, suit, leader):
        return cls(trump=to_c_suit(suit),
                   first=leader,
                   currentTrickSuit=(c_int * 3)(0, 0, 0),
                   currentTrickRank=(c_int * 3)(0, 0, 0),
                   remainCards=b"N:" + hands.encode('ascii'))

class FutureTricks(Structure):
    """The futureTricks struct."""

    _fields_ = [("nodes", c_int),
                ("cards", c_int),
                ("suit", c_int * 13),
                ("rank", c_int * 13),
                ("equals", c_int * 13),
                ("score", c_int * 13)]


SolveBoardStatus = { # Error codes thrown by DDS solver
    1: "No fault",
    -1: "Unknown fault",
    -2: "Zero cards",
    -3: "Target > tricks left",
    -4: "Duplicated cards",
    -5: "Target < -1",
    -7: "Target > 13",
    -8: "Solutions < 1",
    -9: "Solutions > 3",
    -10: "> 52 cards",
    -12: "Invalid deal.currentTrick{Suit,Rank}",
    -13: "Card played in current trick is also remaining",
    -14: "Wrong number of remaining cards in a hand",
    -15: "threadIndex < 0 or >=noOfThreads, noOfThreads is the configured "
         "maximum number of threads",
}

# Load correct library
dll_name = DLL = None
if os.name == "posix":
    if sys.maxsize > 2 ** 32:
        dll_name = os.path.join("DDS_libs", "libdds-64.so")
    else:
        dll_name = os.path.join("DDS_libs", "libdds-32.so")
    DLL = ctypes.CDLL
elif os.name == "nt":
    if sys.maxsize > 2 ** 32:
        dll_name = os.path.join("DDS_libs", "dds-64.dll")
    else:
        dll_name = os.path.join("DDS_libs", "dds-32.dll")
    DLL = ctypes.WinDLL

if dll_name:
    dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            dll_name)

if dll_name and os.path.exists(dll_path):
    dll = DLL(dll_path)
    dll.SolveBoardPBN.argtypes = [
        DealPBN, c_int, c_int, c_int, POINTER(FutureTricks), c_int]
    dll.CalcDDtablePBN.argtypes = [TableDealPBN, POINTER(TableResults)]
    if os.name == "posix":
        dll.SetMaxThreads(0)
