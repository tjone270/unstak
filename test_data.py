import random
import math
import unittest
import balancer
import collections

from player_info import PlayerInfo, PerformanceHistory, PerformanceSnapshot
from player_info import player_info_list_from_steam_id_name_ext_obj_elo_dict

TEST_PLAYER_NAMES = [
    "Alice", "Bob", "Carly", "Daniel", "Eugene", "Fred", "George", "Henry", "Ivan", "Julia", "Kim", "Lucy",
    "Mike", "Nathan", "Olivia", "Patricia", "Quentin", "Robert", "Sandra", "Thomas", "Ulric", "Vivian",
    "William", "Xavier", "Yuri", "Zachary"
]


def clamp(value, min_value=600, max_value=2700):
    return min(max(int(value), min_value), max_value)


def generate_player_set(num_players=10, random_elos=True):
    assert len(TEST_PLAYER_NAMES) >= num_players
    players = random.sample(TEST_PLAYER_NAMES, num_players)
    player_infos = []
    for player in players:
        elo = clamp(random.gauss(1400, 380), min_value=600, max_value=2700)
        elo_confidence = clamp(random.gauss(55, 40), min_value=20, max_value=150)
        perf_snap = PerformanceSnapshot(elo, elo_confidence)
        perf_history = PerformanceHistory()
        perf_history._snapshots.append(perf_snap)
        player_info = PlayerInfo(player, perf_history)
        player_infos.append(player_info)

    return player_infos

def generate_player_info_list_from_elos(player_elos):
    d = {}
    for i, elo in enumerate(player_elos):
        assert i < len(TEST_PLAYER_NAMES)
        d[i] = (TEST_PLAYER_NAMES[i], elo, None)
    return player_info_list_from_steam_id_name_ext_obj_elo_dict(d)

ELOBalanceTestSet = collections.namedtuple("ELOBalanceTestSet", ["input_elos", "team_a", "team_b"])

class TestSets(object):
    TEST_SET_01 = ELOBalanceTestSet(input_elos=[1841, 1616, 1402, 1401, 1395, 1368, 1170, 1091, 921, 816],
                                    team_a=[1841, 1402, 1395, 1091, 816],
                                    team_b=[1616, 1401, 1368, 1170, 921])

    TEST_SET_02 = ELOBalanceTestSet(input_elos=[2150, 1640, 1600, 1212, 929],
                                    team_a=[2150, 1600],
                                    team_b=[1640, 1212, 929])

class UnstakBalanceTest(unittest.TestCase):
    def test_set_1(self):
        elos, (expected_a, expected_b) = TestSets.TEST_SET_01
        players = generate_player_info_list_from_elos(elos)
        balancer.balance_players_by_skill_band(players)



def run_tests():
    pass