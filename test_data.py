import random
import math

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


def test_set_1():
    d = {
        0: ("Ivan", 1841, None),
        1: ("Alice", 1616, None),
        2: ("Sandra", 1401, None),
        3: ("Bob", 1368, None),
        4: ("Mike", 921, None),
        5: ("Henry", 1402, None),
        6: ("Carly", 1395, None),
        7: ("Nathan", 1170, None),
        8: ("Quentin", 1091, None),
        9: ("Zachary", 816, None),
    }
    return player_info_list_from_steam_id_name_ext_obj_elo_dict(d)

def test_set_2():
    d = {
        0: ("Stakz", 2150, None),
        1: ("Purger", 1600, None),
        2: ("Comets", 929, None),
        3: ("the_toilet", 1640, None),
        4: ("merozollo", 1212, None),
    }
    return player_info_list_from_steam_id_name_ext_obj_elo_dict(d)
