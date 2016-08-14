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

ELOBalanceTestSet = collections.namedtuple("ELOBalanceTestSet", ["name", "input_elos", "team_a", "team_b"])


class ELOTestSetRegistry(object):
    GLOBAL_TESTS = collections.OrderedDict()

    @classmethod
    def add_test(cls, test_set):
        assert isinstance(test_set, ELOBalanceTestSet)
        assert test_set.name not in cls.GLOBAL_TESTS
        cls.GLOBAL_TESTS[test_set.name] = test_set

    @classmethod
    def iter_tests(cls):
        for v in cls.GLOBAL_TESTS.values():
            yield v


def register_elo_test(test_set):
    ELOTestSetRegistry.add_test(test_set)

ELO_TEST_DATA = [
    ELOBalanceTestSet(name="Test01",
                      input_elos=[1841, 1616, 1402, 1401, 1395, 1368, 1170, 1091, 921, 816],
                      team_a=[1841, 1402, 1395, 1091, 816],
                      team_b=[1616, 1401, 1368, 1170, 921]),

    ELOBalanceTestSet(name="Test02",
                      input_elos=[2150, 1640, 1600, 1212, 929],
                      team_a=[2150, 1600],
                      team_b=[1640, 1212, 929])
]

for ELO_TEST in ELO_TEST_DATA:
    ELOTestSetRegistry.add_test(ELO_TEST)


def sorted_elos(team):
    return tuple(sorted(team, reverse=True))


def confirm_test_set_match(test_set, balanced_teams, test_label="", print_failure=False, print_success=False):
    balanced_team_a, balanced_team_b = balanced_teams
    output_set = {sorted_elos(balanced_team_a), sorted_elos(balanced_team_b)}
    if len(output_set) <= 1:
        # No players or both teams exactly match
        return True
    expected_set = {sorted_elos(test_set.team_a), sorted_elos(test_set.team_b)}
    success = (expected_set == output_set)
    do_print = (print_failure and not success) or (print_success and success)
    if do_print:
        outcome = "OK" if success else "Failure"
        label = ("%s: " % test_label) if test_label else ""
        input_as_set = set((tuple(test_set.input_elos), ))
        desc = "%s: %s\n\t   input=%s, \n\t  output=%s, \n\texpected=%s" % (outcome, label, input_as_set, output_set, expected_set)
        print desc
    if success:
        return True
    return False


def single_elo_test(test_case, balance_algorithm, print_success=False):
    assert isinstance(test_case, ELOBalanceTestSet)
    test_name, elos, expected_a, expected_b = test_case
    players = generate_player_info_list_from_elos(elos)
    balanced_teams = balance_algorithm(players)
    balanced_team_a, balanced_team_b = balanced_teams

    def elos_only(li):
        return [i.elo for i in li]

    elos_a, elos_b = elos_only(balanced_team_a), elos_only(balanced_team_b)
    balanced_elos = (elos_a, elos_b)
    result = confirm_test_set_match(test_case, balanced_elos, test_label=test_name, print_failure=True, print_success=print_success)
    return result


class UnstakBalanceTest(unittest.TestCase):
    def test_elo_balancing_skill_band(self):
        for test_set in ELOTestSetRegistry.iter_tests():
            result = single_elo_test(test_set, balancer.balance_players_by_skill_variance, print_success=True)
            #self.assertTrue(result, "Team Mismatch")


def run_tests():
    pass

if __name__ == '__main__':
    unittest.main()
