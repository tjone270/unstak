import random
import math
import unittest
from balancer import *
import collections

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
                      team_b=[1640, 1212, 929]),

    ELOBalanceTestSet(name="Test03",
                      input_elos=[2290, 2249, 2073, 2045, 2025, 2019, 1993, 1843, 1691, 1600, 1585, 1532, 1493, 1493, 1437, 1337],
                      team_a=[2073, 2045, 2025, 2019, 1585, 1532, 1493, 1493],
                      team_b=[2290, 2249, 1993, 1843, 1691, 1600, 1437, 1337]),

    ELOBalanceTestSet(name="Test04",
                      input_elos=[2249, 1993, 1941, 1930, 1836, 1689, 1626, 1574, 1493, 1493, 1473, 1176],
                      team_a=[1941, 1836, 1689, 1626, 1574, 1493],
                      team_b=[2249, 1993, 1930, 1493, 1473, 1176]),

    ELOBalanceTestSet(name="Test05",
                      input_elos=[2216, 1984, 1942, 1682, 1589, 1543, 1469, 1337, 1252, 1200, 950, 948, 871, 627],
                      team_a=[1942, 1589, 1543, 1337, 1252, 1200, 948],
                      team_b=[2216, 1984, 1682, 1469, 950, 871, 627])

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


def single_elo_test(test_case, print_success=False, **kwargs):
    assert isinstance(test_case, ELOBalanceTestSet)
    test_name, elos, expected_a, expected_b = test_case
    players = generate_player_info_list_from_elos(elos)
    balanced_team_combos = balance_players_by_skill_variance(players, **kwargs)
    balanced_teams = None
    for i, team_combo in enumerate(balanced_team_combos):
        assert isinstance(team_combo, BalancedTeamCombo)
        if i == 0:
            balanced_teams = team_combo.teams_tup
        print("results[%d]: %s" % (i, describe_balanced_team_combo(team_combo.teams_tup[0],
                                                                            team_combo.teams_tup[1],
                                                                            team_combo.match_prediction)))
    balanced_team_a, balanced_team_b = balanced_teams

    # since its a convenient place, calculate and present the top switch scenarios:
    switch_proposals = generate_switch_proposals(balanced_teams, **kwargs)

    #player_dict = {(player.steam_id, player) for player in players}

    # prioritize switch operations that affects the least players (or better to not care about this???)
    team_names = ["blue", "red"]
    switch_proposals = sorted(switch_proposals, key=lambda sp: abs(sp.balanced_team_combo.match_prediction.distance))
    for i, switch_proposal in enumerate(switch_proposals):
        assert isinstance(switch_proposal, SwitchProposal)
        switch_operation = switch_proposal.switch_operation
        switch_team_combo = switch_proposal.balanced_team_combo
        assert isinstance(switch_operation, SwitchOperation)
        assert isinstance(switch_team_combo, BalancedTeamCombo)
        match_prediction = switch_team_combo.match_prediction
        assert isinstance(match_prediction, MatchPrediction)
        print("switch option [%d]: %s | %s" %
              (i, describe_switch_operation(switch_operation, team_names=team_names),
               match_prediction.describe_prediction_short(team_names=team_names)))

    def elos_only(li):
        return [i.elo for i in li]

    elos_a, elos_b = elos_only(balanced_team_a), elos_only(balanced_team_b)
    balanced_elos = (elos_a, elos_b)
    result = confirm_test_set_match(test_case, balanced_elos, test_label=test_name, print_failure=True, print_success=print_success)
    return result


class UnstakBalanceTest(unittest.TestCase):
    def test_elo_balancing_skill_band(self):
        for test_set in ELOTestSetRegistry.iter_tests():
            result = single_elo_test(test_set, print_success=True, max_results=5)
            #self.assertTrue(result, "Team Mismatch")


def run_tests():
    pass

if __name__ == '__main__':
    unittest.main()
