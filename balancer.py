# UNSTAK_START ---------------------------------------------------------------------------------------------------------
#
# unstak, an alternative balancing method for minqlx created by github/hyperwired aka "stakz", 2016-07-31
# This code is released under the MIT Licence:
#
# The MIT License (MIT)
#
# Copyright (c) 2016
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------
import bisect
import collections
import itertools
import math
import operator
import random

def format_obj_desc_str(obj):
    oclass = obj.__class__
    a = str(obj.__module__)
    b = str(obj.__class__.__name__)
    return "%s.%s %s" % (a, b, obj.desc())


def format_obj_desc_repr(obj):
    return "<%s object @ 0x%x>" % (format_obj_desc_str(obj), id(obj))


class PerformanceSnapshot(object):
    def __init__(self, elo, elo_variance):
        self._elo = elo
        self._elo_variance = elo_variance

    @property
    def elo(self):
        return self._elo

    @property
    def elo_variance(self):
        return self._elo_variance

    def desc(self):
        return "elo=%s (~%s)" % (self._elo, self._elo_variance)

    def __str__(self):
        return format_obj_desc_str(self)

    def __repr__(self):
        return format_obj_desc_repr(self)


class PerformanceHistory(object):
    def __init__(self):
        self._snapshots = []

    def has_data(self):
        return len(self._snapshots)

    def latest_snapshot(self):
        if self.has_data():
            return self._snapshots[-1]
        return None

    def desc(self):
        latest = self.latest_snapshot()
        if latest:
            return "%s, history=%s" % (latest.desc(), len(self._snapshots))
        return "<empty>"

    def __str__(self):
        return format_obj_desc_str(self)

    def __repr__(self):
        return format_obj_desc_repr(self)


class PlayerInfo(object):
    def __init__(self, name=None, perf_history=None, steam_id=None, ext_obj=None):
        self._name = name
        self._perf_history = perf_history
        self._steam_id = steam_id
        self._ext_obj = ext_obj

    @property
    def steam_id(self):
        return self._steam_id

    @property
    def ext_obj(self):
        return self._ext_obj

    @property
    def perf_history(self):
        return self._perf_history

    @property
    def latest_perf(self):
        return self._perf_history.latest_snapshot()

    @property
    def elo(self):
        return self.latest_perf.elo

    @property
    def elo_variance(self):
        return self.latest_perf.elo_variance

    @property
    def name(self):
        return self._name

    def desc(self):
        return "'%s': %s" % (self._name, self._perf_history.desc())

    def __str__(self):
        return format_obj_desc_str(self)

    def __repr__(self):
        return format_obj_desc_repr(self)


def player_info_list_from_steam_id_name_ext_obj_elo_dict(d):
    out = []
    for steam_id, (name, elo, ext_obj) in d.items():
        perf_snap = PerformanceSnapshot(elo, 0)
        perf_history = PerformanceHistory()
        perf_history._snapshots.append(perf_snap)
        player_info = PlayerInfo(name, perf_history, steam_id=steam_id, ext_obj=ext_obj)
        out.append(player_info)
    return out


class FixedSizePriorityQueue(object):
    def __init__(self, max_count):
        assert max_count
        self.max_count = max_count
        self.items = []

    def __len__(self):
        return len(self.items)

    def add_item(self, item):
        bisect.insort_right(self.items, item)
        if len(self.items) > self.max_count:
            self.items.pop()

    def nsmallest(self):
        return self.items[:self.max_count]


def sort_by_skill_rating_descending(players):
    return sorted(players, key=lambda p: (p.elo, p.name), reverse=True)


def skill_rating_list(players):
    return [p.elo for p in players]


def calc_mean(values):
    return sum(values)/(1.0*len(values))


def calc_standard_deviation(values, mean=None):
    if mean is None:
        mean = calc_mean(values)
    variance = calc_mean([(val - mean) ** 2 for val in values])
    return math.sqrt(variance)


class PlayerStats(object):
    def __init__(self, player):
        self.player = player
        self.relative_deviation = 0

    @property
    def relative_deviation_category(self):
        if self.relative_deviation < 0:
            return math.ceil(self.relative_deviation)
        return math.floor(self.relative_deviation)


class TeamStats(object):
    def __init__(self, team_players):
        self.players = team_players

    def get_elo_list(self, player_stats_dict):
        return [player_stats_dict[pid].player.elo for pid in self.players]

    def combined_skill_rating(self, player_stats_dict):
        return sum(self.get_elo_list(player_stats_dict))

    def skill_rating_stdev(self, player_stats_dict):
        return calc_standard_deviation(self.get_elo_list(player_stats_dict))



class SingleTeamBakedStats(object):
    def __init__(self):
        self.skill_rating_sum = 0
        self.skill_rating_mean = 0
        self.skill_rating_stdev = 0
        self.num_players = 0
        self.players_by_stdev_rel_server_dict = {}
        self.players_by_speed_rel_server_dict = {}


class MatchPrediction(object):
    def __init__(self):
        self.team_a = SingleTeamBakedStats()
        self.team_b = SingleTeamBakedStats()
        self.bias = 0
        self.distance = 0
        self.confidence = 0

    def describe_prediction_short(self, team_names=None):
        # assuming bias is in the interval [-1,1], convert it to favoured chance so that
        # a bias of zero gets presented as a 50%/50% win prediction
        left_team_desc = ""
        right_team_desc = ""
        if team_names:
            assert len(team_names) == 2
            left_team_desc = "%s " % team_names[0]
            right_team_desc = " %s" % team_names[1]

        left_win_chance = self.bias * 100
        right_win_chance = 100 - left_win_chance
        return "%s%.2f%%/%.2f%%%s" % (left_team_desc, left_win_chance, right_win_chance, right_team_desc)

    def get_desc(self):
        raise NotImplementedError


def generate_match_prediction(team_a_baked, team_b_baked):
    assert isinstance(team_a_baked, SingleTeamBakedStats)
    assert isinstance(team_b_baked, SingleTeamBakedStats)
    prediction = MatchPrediction()
    prediction.team_a = team_a_baked
    prediction.team_b = team_b_baked
    prediction.bias = (1.0 * team_b_baked.skill_rating_sum) / (team_a_baked.skill_rating_sum + team_b_baked.skill_rating_sum)
    prediction.distance = (0.5 - prediction.bias)
    return prediction


class BalancePrediction(object):
    def __init__(self, team_a, team_b):
        self.team_a_stats = TeamStats(team_a)
        self.team_b_stats = TeamStats(team_b)

    def generate_match_prediction(self, player_stats_dict):
        stats = []
        for team in [self.team_a_stats, self.team_b_stats]:
            bs = SingleTeamBakedStats()
            bs.skill_rating_sum = team.combined_skill_rating(player_stats_dict)
            bs.num_players = len(team.players)
            assert bs.num_players
            bs.skill_rating_mean = bs.skill_rating_sum/bs.num_players
            bs.skill_rating_stdev = team.skill_rating_stdev(player_stats_dict)
            stats.append(bs)
        return generate_match_prediction(*stats)


def nchoosek(n, r):
        r = min(r, n - r)
        if r == 0:
            return 1
        numerator = reduce(operator.mul, xrange(n, n - r, -1))
        denominator = reduce(operator.mul, xrange(1, r + 1))
        return numerator // denominator


BalancedTeamCombo = collections.namedtuple("BalancedTeamCombo", ["teams_tup", "match_prediction"])


def player_ids_only(team):
    if team and isinstance(team[0], PlayerInfo):
        return [p.steam_id for p in team]
    return team


def describe_balanced_team_combo(team_a, team_b, match_prediction):
    assert isinstance(match_prediction, MatchPrediction)
    return "Team A: %s | Team B: %s | outcome: %s" % (player_ids_only(team_a), player_ids_only(team_b),
                                                        match_prediction.describe_prediction_short())


def balance_players_by_skill_variance(players, verbose=False, prune_search_space=True, max_results=None):
    players = sort_by_skill_rating_descending(players)
    player_stats = collections.OrderedDict()
    for p in players:
        player_stats[p.steam_id] = PlayerStats(p)

    # We assume that the skill rating is based on a Gaussian (normal) distribution in the global player population.
    # The input players is a sample of the actual population, but for the purposes of this algorithm, we treat the
    # sample as the population, since we are not trying to make inferences on the actual population. We also
    # assume a normal distribution in the sample to find outlier players, which should be true in most scenarios.
    sample_mean = calc_mean(skill_rating_list(players))
    sample_stdev = calc_standard_deviation(skill_rating_list(players), mean=sample_mean)

    if verbose:
        print "sample mean: %.2f" % sample_mean
        print "sample stdev: %.2f" % sample_stdev

    # for each player, determine their distance in standard deviations from the sample mean.
    deviation_categories = collections.OrderedDict()
    for player_stat in player_stats.values():
        assert isinstance(player_stat, PlayerStats)
        player = player_stat.player
        assert isinstance(player, PlayerInfo)
        player_stat.relative_deviation = 1.0 * ((player_stat.player.elo - sample_mean) / (sample_stdev * 1.0))
        print "%s(%d): skill=%s stdev=%.2f" % (player.name, player.steam_id, player.elo, player_stat.relative_deviation)
        if player_stat.relative_deviation_category not in deviation_categories:
            deviation_categories[player_stat.relative_deviation_category] = []
        deviation_categories[player_stat.relative_deviation_category].append(player_stat)

    if verbose:
        print deviation_categories

    # Do a brute force space search by trying different combos of player picks.
    #   - Search space reduction techniques:
    #       - generating combinations (n choose n/2) per standard deviation
    #       - now do a search through the cartesian product of 1 pick from each stdev combo
    #       - picks where the distance between left and right teams is more than 2 players per stdev can be ommitted.
    #   - Define and use a set of heuristics to predict the match quality of each possible teams pick.
    #   - Only maintain the top N matches in memory/results

    categories = []

    def generate_category_combo_sets(player_stats_list):
        # generate all combos of 2-team player picks within an stdev skill category
        full_set = set((player_stat.player.steam_id for player_stat in player_stats_list))
        if len(full_set) == 1:
            # handle scenario where there is only one player in the category. Add a dummy None player
            full_set.add(None)
        full_list = list(full_set)
        for category_combo in itertools.combinations(full_list, int(math.ceil(len(full_list)/2.0))):
            yield (category_combo, tuple(full_set.difference(category_combo)))

    def generate_team_combinations(deviation_categories_dict, total_players, prune_search=False):
        # generate all valid cross-category 2-team picks via a filtered cartesian product of valid in-category combos.
        generators = collections.OrderedDict()
        for deviation_category, player_stats in deviation_categories_dict.items():
            generators[deviation_category] = generate_category_combo_sets(player_stats)

        # feed the generators into the cartesian product generator
        for teams_combo in (itertools.product(*generators.values())):
            running_delta = 0
            valid_combo = True
            # strip out dummy/None players
            strip_none = lambda ps: tuple(p for p in ps if p is not None)
            teams_combo = tuple((strip_none(team_category[0]), strip_none(team_category[1])) for team_category in teams_combo)
            counted_players = sum(len(team_category) for team_category in itertools.chain.from_iterable(teams_combo))
            if prune_search_space:
                for team_category in teams_combo:
                    # filter to disallow bias on same team in 2 adjacent skill categories
                    players_a, players_b = team_category
                    category_delta = len(players_b) - len(players_a)
                    if abs(category_delta) >= 2:
                        valid_combo = False
                        break
                    running_delta += category_delta
                    if abs(running_delta) >= 2:
                        valid_combo = False
                        break
            if valid_combo:
                yield teams_combo

    def worst_case_search_space_combo_count(players):
        players_count = len(players)
        return nchoosek(players_count, int(math.ceil(players_count/2.0)))

    def search_optimal_team_combinations(teams_generator):
        # iterate through the generated teams, using heuristics to rate them and keep the top N
        for teams_combo in teams_generator:
            teams = tuple(tuple(itertools.chain.from_iterable(team)) for team in zip(*teams_combo))
            yield teams

    def analyze_teams(teams):
        return BalancePrediction(teams[0], teams[1])


    total_players = len(players)
    teams_combos_generator = generate_team_combinations(deviation_categories, total_players, prune_search=prune_search_space)

    max_iterations = worst_case_search_space_combo_count(players)
    max_iteration_digits = int(math.log10(max_iterations)+1)
    FixedSizePriorityQueue(max_results)

    results = FixedSizePriorityQueue(max_results)

    for i, teams in enumerate(search_optimal_team_combinations(teams_combos_generator)):
        balance_prediction = analyze_teams(teams)
        assert isinstance(balance_prediction, BalancePrediction)
        match_prediction = balance_prediction.generate_match_prediction(player_stats)
        abs_balance_distance = abs(match_prediction.distance)
        results.add_item((abs_balance_distance, match_prediction, teams))
        if verbose:
            combo_desc = str(i+1).ljust(max_iteration_digits, " ")
            print "Combo %s : %s" % (combo_desc, describe_balanced_team_combo(teams[0], teams[1], match_prediction))

    # This step seems heavyweight if we are to return a lot of results, so max_results should always be small.
    # convert it back into a list of players
    result_combos = []
    for result in results.nsmallest():
        teams_as_players = []
        (abs_balance_distance, match_prediction, teams) = result
        for team in teams:
            teams_as_players.append(tuple(player_stats[pid].player for pid in team))
        team_combo = BalancedTeamCombo(teams_tup=tuple(teams_as_players), match_prediction=match_prediction)
        result_combos.append(team_combo)
    return result_combos


SwitchOperation = collections.namedtuple("SwitchOperation", ["players_affected",
                                                             "players_moved_from_a_to_b", "players_moved_from_b_to_a"])

SwitchProposal = collections.namedtuple("SwitchProposal", ["switch_operation", "balanced_team_combo"])


def get_proposed_team_combo_moves(team_combo_1, team_combo_2):
    # team_combo_1 is current, team_combo_2 is a proposed combination
    assert len(team_combo_1) == 2 and len(team_combo_2) == 2
    team1a, team1b = set(team_combo_1[0]), set(team_combo_1[1])
    if isinstance(team_combo_2, BalancedTeamCombo):
        team2a, team2b = set(team_combo_2.teams_tup[0]), set(team_combo_2.teams_tup[1])
    else:
        team2a, team2b = set(team_combo_2[0]), set(team_combo_2[1])
    assert team1a.union(team1b) == team2a.union(team2b), "inconsistent input data"
    assert not team1a.intersection(team1b), "inconsistent input data"
    assert not team2a.intersection(team2b), "inconsistent input data"
    players_moved_from_a_to_b = team2a.difference(team1a)
    players_moved_from_b_to_a = team2b.difference(team1b)
    players_affected = players_moved_from_a_to_b.union(players_moved_from_b_to_a)
    return SwitchOperation(players_affected=players_affected,
                           players_moved_from_a_to_b=players_moved_from_a_to_b,
                           players_moved_from_b_to_a=players_moved_from_b_to_a)


def describe_switch_operation(switch_op, team_names=None):
    assert isinstance(switch_op, SwitchOperation)
    left_team_desc = ""
    right_team_desc = ""
    if team_names:
        assert len(team_names) == 2
        left_team_desc = "%s " % team_names[0]
        right_team_desc = " %s" % team_names[1]

    def get_names(player_set):
        s = []
        for i, player in enumerate(sorted(list(player_set), key=lambda p: p.elo, reverse=True)):
            if i != 0:
                s.append(", ")
            s.append("%s(%d)" % (player.name, player.elo))
        return "".join(s)

    out = []
    if switch_op.players_moved_from_a_to_b:
        out.append("%s --->%s" % (get_names(switch_op.players_moved_from_a_to_b), right_team_desc))
    if switch_op.players_moved_from_a_to_b and switch_op.players_moved_from_b_to_a:
        out.append(" | ")
    if switch_op.players_moved_from_b_to_a:
        out.append("%s<--- %s" % (left_team_desc, get_names(switch_op.players_moved_from_b_to_a)))
    return "".join(out)


def generate_switch_proposals(teams, verbose=False, max_results=5):
    # add 1 to max results, because if the input teams are optimal, then they will come as a result.
    players = []
    [[players.append(p) for p in team_players] for team_players in teams]
    balanced_team_combos = balance_players_by_skill_variance(players,
                                                             verbose=verbose,
                                                             prune_search_space=True,
                                                             max_results=max_results+1)
    switch_proposals = []
    for balanced_combo in balanced_team_combos:
        switch_op = get_proposed_team_combo_moves(teams, balanced_combo)
        assert isinstance(switch_op, SwitchOperation)
        assert isinstance(balanced_combo, BalancedTeamCombo)
        if not switch_op.players_affected:
            # no change
            continue
        switch_proposals.append(SwitchProposal(switch_operation=switch_op, balanced_team_combo=balanced_combo))

    return switch_proposals

# UNSTAK_END -----------------------------------------------------------------------------------------------------------
