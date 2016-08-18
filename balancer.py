#----------------------------------------------------------------------------------------------------------------------------------------
# unstak, an alternative balancing method for minqlx created by github/hyperwired aka "stakz", 2016-07-31
# This plugin is released to everyone, for any purpose. It comes with no warranty, no guarantee it works, it's released AS IS.
# You can modify everything, except for lines 1-4. They're there to indicate I whacked this together originally. Please make it better :D

import collections
import itertools
import math
import operator
import random
import heapq

from player_info import PlayerInfo


class FixedSizePriorityQueue(object):
    def __init__(self, max_count=None):
        self.max_count = max_count
        self.heap = []
        heapq.heapify(self.heap)

    def __len__(self):
        return len(self.heap)

    def add_item(self, item):
        if not self.max_count or len(self.heap) < self.max_count:
            heapq.heappush(self.heap, item)
        else:
            heapq.heappushpop(self.heap, item)

    def nlargest(self):
        n = self.max_count if self.max_count else len(self.heap)
        return heapq.nlargest(n, self.heap)

    def nsmallest(self):
        n = self.max_count if self.max_count else len(self.heap)
        return heapq.nsmallest(n, self.heap)


def sort_by_skill_rating_descending(players):
    return sorted(players, key=lambda p: (p.elo, p.name), reverse=True)


def balance_players_random(players):
    """
    Shuffle teams completely randomly.
    Non deterministic (random)

    :param players: a list of all the players that are to be balanced
    :return: (team_a, team_b) 2-tuple of lists of players
    """
    out = list(players)
    random.shuffle(out)
    total = len(out)
    return out[:total/2], out[total/2:]


def balance_players_ranked_odd_even(players):
    """
    Balance teams by first sorting players by skill and then picking players alternating to teams.
    Deterministic (Stable) for a given input.

    :param players: a list of all the players that are to be balanced
    :return: (team_a, team_b) 2-tuple of lists of players
    """
    presorted = sort_by_skill_rating_descending(players)
    teams = ([], [])
    for i, player in enumerate(presorted):
        teams[i % 2].append(player)
    return teams


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

    def combined_skill_rating(self, player_stats_dict):
        return sum(player_stats_dict[pid].player.elo for pid in self.players)


class BalancePrediction(object):
    def __init__(self, team_a, team_b):
        self.team_a_stats = TeamStats(team_a)
        self.team_b_stats = TeamStats(team_b)

    def balance_indicator(self, player_stats_dict):
        team_a_skill_sum = self.team_a_stats.combined_skill_rating(player_stats_dict)
        team_b_skill_sum = self.team_b_stats.combined_skill_rating(player_stats_dict)
        return (1.0 * team_b_skill_sum)/team_a_skill_sum

    def balance_distance(self, player_stats_dict):
        distance = (1.0 - self.balance_indicator(player_stats_dict))
        return distance


def nchoosek(n, r):
        r = min(r, n - r)
        if r == 0:
            return 1
        numerator = reduce(operator.mul, xrange(n, n - r, -1))
        denominator = reduce(operator.mul, xrange(1, r + 1))
        return numerator // denominator


def balance_players_by_skill_variance(players, verbose=True, prune_search_space=True, max_results=None):
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
        balance_distance = balance_prediction.balance_distance(player_stats)
        abs_balance_distance = abs(balance_distance)
        results.add_item((abs_balance_distance, balance_prediction, teams))
        if verbose:
            combo_desc = str(i+1).ljust(max_iteration_digits, " ")
            print "Combo %s : Team A: %s | Team B: %s | outcome: %.4f" % (combo_desc, teams[0], teams[1], balance_distance)

    # TODO: this step seems heavyweight if we are to return multiple results. review.
    # convert it back into a list of players
    result_teams = []
    for result in results.nsmallest():
        teams_as_players = []
        (abs_balance_distance, balance_prediction, teams) = result
        for team in teams:
            teams_as_players.append(tuple(player_stats[pid].player for pid in team))
        result_teams.append(tuple(teams_as_players))
    return result_teams

# end unstak
#----------------------------------------------------------------------------------------------------------------------------------------
