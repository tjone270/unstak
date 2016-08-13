#----------------------------------------------------------------------------------------------------------------------------------------
# unstak, an alternative balancing method for minqlx created by github/hyperwired aka "stakz", 2016-07-31
# This plugin is released to everyone, for any purpose. It comes with no warranty, no guarantee it works, it's released AS IS.
# You can modify everything, except for lines 1-4. They're there to indicate I whacked this together originally. Please make it better :D

import collections
import hashlib
import math
import random

from player_info import PlayerInfo

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


def balance_players_by_skill_variance(players):
    players = sort_by_skill_rating_descending(players)
    # We assume that the skill rating is based on a normal distribution in the global player population.
    # The input players is a sample of the player base. For the purposes of this algorithm, we treat the
    # sample as the population, since we are not trying to make inferences on the population. We also
    # assume a normal distribution in the sample to find outlier players, which should be true in most scenarios.
    player_stats = collections.OrderedDict()
    for p in players:
        player_stats[p.steam_id] = PlayerStats(p)

    # for each player, determine their distance in standard deviations from the sample mean.
    sample_mean = calc_mean(skill_rating_list(players))
    sample_stdev = calc_standard_deviation(skill_rating_list(players), mean=sample_mean)


    print "sample mean: %.2f" % sample_mean
    print "sample stdev: %.2f" % sample_stdev

    for player_stat in player_stats.values():
        assert isinstance(player_stat, PlayerStats)
        player = player_stat.player
        assert isinstance(player, PlayerInfo)
        player_stat.relative_deviation = 1.0 * ((player_stat.player.elo - sample_mean) / (sample_stdev * 1.0))
        print "%s: skill=%s stdev=%.2f" % (player.name, player.elo, player_stat.relative_deviation)

    # TODO: actual algo
    # outline:
    # do a brute force space search by:
    # search space reduction:
        # generating combinations (n choose n/2) per standard deviation
        # now do a search through the cartesian product of 1 pick from each stdev combo
        # picks where the distance between left and right teams is more than 2 players per stdev can be ommitted.
    # define and use a set of heuristics to classify the output of each possible teams pick.
    # only maintain the top N matches in memory.

# end unstak
#----------------------------------------------------------------------------------------------------------------------------------------
