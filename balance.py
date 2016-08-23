# minqlx - A Quake Live server administrator bot.
# Copyright (C) 2015 Mino <mino@minomino.org>

# This file is part of minqlx.

# minqlx is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# minqlx is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with minqlx. If not, see <http://www.gnu.org/licenses/>.

import minqlx
import requests
import itertools
import threading
import random
import time

from minqlx.database import Redis

RATING_KEY = "minqlx:players:{0}:ratings:{1}" # 0 == steam_id, 1 == short gametype.
MAX_ATTEMPTS = 3
CACHE_EXPIRE = 60*30 # 30 minutes TTL.
DEFAULT_RATING = 1200
SUPPORTED_GAMETYPES = ("ca", "ctf", "dom", "ft", "tdm")
# Externally supported game types. Used by !getrating for game types the API works with.
EXT_SUPPORTED_GAMETYPES = ("ca", "ctf", "dom", "ft", "tdm", "duel", "ffa")

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

    def get_desc(self):
        raise NotImplementedError


def generate_match_prediction(team_a_baked, team_b_baked):
    assert isinstance(team_a_baked, SingleTeamBakedStats)
    assert isinstance(team_b_baked, SingleTeamBakedStats)
    prediction = MatchPrediction()
    prediction.team_a = team_a_baked
    prediction.team_b = team_b_baked
    prediction.bias = (1.0 * team_b_baked.skill_rating_sum) / team_a_baked.skill_rating_sum
    prediction.distance = (1.0 - prediction.bias)
    return prediction


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
    return "Team A: %s | Team B: %s | outcome: %.4f" % (player_ids_only(team_a), player_ids_only(team_b), match_prediction.distance)


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


def describe_switch_operation(switch_op):
    assert isinstance(switch_op, SwitchOperation)

    def get_names(player_set):
        s = ["["]
        for i, player in enumerate(sorted(list(player_set), key=lambda p: p.elo, reverse=True)):
            if i != 0:
                s.append(", ")
            s.append("%s(%d)" % (player.name, player.elo))
        s.append("]")
        return "".join(s)

    return "switch %s---> | <---%s" % (get_names(switch_op.players_moved_from_a_to_b),
                                   get_names(switch_op.players_moved_from_b_to_a))


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

class balance(minqlx.Plugin):
    database = Redis
    
    def __init__(self):
        self.add_hook("round_countdown", self.handle_round_countdown)
        self.add_hook("round_start", self.handle_round_start)
        self.add_hook("vote_ended", self.handle_vote_ended)
        self.add_command(("setrating", "setelo", "setglicko"), self.cmd_setrating, 3, usage="<id> <rating>")
        self.add_command(("getrating", "getelo", "elo", "glicko"), self.cmd_getrating, usage="<id> [gametype]")
        self.add_command(("remrating", "remelo", "remglicko"), self.cmd_remrating, 3, usage="<id>")
        self.add_command("balance", self.cmd_balance, 1)
        self.add_command("unstak", self.cmd_unstak, 1)
        self.add_command(("teams", "teens"), self.cmd_teams)
        self.add_command("do", self.cmd_do, 1)
        self.add_command(("agree", "a"), self.cmd_agree)
        self.add_command(("ratings", "elos", "selo", "sglickos"), self.cmd_ratings)

        self.ratings_lock = threading.RLock()
        # Keys: steam_id - Items: {"ffa": {"elo": 123, "games": 321, "local": False}, ...}
        self.ratings = {}
        # Keys: request_id - Items: (players, callback, channel)
        self.requests = {}
        self.request_counter = itertools.count()
        self.suggested_pair = None
        self.suggested_agree = [False, False]
        self.in_countdown = False

        self.set_cvar_once("qlx_balanceUseLocal", "1")
        self.set_cvar_once("qlx_balanceUrl", "qlstats.net:8080")
        self.set_cvar_once("qlx_balanceAuto", "1")
        self.set_cvar_once("qlx_balanceMinimumSuggestionDiff", "25")
        self.set_cvar_once("qlx_balanceApi", "elo")

        self.use_local = self.get_cvar("qlx_balanceUseLocal", bool)
        self.api_url = "http://{}/{}/".format(self.get_cvar("qlx_balanceUrl"), self.get_cvar("qlx_balanceApi"))

    def handle_round_countdown(self, *args, **kwargs):
        if all(self.suggested_agree):
            # If we don't delay the switch a bit, the round countdown sound and
            # text disappears for some weird reason.
            @minqlx.next_frame
            def f():
                self.execute_suggestion()
            f()
        
        self.in_countdown = True

    def handle_round_start(self, *args, **kwargs):
        self.in_countdown = False

    def handle_vote_ended(self, votes, vote, args, passed):
        if passed == True and vote == "shuffle" and self.get_cvar("qlx_balanceAuto", bool):
            gt = self.game.type_short
            if gt not in SUPPORTED_GAMETYPES:
                return

            @minqlx.delay(3.5)
            def f():
                players = self.teams()
                if len(players["red"] + players["blue"]) % 2 != 0:
                    self.msg("Teams were ^4NOT^7 balanced due to the total number of players being an odd number.")
                    return
                
                players = dict([(p.steam_id, gt) for p in players["red"] + players["blue"]])
                self.add_request(players, self.callback_balance, minqlx.CHAT_CHANNEL)
            f()

    @minqlx.thread
    def fetch_ratings(self, players, request_id):
        if not players:
            return

        # We don't want to modify the actual dict, so we use a copy.
        players = players.copy()

        # Get local ratings if present in DB.
        if self.use_local:
            for steam_id in players.copy():
                gt = players[steam_id]
                key = RATING_KEY.format(steam_id, gt)
                if key in self.db:
                    with self.ratings_lock:
                        if steam_id in self.ratings:
                            self.ratings[steam_id][gt] = {"games": -1, "elo": int(self.db[key]), "local": True, "time": -1}
                        else:
                            self.ratings[steam_id] = {gt: {"games": -1, "elo": int(self.db[key]), "local": True, "time": -1}}
                    del players[steam_id]

        attempts = 0
        last_status = 0
        while attempts < MAX_ATTEMPTS:
            attempts += 1
            url = self.api_url + "+".join([str(sid) for sid in players])
            res = requests.get(url)
            last_status = res.status_code
            if res.status_code != requests.codes.ok:
                continue
            
            js = res.json()
            if "players" not in js:
                last_status = -1
                continue

            # Fill our ratings dict with the ratings we just got.
            for p in js["players"]:
                sid = int(p["steamid"])
                del p["steamid"]
                t = time.time()

                with self.ratings_lock:
                    if sid not in self.ratings:
                        self.ratings[sid] = {}
                    
                    for gt in p:
                        p[gt]["time"] = t
                        p[gt]["local"] = False
                        self.ratings[sid][gt] = p[gt]
                        
                        if sid in players and gt == players[sid]:
                            # The API gave us the game type we wanted, so we remove it.
                            del players[sid]

                    # Fill the rest of the game types the API didn't return but supports.
                    for gt in SUPPORTED_GAMETYPES:
                        if gt not in self.ratings[sid]:
                            self.ratings[sid][gt] = {"games": -1, "elo": DEFAULT_RATING, "local": False, "time": time.time()}

            # If the API didn't return all the players, we set them to the default rating.
            for sid in players:
                with self.ratings_lock:
                    if sid not in self.ratings:
                        self.ratings[sid] = {}
                    self.ratings[sid][players[sid]] = {"games": -1, "elo": DEFAULT_RATING, "local": False, "time": time.time()}

            break

        if attempts == MAX_ATTEMPTS:
            self.handle_ratings_fetched(request_id, last_status)
            return

        self.handle_ratings_fetched(request_id, requests.codes.ok)

    @minqlx.next_frame
    def handle_ratings_fetched(self, request_id, status_code):
        players, callback, channel, args = self.requests[request_id]
        del self.requests[request_id]
        if status_code != requests.codes.ok:
            # TODO: Put a couple of known errors here for more detailed feedback.
            channel.reply("ERROR {}: Failed to fetch glicko ratings.".format(status_code))
        else:
            callback(players, channel, *args)

    def add_request(self, players, callback, channel, *args):
        req = next(self.request_counter)
        self.requests[req] = players.copy(), callback, channel, args

        # Only start a new thread if we need to make an API request.
        if self.remove_cached(players):
            self.fetch_ratings(players, req)
        else:
            # All players were cached, so we tell it to go ahead and call the callbacks.
            self.handle_ratings_fetched(req, requests.codes.ok)

    def remove_cached(self, players):
        with self.ratings_lock:
            for sid in players.copy():
                gt = players[sid]
                if sid in self.ratings and gt in self.ratings[sid]:
                    t = self.ratings[sid][gt]["time"]
                    if t == -1 or time.time() < t + CACHE_EXPIRE:
                        del players[sid]

        return players

    def cmd_getrating(self, player, msg, channel):
        if len(msg) == 1:
            sid = player.steam_id
        else:
            try:
                sid = int(msg[1])
                target_player = None
                if 0 <= sid < 64:
                    target_player = self.player(sid)
                    sid = target_player.steam_id
            except ValueError:
                player.tell("Invalid ID. Use either a client ID or a SteamID64.")
                return minqlx.RET_STOP_ALL
            except minqlx.NonexistentPlayerError:
                player.tell("Invalid client ID. Use either a client ID or a SteamID64.")
                return minqlx.RET_STOP_ALL

        if len(msg) > 2:
            if msg[2].lower() in EXT_SUPPORTED_GAMETYPES:
                gt = msg[2].lower()
            else:
                player.tell("Invalid gametype. Supported gametypes: {}"
                    .format(", ".join(EXT_SUPPORTED_GAMETYPES)))
                return minqlx.RET_STOP_ALL
        else:
            gt = self.game.type_short
            if gt not in EXT_SUPPORTED_GAMETYPES:
                player.tell("This game mode is not supported by the balance plugin.")
                return minqlx.RET_STOP_ALL

        self.add_request({sid: gt}, self.callback_getrating, channel, gt)

    def callback_getrating(self, players, channel, gametype):
        sid = next(iter(players))
        player = self.player(sid)
        if player:
            name = player.name
        else:
            name = sid
        
        channel.reply("{} has a glicko rating of ^4{}^7 in {}.".format(name, self.ratings[sid][gametype]["elo"], gametype.upper()))

    def cmd_setrating(self, player, msg, channel):
        if len(msg) < 3:
            return minqlx.RET_USAGE
        
        try:
            sid = int(msg[1])
            target_player = None
            if 0 <= sid < 64:
                target_player = self.player(sid)
                sid = target_player.steam_id
        except ValueError:
            player.tell("Invalid ID. Use either a client ID or a SteamID64.")
            return minqlx.RET_STOP_ALL
        except minqlx.NonexistentPlayerError:
            player.tell("Invalid client ID. Use either a client ID or a SteamID64.")
            return minqlx.RET_STOP_ALL
        
        try:
            rating = int(msg[2])
        except ValueError:
            player.tell("Invalid rating.")
            return minqlx.RET_STOP_ALL

        if target_player:
            name = target_player.name
        else:
            name = sid
        
        gt = self.game.type_short
        self.db[RATING_KEY.format(sid, gt)] = rating

        # If we have the player cached, set the rating.
        with self.ratings_lock:
            if sid in self.ratings and gt in self.ratings[sid]:
                self.ratings[sid][gt]["elo"] = rating
                self.ratings[sid][gt]["local"] = True
                self.ratings[sid][gt]["time"] = -1

        channel.reply("{}'s {} glicko rating has been set to ^4{}^7.".format(name, gt.upper(), rating))

    def cmd_remrating(self, player, msg, channel):
        if len(msg) < 2:
            return minqlx.RET_USAGE
        
        try:
            sid = int(msg[1])
            target_player = None
            if 0 <= sid < 64:
                target_player = self.player(sid)
                sid = target_player.steam_id
        except ValueError:
            player.tell("Invalid ID. Use either a client ID or a SteamID64.")
            return minqlx.RET_STOP_ALL
        except minqlx.NonexistentPlayerError:
            player.tell("Invalid client ID. Use either a client ID or a SteamID64.")
            return minqlx.RET_STOP_ALL
        
        if target_player:
            name = target_player.name
        else:
            name = sid
        
        gt = self.game.type_short
        del self.db[RATING_KEY.format(sid, gt)]

        # If we have the player cached, remove the game type.
        with self.ratings_lock:
            if sid in self.ratings and gt in self.ratings[sid]:
                del self.ratings[sid][gt]

        channel.reply("{}'s locally set {} rating has been deleted.".format(name, gt.upper()))

    def cmd_balance(self, player, msg, channel):
        gt = self.game.type_short
        if gt not in SUPPORTED_GAMETYPES:
            player.tell("This game mode is not supported by the balance plugin.")
            return minqlx.RET_STOP_ALL

        teams = self.teams()
        if len(teams["red"] + teams["blue"]) % 2 != 0:
            player.tell("The total number of players should be an even number.")
            return minqlx.RET_STOP_ALL
        
        players = dict([(p.steam_id, gt) for p in teams["red"] + teams["blue"]])
        self.add_request(players, self.callback_balance, minqlx.CHAT_CHANNEL)

    def callback_balance(self, players, channel):
        # We check if people joined while we were requesting ratings and get them if someone did.
        teams = self.teams()
        current = teams["red"] + teams["blue"]
        gt = self.game.type_short

        for p in current:
            if p.steam_id not in players:
                d = dict([(p.steam_id, gt) for p in current])
                self.add_request(d, self.callback_balance, channel)
                return

        # Start out by evening out the number of players on each team.
        diff = len(teams["red"]) - len(teams["blue"])
        if abs(diff) > 1:
            if diff > 0:
                for i in range(diff - 1):
                    p = teams["red"].pop()
                    p.put("blue")
                    teams["blue"].append(p)
            elif diff < 0:
                for i in range(abs(diff) - 1):
                    p = teams["blue"].pop()
                    p.put("red")
                    teams["red"].append(p)

        # Start shuffling by looping through our suggestion function until
        # there are no more switches that can be done to improve teams.
        switch = self.suggest_switch(teams, gt)
        if switch:
            while switch:
                p1 = switch[0][0]
                p2 = switch[0][1]
                self.switch(p1, p2)
                teams["blue"].append(p1)
                teams["red"].append(p2)
                teams["blue"].remove(p2)
                teams["red"].remove(p1)
                switch = self.suggest_switch(teams, gt)
            avg_red = self.team_average(teams["red"], gt)
            avg_blue = self.team_average(teams["blue"], gt)
            diff_rounded = abs(round(avg_red) - round(avg_blue)) # Round individual averages.
            if round(avg_red) > round(avg_blue):
                self.msg("^1{} ^7vs ^4{}^7 - DIFFERENCE: ^1{}"
                    .format(round(avg_red), round(avg_blue), diff_rounded))
            elif round(avg_red) < round(avg_blue):
                self.msg("^1{} ^7vs ^4{}^7 - DIFFERENCE: ^4{}"
                    .format(round(avg_red), round(avg_blue), diff_rounded))
            else:
                self.msg("^1{} ^7vs ^4{}^7 - Holy shit!"
                    .format(round(avg_red), round(avg_blue)))
        else:
            channel.reply("Teams are good! Nothing to balance.")
        return True

    def cmd_unstak(self, player, msg, channel):
        gt = self.game.type_short
        if gt not in SUPPORTED_GAMETYPES:
            player.tell("This game mode is not supported by the balance plugin.")
            return minqlx.RET_STOP_ALL

        teams = self.teams()
        if len(teams["red"] + teams["blue"]) <= 2:
            player.tell("Nothing to balance.")
            return minqlx.RET_STOP_ALL

        players = dict([(p.steam_id, gt) for p in teams["red"] + teams["blue"]])
        self.add_request(players, self.callback_unstak, minqlx.CHAT_CHANNEL)

    def callback_unstak(self, players, channel):
        # We check if people joined while we were requesting ratings and get them if someone did.
        teams = self.teams()
        current = teams["red"] + teams["blue"]
        gt = self.game.type_short

        for p in current:
            if p.steam_id not in players:
                d = dict([(p.steam_id, gt) for p in current])
                self.add_request(d, self.callback_unstak, channel)
                return

        # Generate an unstak PlayerInfo list
        players_dict = {}
        for p in current:
            player_steam_id = p.steam_id
            player_name = p.clean_name
            player_elo = self.ratings[p.steam_id][gt]["elo"]
            players_dict[p.steam_id] = (player_name, player_elo, p)
        players_info = player_info_list_from_steam_id_name_ext_obj_elo_dict(players_dict)

        # do unstak balancing on player data (doesnt actually do any balancing operations)
        new_blue_team, new_red_team = balance_players_by_skill_band(players_info)

        def move_players_to_new_team(team, team_index):
            """
            Move the given players to this team.
            :param team: PlayerInfo list for one of the teams
            :param team_index: The corresponding index of team. 0 = blue, 1 = red
            :return: True if any players were moved
            """
            team_names = ["blue", "red"]
            players_moved = False
            this_team_name = team_names[team_index]
            other_team_name = team_names[1 - team_index]
            for player_info in team:
                assert isinstance(player_info, PlayerInfo)
                p = player_info.ext_obj
                assert p
                players_moved = (p.team != this_team_name) 
                p.team = this_team_name
                    
            return players_moved

        moved_players = False
        moved_players = move_players_to_new_team(new_blue_team, 0) or moved_players
        moved_players = move_players_to_new_team(new_red_team, 1) or moved_players

        if not moved_players:
            channel.reply("No one was moved.")
            return True

        self.report_team_stats(teams, gt, new_blue_team, new_red_team)
        return True

    # TODO: other reporting sites (e.g. balance/teams command) could be updated to use this (needs to use PlayerInfo)
    def report_team_stats(self, teams, gt, new_blue_team, new_red_team):
        # print some stats
        avg_red = self.team_average(teams["red"], gt)
        avg_blue = self.team_average(teams["blue"], gt)
        diff_rounded = abs(round(avg_red) - round(avg_blue))  # Round individual averages.

        def team_color(team_index):
            if team_index == 0:
                # red
                return "^1"
            elif team_index == 1:
                # blue
                return "^4"
            return ""

        def stronger_team_index(red_amount, blue_amount):
            if red_amount > blue_amount:
                return 0
            if red_amount < blue_amount:
                return 1
            return None

        round_avg_red = round(avg_red)
        round_avg_blue = round(avg_blue)
        favoured_team_colour_prefix = team_color(stronger_team_index(round_avg_red, round_avg_blue))
        avg_msg = "^1{} ^7vs ^4{}^7 - DIFFERENCE: ^{}{}".format(round_avg_red,
                                                                round_avg_blue,
                                                                favoured_team_colour_prefix,
                                                                diff_rounded)
        self.msg(avg_msg)
        # print some skill band stats
        bands_msg = []
        blue_bands = split_players_by_skill_band(new_blue_team)
        red_bands = split_players_by_skill_band(new_red_team)
        for category_name in blue_bands.keys():
            blue_players = blue_bands[category_name]
            red_players = red_bands[category_name]
            difference = abs(len(blue_players) - len(red_players))
            if difference:
                adv_team_idx = stronger_team_index(len(red_players), len(blue_players))
                bands_msg.append("{}:{}+{}".format(category_name,
                                                   team_color(adv_team_idx),
                                                   difference))
        bands_diff_content = "Balanced"
        if bands_msg:
            bands_diff_content = "^7, ".join(bands_msg)

        bands_msg = "Net skill band diff: " + bands_diff_content
        self.msg(bands_msg)

    def cmd_teams(self, player, msg, channel):
        gt = self.game.type_short
        if gt not in SUPPORTED_GAMETYPES:
            player.tell("This game mode is not supported by the balance plugin.")
            return minqlx.RET_STOP_ALL
        
        teams = self.teams()
        if len(teams["red"]) != len(teams["blue"]):
            player.tell("Both teams should have the same number of players.")
            return minqlx.RET_STOP_ALL
        
        teams = dict([(p.steam_id, gt) for p in teams["red"] + teams["blue"]])
        self.add_request(teams, self.callback_teams, channel)

    def callback_teams(self, players, channel):
        # We check if people joined while we were requesting ratings and get them if someone did.
        teams = self.teams()
        current = teams["red"] + teams["blue"]
        gt = self.game.type_short

        for p in current:
            if p.steam_id not in players:
                d = dict([(p.steam_id, gt) for p in current])
                self.add_request(d, self.callback_teams, channel)
                return

        avg_red = self.team_average(teams["red"], gt)
        avg_blue = self.team_average(teams["blue"], gt)
        switch = self.suggest_switch(teams, gt)
        diff_rounded = abs(round(avg_red) - round(avg_blue)) # Round individual averages.
        if round(avg_red) > round(avg_blue):
            channel.reply("^1{} ^7vs ^4{}^7 - DIFFERENCE: ^1{}"
                .format(round(avg_red), round(avg_blue), diff_rounded))
        elif round(avg_red) < round(avg_blue):
            channel.reply("^1{} ^7vs ^4{}^7 - DIFFERENCE: ^4{}"
                .format(round(avg_red), round(avg_blue), diff_rounded))
        else:
            channel.reply("^1{} ^7vs ^4{}^7 - Holy shit!"
                .format(round(avg_red), round(avg_blue)))

        minimum_suggestion_diff = self.get_cvar("qlx_balanceMinimumSuggestionDiff", int)
        if switch and switch[1] >= minimum_suggestion_diff:
            channel.reply("SUGGESTION: switch ^4{}^7 with ^4{}^7. Mentioned players can type ^4!a^7 to agree."
                .format(switch[0][0].clean_name, switch[0][1].clean_name))
            if not self.suggested_pair or self.suggested_pair[0] != switch[0][0] or self.suggested_pair[1] != switch[0][1]:
                self.suggested_pair = (switch[0][0], switch[0][1])
                self.suggested_agree = [False, False]
        else:
            i = random.randint(0, 99)
            if not i:
                channel.reply("Teens look ^4good!")
            else:
                channel.reply("Teams look good!")
            self.suggested_pair = None

        return True

    def cmd_do(self, player, msg, channel):
        """Forces a suggested switch to be done."""
        if self.suggested_pair:
            self.execute_suggestion()

    def cmd_agree(self, player, msg, channel):
        """After the bot suggests a switch, players in question can use this to agree to the switch."""
        if self.suggested_pair and not all(self.suggested_agree):
            p1, p2 = self.suggested_pair
            
            if p1 == player:
                self.suggested_agree[0] = True
            elif p2 == player:
                self.suggested_agree[1] = True

            if all(self.suggested_agree):
                # If the game's in progress and we're not in the round countdown, wait for next round.
                if self.game.state == "in_progress" and not self.in_countdown:
                    self.msg("The switch will be executed at the start of next round.")
                    return

                # Otherwise, switch right away.
                self.execute_suggestion()

    def cmd_ratings(self, player, msg, channel):
        gt = self.game.type_short
        if gt not in EXT_SUPPORTED_GAMETYPES:
            player.tell("This game mode is not supported by the balance plugin.")
            return minqlx.RET_STOP_ALL
        
        players = dict([(p.steam_id, gt) for p in self.players()])
        self.add_request(players, self.callback_ratings, channel)

    def callback_ratings(self, players, channel):
        # We check if people joined while we were requesting ratings and get them if someone did.
        teams = self.teams()
        current = self.players()
        gt = self.game.type_short

        for p in current:
            if p.steam_id not in players:
                d = dict([(p.steam_id, gt) for p in current])
                self.add_request(d, self.callback_ratings, channel)
                return

        if teams["free"]:
            free_sorted = sorted(teams["free"], key=lambda x: self.ratings[x.steam_id][gt]["elo"], reverse=True)
            free = ", ".join(["{}: ^4{}^7".format(p.clean_name, self.ratings[p.steam_id][gt]["elo"]) for p in free_sorted])
            channel.reply(free)
        if teams["red"]:
            red_sorted = sorted(teams["red"], key=lambda x: self.ratings[x.steam_id][gt]["elo"], reverse=True)
            red = ", ".join(["{}: ^1{}^7".format(p.clean_name, self.ratings[p.steam_id][gt]["elo"]) for p in red_sorted])
            channel.reply(red)
        if teams["blue"]:
            blue_sorted = sorted(teams["blue"], key=lambda x: self.ratings[x.steam_id][gt]["elo"], reverse=True)
            blue = ", ".join(["{}: ^4{}^7".format(p.clean_name, self.ratings[p.steam_id][gt]["elo"]) for p in blue_sorted])
            channel.reply(blue)
        if teams["spectator"]:
            spec_sorted = sorted(teams["spectator"], key=lambda x: self.ratings[x.steam_id][gt]["elo"], reverse=True)
            spec = ", ".join(["{}: {}".format(p.clean_name, self.ratings[p.steam_id][gt]["elo"]) for p in spec_sorted])
            channel.reply(spec)

    def suggest_switch(self, teams, gametype):
        """Suggest a switch based on average team ratings."""
        avg_red = self.team_average(teams["red"], gametype)
        avg_blue = self.team_average(teams["blue"], gametype)
        cur_diff = abs(avg_red - avg_blue)
        min_diff = 999999
        best_pair = None

        for red_p in teams["red"]:
            for blue_p in teams["blue"]:
                r = teams["red"].copy()
                b = teams["blue"].copy()
                b.append(red_p)
                r.remove(red_p)
                r.append(blue_p)
                b.remove(blue_p)
                avg_red = self.team_average(r, gametype)
                avg_blue = self.team_average(b, gametype)
                diff = abs(avg_red - avg_blue)
                if diff < min_diff:
                    min_diff = diff
                    best_pair = (red_p, blue_p)

        if min_diff < cur_diff:
            return (best_pair, cur_diff - min_diff)
        else:
            return None

    def team_average(self, team, gametype):
        """Calculates the average rating of a team."""
        avg = 0
        if team:
            for p in team:
                avg += self.ratings[p.steam_id][gametype]["elo"]
            avg /= len(team)

        return avg

    def execute_suggestion(self):
        p1, p2 = self.suggested_pair
        try:
            p1.update()
            p2.update()
        except minqlx.NonexistentPlayerError:
            return
        
        if p1.team != "spectator" and p2.team != "spectator":
            self.switch(self.suggested_pair[0], self.suggested_pair[1])
        
        self.suggested_pair = None
        self.suggested_agree = [False, False]
