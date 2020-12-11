import os
import re
import csv
from dev.crawler import *


def parseDatasets():
    # Reads in a fixtures file line by line and appends extra data from other datasets.
    # Outputs the results to a new file "../datasets/premier_league_2019-2020_fixtures.csv"
    path_fixtures = "../datasets/prem_2015-2016/fbref_2015-2016_prem_fixtures.csv"
    output_file = "../datasets/prem_2015-2016/premier_league_2015-2016_fixtures.csv"

    if os.path.exists(output_file):
        os.remove(output_file)

    # The list of html fixtures is created here, thus it will only be created once
    fixtures_url = 'https://fbref.com/en/comps/9/1467/schedule/2015-2016-Premier-League-Scores-and-Fixtures'
    html_fixtures_array = getFixtureAsHTML(fixtures_url)

    # Build the player statistics dictionary
    players_path = "../datasets/prem_2015-2016/fbref_2015-2016_prem_players.csv"
    players_dict = playerCSVToDictionary(players_path)

    fixtures_indices = [2, 4, 6, 8]

    file = open(path_fixtures, "r", encoding="utf-8")

    for line in file:
        line = line[:-1]
        line = line.split(",")
        line = [line[i] for i in fixtures_indices]  # Remove redundant information

        if line[0] != "Date":
            home_goals = int(line[2][0])    # Get home and away goals from Score column
            away_goals = int(line[2][-1])

            line[2] = "%d - %d" % (home_goals, away_goals)  # Rewrite it to avoid an issue with a character

            if home_goals > away_goals:
                result = "W"
            elif home_goals < away_goals:
                result = "L"
            else:
                result = "D"

            line.append(result)
            stats = calculateStats(line[1], line[3], html_fixtures_array, players_dict)
            line = line + stats

        else:
            result = "Result"
            headings = getStats("Headings")
            line.append(result)
            line = line + headings
            line = line + getLineupsHeadings()  # Add the lineups headings

        line = ','.join(str(item) for item in line)
        print(line)
        line += "\n"
        new_file = open(output_file, "a", encoding="utf-8")
        new_file.write(line)


def calculateStats(home_team, away_team, fixtures_array, players_dict):
    home_stats = getStats(home_team)
    away_stats = getStats(away_team)

    lineup_indices = [5, 7, 8, 9, 14, 15]

    lineup_stats = getLineupStats(home_team, away_team, fixtures_array, players_dict, lineup_indices)

    final_stats = []

    for i in range(len(home_stats)):
        result = float(home_stats[i]) - float(away_stats[i])
        final_stats.append("%.3f" % result)

    final_stats = final_stats + lineup_stats

    return final_stats


def getStats(team_name):

    path_standings = "../datasets/prem_2015-2016/fbref_2015-2016_prem_table.csv"
    standings_indices = [3, 4, 5, 9]

    path_squad = "../datasets/prem_2015-2016/fbref_2015-2016_prem_squad_stats.csv"
    squad_indices = [3, 7, 8, 11, 12, 13, 14]

    path_shooting = "../datasets/prem_2015-2016/fbref_2015-2016_prem_squad_shooting.csv"
    shooting_indices = [4, 5, 6, 7, 8, 9, 10]

    path_goalkeeping = "../datasets/prem_2015-2016/fbref_2015-2016_prem_squad_goalkeeping.csv"
    goalkeeping_indices = [5, 6, 7, 8, 9, 13, 14]

    path_passing = "../datasets/prem_2015-2016/fbref_2015-2016_prem_squad_passing.csv"
    passing_indices = [5, 20, 21, 24]

    path_shot_creation = "../datasets/prem_2015-2016/fbref_2015-2016_prem_squad_shot_creation.csv"
    shot_creation_indices = [3, 4, 5, 7, 8, 10, 11, 12, 13, 15, 16, 18]

    path_defensive = "../datasets/prem_2015-2016/fbref_2015-2016_prem_squad_defense.csv"
    defensive_indices = [4, 18, 22, 25]

    path_possession = "../datasets/prem_2015-2016/fbref_2015-2016_prem_squad_possession.csv"
    possession_indices = [10, 22]

    standings_stats = getSpecificStats(team_name, path_standings, standings_indices)
    squad_stats = getSpecificStats(team_name, path_squad, squad_indices)
    shooting_stats = getSpecificStats(team_name, path_shooting, shooting_indices)
    goalkeeping_stats = getSpecificStats(team_name, path_goalkeeping, goalkeeping_indices)
    passing_stats = getSpecificStats(team_name, path_passing, passing_indices)
    shot_creation_stats = getSpecificStats(team_name, path_shot_creation, shot_creation_indices)
    defensive_stats = getSpecificStats(team_name, path_defensive, defensive_indices)
    possession_stats = getSpecificStats(team_name, path_possession, possession_indices)

    all_stats = standings_stats + squad_stats + shooting_stats + goalkeeping_stats + passing_stats + shot_creation_stats + defensive_stats + possession_stats

    return all_stats


def getSpecificStats(team_name, path, indices):
    # Given a team name, a path csv and a list of indices, return a csv style string of the relevant statistics.
    if team_name == "Headings":
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            line = line.split(",")
            headings = [line[i] for i in indices]

            if headings[0] == "Poss":
                headings = modifySquadHeadings(headings)
            if headings[0] == "SCA":
                headings = modifyShotCreationHeadings(headings)

            if headings[-1][-1] == "\n":    # Remove new line character
                headings[-1] = headings[-1][:-1]

            return headings
    else:
        file = open(path, "r", encoding="utf-8")
        for line in file:
            if line[-1] == "\n":
                line = line[:-1]    # Remove newline character

            line = line.split(",")
            if line[0] == team_name or line[1] == team_name:
                team_line = [line[i] for i in indices]
                team_line[:] = [0 if x == "" else x for x in team_line]

                return team_line


def modifySquadHeadings(headings):
    # Modify the names of headings
    headings[-1] += "Per90"
    headings[-2] += "Per90"
    return headings


def modifyShotCreationHeadings(headings):
    # Modify the names of headings
    for i in range (2, 6):
        headings[i] += "SCA"
    for i in range (8, 12):
        headings[i] += "GCA"

    return headings


def getLineupStats(home_team, away_team, html_fixtures_array, players_dict, lineup_indices):
    # Get the stats for both team's lineups
    gk_indices = lineup_indices[:2]
    lineups = getLineups(home_team, away_team, html_fixtures_array)  # Returns a list of 22 players, 11 for each team

    home_gk_array = []
    home_df_array = []
    home_mf_array = []
    home_fw_array = []

    away_gk_array = []
    away_df_array = []
    away_mf_array = []
    away_fw_array = []

    # Below here needs to be redone with the dictionary, the lineup indices also have to change by -2
    # Loop through the lineups list, get the dictionary values, then do the if statements for position
    # Need to handle players with same name, no worry for now

    # MINUS 2 FROM THE INDICES LISTS
    for player in lineups:
        player_stats = players_dict.get(player)
        player_stats[1] = player_stats[1][:2]   # Some players have 2 positions, main position is stored first so just take this

        if player_stats[1] == "GK":
            home_bool = True if player_stats[2] == home_team else False
            stats = [player_stats[i] for i in gk_indices]
            home_gk_array.append(stats) if home_bool else away_gk_array.append(stats)
        elif player_stats[1] == "DF":
            home_bool = True if player_stats[2] == home_team else False
            stats = [player_stats[i] for i in lineup_indices]
            home_df_array.append(stats) if home_bool else away_df_array.append(stats)
        elif player_stats[1] == "MF":
            home_bool = True if player_stats[2] == home_team else False
            stats = [player_stats[i] for i in lineup_indices]
            home_mf_array.append(stats) if home_bool else away_mf_array.append(stats)
        elif player_stats[1] == "FW":
            home_bool = True if player_stats[2] == home_team else False
            stats = [player_stats[i] for i in lineup_indices]
            home_fw_array.append(stats) if home_bool else away_fw_array.append(stats)

    home_stats = home_gk_array + home_df_array + home_mf_array + home_fw_array
    away_stats = away_gk_array + away_df_array + away_mf_array + away_fw_array
    lineup_stats = home_stats + away_stats

    lineup_stats = [str(item) for sublist in lineup_stats for item in sublist]     # Flatten the list of lists

    return lineup_stats

def getLineupsHeadings():
    # Given a  line of headings, parse it to a list of the headings wanted.
    # pos 0 & 11 are goalkeepers, rest are outfield.
    # MP_1_Home, Min_1_Home, MP_2_Home etc. 1-11 for both teams.
    gk_headings = ["MP_1", "Min_1"]
    outfield_headings = ["MP", "Min", "Gls", "Ast", "GlsPer90", "AstPer90"]

    headings = [gk_headings]

    for i in range(2, 12):
        player_headings = []
        for heading in outfield_headings:
            player_headings.append("%s_%d" % (heading, i))  # Add the heading + the player's position to headings

        headings.append(player_headings)

    home_headings = []
    away_headings = []
    for player_heading in headings:
        home_headings.append([item + "_Home" for item in player_heading])  # Append Home to the end of every heading
        away_headings.append([item + "_Away" for item in player_heading])  # Append Away to the end of every heading

    lineup_headings = home_headings + away_headings
    lineup_headings = [item for sublist in lineup_headings for item in sublist]     # Flatten the list of lists

    return lineup_headings

def playerCSVToDictionary(path):
    # Given a path to a csv, return a python dictionary
    csv_dict = {}
    for line in open(path, "r", encoding="utf-8").readlines()[1:]:  # Skip the heading line
        line = line.replace('\n', '')
        line = line.split(',')
        line[1] = re.search('(.*)\\\\', line[1]).group(1)   # Regular expression to adjust the player name field

        csv_dict[line[1]] = line[2:]    # Add the new entry to the dictionary

    return csv_dict

if __name__ == "__main__":
    #print("\n--- Stats ---\n")
    #getStats("Headings")
    #getStats("Liverpool")

    #print(calculateStats("Aston Villa", "Newcastle Utd"))

    #stats = getLineupStats("Liverpool", "Chelsea")
    #print(stats)

    #headings = getLineupsHeadings()
    #print(headings)
    parseDatasets()

    #playerCSVToDictionary("../datasets/fbref_2019-2020_prem_players.csv")

    #path_defensive = "../datasets/prem_2018-2019/fbref_2018-2019_prem_squad_defense.csv"
    #defensive_indices = [4, 18, 22, 25]
    #defensive_stats = getSpecificStats("Wolves", path_defensive, defensive_indices)
    #print(defensive_stats)
