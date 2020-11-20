import pandas as pd
import numpy as np

#df = pd.read_csv("../datasets/updated_fbref_2019-2020_prem_fixtures.csv")

#print(df.head())

#print(df.describe(include="all"))

#features = ["Home", "Away", "Result"]
#df.drop("Wk", axis=1, inplace=True)
#rest_of_df = df[features]

#print(rest_of_df.dtypes)

# The final dataset should contain more info, such as date of game and final score, but then in pandas we'll drop these.

# In the top method:    Headings only gets called once at the start.

def parseDatasets():
    # Reads in a fixtures file line by line and appends extra data from other datasets.
    # Outputs the results to a new file "../datasets/premier_league_2019-2020_fixtures.csv"
    path_fixtures = "../datasets/fbref_2019-2020_prem_fixtures.csv"
    output_file = "../datasets/premier_league_2019-2020_fixtures.csv"

    fixtures_indices = [2, 4, 6, 8]

    file = open(path_fixtures, "r")

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
            stats = calculateStats(line[1], line[3])
            line = line + stats

        else:
            result = "Result"
            headings = getStats("Headings")
            line.append(result)
            line = line + headings

        line = ','.join(line)
        line += "\n"
        print(line)
        new_file = open(output_file, "a")
        new_file.write(line)


def calculateStats(home_team, away_team):
    home_stats = getStats(home_team)
    away_stats = getStats(away_team)

    final_stats = []

    for i in range(len(home_stats)):
        result = float(home_stats[i]) - float(away_stats[i])
        final_stats.append("%.6f" % result)

    return final_stats


def getStats(team_name):

    path_standings = "../datasets/fbref_2019-2020_prem_table.csv"
    standings_indices = [3, 4, 5, 9]

    path_squad = "../datasets/fbref_2019-2020_prem_squad_stats.csv"
    squad_indices = [3, 7, 8, 11, 12, 13, 14]

    path_shooting = "../datasets/fbref_2019-2020_prem_squad_shooting.csv"
    shooting_indices = [4, 5, 6, 7, 8, 9, 10]

    path_goalkeeping = "../datasets/fbref_2019-2020_prem_squad_goalkeeping.csv"
    goalkeeping_indices = [5, 6, 7, 8, 9, 13, 14]

    path_passing = "../datasets/fbref_2019-2020_prem_squad_passing.csv"
    passing_indices = [5, 20, 21, 24]

    path_shot_creation = "../datasets/fbref_2019-2020_prem_squad_shot_creation.csv"
    shot_creation_indices = [3, 4, 5, 7, 8, 10, 11, 12, 13, 15, 16, 18]

    path_defensive = "../datasets/fbref_2019-2020_prem_squad_defense.csv"
    defensive_indices = [4, 18, 22, 25]

    path_possession = "../datasets/fbref_2019-2020_prem_squad_possession.csv"
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
        with open(path) as f:
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
        file = open(path, "r")
        for line in file:
            line = line[:-1]    # Remove newline character
            line = line.split(",")
            if line[0] == team_name or line[1] == team_name:
                team_line = [line[i] for i in indices]

                if team_line[-1] == "\n":    # Remove new line character
                    team_line = team_line[:-1]

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


if __name__ == "__main__":

    #print("\n--- Stats ---\n")
    #getStats("Headings")
    #getStats("Liverpool")

    #print(calculateStats("Aston Villa", "Newcastle Utd"))
    parseDatasets()
