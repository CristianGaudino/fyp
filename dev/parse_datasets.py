import pandas as pd

df = pd.read_csv("../datasets/updated_fbref_2019-2020_prem_fixtures.csv")

#print(df.head())

#print(df.describe(include="all"))

#features = ["Home", "Away", "Result"]
#df.drop("Wk", axis=1, inplace=True)
#rest_of_df = df[features]

#print(rest_of_df.dtypes)


# Create a method to run all of the get methods.
# Remember that we need do minus the home from away team on their stats etc.
# The final dataset should contain more info, such as date of game and final score, but then in pandas we'll drop these.

def getSquadStats(team_name):
    # Given the team name, return a csv style string of relevant squad statistics, under specific headings

    path = "../datasets/fbref_2019-2020_prem_squad_stats.csv"

    if team_name == "Home" or team_name == "Away":
        with open(path) as f:
            line = f.readline()
            line = line.split(",")
            headings = [line[3], line[7], line[8], line[11], line[12], line[13], line[14]]
            headings[-1] = "AstPer90"
            headings[-2] = "GlsPer90"
            headings = ','.join(headings)   # Convert from list back to string
            headings += "\n"

            return headings
    else:
        file = open(path, "r")
        for line in file:
            line = line[:-1]    # Remove newline character
            line = line.split(",")
            if line[0] == team_name:
                team_line = [line[3], line[7], line[8], line[11], line[12], line[13], line[14]]
                team_line = ','.join(team_line)   # Convert from list back to string
                team_line += "\n"

                return team_line

def getSquadShootingStats(team_name):
    # Given the team name, return a csv style string of relevant squad shooting statistics, under specific headings

    path = "../datasets/fbref_2019-2020_prem_squad_shooting.csv"

    if team_name == "Home" or team_name == "Away":
        with open(path) as f:
            line = f.readline()
            line = line.split(",")
            headings = [line[4], line[5], line[6], line[7], line[8], line[9], line[10]]
            headings = ','.join(headings)   # Convert from list back to string
            headings += "\n"

            return headings
    else:
        file = open(path, "r")
        for line in file:
            line = line[:-1]    # Remove newline character
            line = line.split(",")
            if line[0] == team_name:
                team_line = [line[4], line[5], line[6], line[7], line[8], line[9], line[10]]
                team_line = ','.join(team_line)   # Convert from list back to string
                team_line += "\n"

                return team_line

def getSquadGoalkeepingStats(team_name):
    # Given the team name, return a csv style string of relevant squad goalkeeping statistics, under specific headings

    path = "../datasets/fbref_2019-2020_prem_squad_goalkeeping.csv"

    if team_name == "Home" or team_name == "Away":
        with open(path) as f:
            line = f.readline()
            line = line.split(",")
            headings = [line[5], line[6], line[7], line[8], line[9], line[13], line[14]]
            headings = ','.join(headings)   # Convert from list back to string
            headings += "\n"

            return headings
    else:
        file = open(path, "r")
        for line in file:
            line = line[:-1]    # Remove newline character
            line = line.split(",")
            if line[0] == team_name:
                team_line = [line[5], line[6], line[7], line[8], line[9], line[13], line[14]]
                team_line = ','.join(team_line)   # Convert from list back to string
                team_line += "\n"

                return team_line

def addResult(file_path):
    # Adds the result column to the fixtures dataset, writes to new file so as to not alter original

    file = open(file_path, "r")

    for line in file:
        line = line[:-1]
        print(line)
        line = line.split(",")
        if line[6][0] != "S":
            home_goals = int(line[6][0])
            away_goals = int(line[6][-1])

            line[6] = "%d - %d" % (home_goals, away_goals)

            if home_goals > away_goals:
                result = "W"
            elif home_goals < away_goals:
                result = "L"
            else:
                result = "D"
        else:
            result = "Result"

        line.append(result)
        line = ','.join(line)
        line += "\n"
        new_file = open("../datasets/updated_fbref_2019-2020_prem_fixtures.csv", "a")
        new_file.write(line)


if __name__ == "__main__":
    print("\n--- Squad Stats ---\n")
    print(getSquadStats("Home"))
    print(getSquadStats("Arsenal"))

    print("\n--- Squad Shooting Stats ---\n")
    print(getSquadShootingStats("Home"))
    print(getSquadShootingStats("Liverpool"))

    print("\n--- Squad Goalkeeping Stats ---\n")
    print(getSquadGoalkeepingStats("Home"))
    print(getSquadGoalkeepingStats("Liverpool"))
