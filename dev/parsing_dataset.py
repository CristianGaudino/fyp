import csv

def parseFixtures(file_path):
    file = open(file_path, "r")

    for line in file:
        line = line.split(",")
        line.pop(-1)
        line.pop(12)
        line.pop(11)
        line.pop(10)
        line.pop(9)
        line.pop(7)
        line.pop(5)
        line.pop(3)
        line.pop(2)
        line.pop(1)
        line.pop(0)

        if line[1][0] != "S":
            home_goals = int(line[1][0])
            away_goals = int(line[1][-1])

            line[1] = "%d - %d" % (home_goals, away_goals)

            if home_goals > away_goals:
                result = "W"
            elif home_goals < away_goals:
                result = "L"
            else:
                result = "D"
        else:
            result = "Result"

        line.append(result)

    return file
