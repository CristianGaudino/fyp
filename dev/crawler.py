import requests
from bs4 import BeautifulSoup
import re

def getFixtureURL(home_team, away_team):
    fixtures_url = 'https://fbref.com/en/comps/9/3232/schedule/2019-2020-Premier-League-Scores-and-Fixtures'

    html = requests.get(fixtures_url).content    # Get the raw html text from the web page
    # print(html)

    soup = BeautifulSoup(html, 'lxml')  # Parse the HTML as a string

    table = soup.find_all('table')[0]  # Grab the first table from the html text

    rows = []
    row_marker = 0
    for row in table.find_all('tr'):
        column = []
        column_marker = 0
        columns = row.find_all('td')
        for col in columns:
            column.append(col)
            column_marker += 1
        rows.append(column)
        row_marker += 1

    # Rows is now a list of rows for each fixture

    for i in range(len(rows)):
        for a in range(len(rows[i])):
            # Home Team
            target = str(rows[i][3]).split('">')    # Split the row to make the regular expression easier
            if len(target) > 2:
                target = re.search('(.*)</a>', target[-1])  # Regular expression to remove unwanted characters
                target_home = target.group(1)

                # Away Team
                target = str(rows[i][7]).split('">')  # Split the row to make the regular expression easier
                if len(target) > 2:
                    target = re.search('(.*)</a>', target[-1])  # Regular expression to remove unwanted characters
                    target_away = target.group(1)

                    if target_home == home_team and target_away == away_team:
                        match_url = str(rows[i][-2])[:-23]   # Get url from the row, convert to string and remove the final characters
                        match_url = match_url.split(" ")
                        match_url = match_url[-1][6:]
                        match_url = "https://fbref.com" + match_url
                        return match_url
            break


def getLineups(home_team, away_team):
    match_url = getFixtureURL(home_team, away_team)

    html = requests.get(match_url).content  # Get the raw html text from the web page
    # print(html)

    players = []

    soup = BeautifulSoup(html, 'lxml')  # Parse the HTML as a string
    for i in range(2):
        table = soup.find_all('table')[i]  # Grab the first table from the html text, the home team's lineup

        rows = []
        row_marker = 0
        for row in table.find_all('tr'):
            column = []
            column_marker = 0
            columns = row.find_all('td')
            for col in columns:
                column.append(col)
                column_marker += 1
            rows.append(column)
            row_marker += 1

        rows = rows[1:-10]   # Remove unnecessary info
        for i in range(len(rows)):
            player = rows[i][1]
            result = re.search('">(.*)</a>', str(player))   # Use a regular expression to get the player name from the table row
            players.append(result.group(1))     # Append the player to the players list

    return players


if __name__ == "__main__":
    lineups = getLineups("Liverpool", "Chelsea")
    print(lineups)
