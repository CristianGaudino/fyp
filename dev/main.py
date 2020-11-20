from dev.parsing_dataset import *

parsed_fixtures = parseFixtures("../datasets/fbref_2019-2020_prem_fixtures.csv")

for line in parsed_fixtures:
    print(line)
