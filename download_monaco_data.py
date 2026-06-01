import fastf1
import pandas as pd

fastf1.Cache.enable_cache('cache')  # Enable caching to speed up future runs

# Load the session data for the Monaco Grand Prix 2023
years = [2019,2021,2022,2023,2024,2025]
all_data = []

for year in years:
    print(f"Loading Monaco {year}")
    # Race Session
    race = fastf1.get_session(year, 'Monaco', 'R')
    race.load()
   # Qualification Session
    quali = fastf1.get_session(year, 'Monaco', 'Q')
    quali.load()
   # Race Results
    race_results = race.results[
        ['Abbreviation',
         'TeamName',
         'GridPosition',
         'Position']
    ].copy()
    # Qualifying Results
    quali_results = quali.results[
        ['Abbreviation', 'Position']
    ].copy()

    quali_results.rename(
        columns = {"Position" : "QualifyingPosition"},
        inplace = True
    )

    merged = pd.merge(
        race_results,
        quali_results,
        on = 'Abbreviation'
    )

    # Create an year column
    merged['Year'] = year

    #rename columns
    merged.rename(
        columns = {
        'Abbreviation': 'Driver',
        'TeamName': 'Team',
        'Position': 'FinishPosition'
        }
    )

    all_data.append(merged)

# Concatenate all the data into a single DataFrame
final_df = pd.concat(all_data)
print(final_df.head())

final_df.to_csv('monaco_data.csv', index=False)
print("Data saved to monaco_data.csv")