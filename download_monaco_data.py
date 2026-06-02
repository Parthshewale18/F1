import fastf1
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache('cache')

years = [2019, 2021, 2022, 2023, 2024, 2025]
all_data = []
failed_years = []

for year in years:
    print(f"\n{'='*40}\nLoading Monaco {year}...")
    try:
        # ── Race ─────────────────────────────────────────────
        race = fastf1.get_session(year, 'Monaco', 'R')
        race.load(telemetry=False, weather=False, messages=False)

        race_res = race.results[
            ['Abbreviation', 'TeamName', 'GridPosition', 'Position']
        ].copy()
        race_res.columns = ['Driver', 'Team', 'GridPosition', 'FinishPosition']
        race_res['GridPosition']   = pd.to_numeric(race_res['GridPosition'],   errors='coerce')
        race_res['FinishPosition'] = pd.to_numeric(race_res['FinishPosition'], errors='coerce')

        # ── Qualifying ───────────────────────────────────────
        quali = fastf1.get_session(year, 'Monaco', 'Q')
        quali.load(telemetry=False, weather=False, messages=False)

        q = quali.results[['Abbreviation', 'Q1', 'Q2', 'Q3']].copy()
        q.rename(columns={'Abbreviation': 'Driver'}, inplace=True)

        # ── Reconstruct qualifying position from Q times ─────
        # Q3 set → top 10 | Q2 set but no Q3 → positions 11-15
        # Q1 set but no Q2 → positions 16-20
        # Convert Timedelta to total seconds for comparison (NaT → NaN)
        for seg in ['Q1', 'Q2', 'Q3']:
            q[f'{seg}_sec'] = pd.to_timedelta(q[seg], errors='coerce').dt.total_seconds()

        # Assign group rank within each qualifying segment
        # Drivers with Q3 time → ranked 1-10 by Q3
        # Drivers with Q2 but no Q3 → ranked 11-15 by Q2
        # Drivers with only Q1 → ranked 16-20 by Q1
        q3_drivers  = q[q['Q3_sec'].notna()].copy()
        q2_drivers  = q[q['Q3_sec'].isna() & q['Q2_sec'].notna()].copy()
        q1_drivers  = q[q['Q2_sec'].isna() & q['Q1_sec'].notna()].copy()
        no_time     = q[q['Q1_sec'].isna()].copy()

        q3_drivers  = q3_drivers.sort_values('Q3_sec')
        q3_drivers['QualifyingPosition'] = range(1, len(q3_drivers) + 1)

        q2_drivers  = q2_drivers.sort_values('Q2_sec')
        q2_drivers['QualifyingPosition'] = range(
            len(q3_drivers) + 1,
            len(q3_drivers) + len(q2_drivers) + 1
        )

        q1_drivers  = q1_drivers.sort_values('Q1_sec')
        q1_drivers['QualifyingPosition'] = range(
            len(q3_drivers) + len(q2_drivers) + 1,
            len(q3_drivers) + len(q2_drivers) + len(q1_drivers) + 1
        )

        no_time['QualifyingPosition'] = np.nan

        quali_final = pd.concat(
            [q3_drivers, q2_drivers, q1_drivers, no_time]
        )[['Driver', 'QualifyingPosition']]

        # ── Merge race + qualifying ──────────────────────────
        merged = pd.merge(race_res, quali_final, on='Driver', how='left')
        merged['Year'] = year

        # ── Fill missing GridPosition from QualifyingPosition ─
        # (Monaco rarely has grid penalties, so this is a safe fallback)
        missing_grid = merged['GridPosition'].isna()
        if missing_grid.any():
            n = missing_grid.sum()
            print(f"  Filling {n} missing GridPosition(s) from QualifyingPosition")
            merged.loc[missing_grid, 'GridPosition'] = \
                merged.loc[missing_grid, 'QualifyingPosition']

        merged = merged[[
            'Driver', 'Team', 'GridPosition',
            'FinishPosition', 'QualifyingPosition', 'Year'
        ]]

        print(f"{len(merged)} drivers loaded | NaNs: {merged.isna().sum().sum()}")
        all_data.append(merged)

    except Exception as e:
        print(f"FAILED for {year}: {e}")
        failed_years.append(year)

# ── Final assembly ────────────────────────────────────────
if not all_data:
    print("\nNo data loaded at all. Check your internet/cache and FastF1 version.")
else:
    final_df = pd.concat(all_data, ignore_index=True)

    for col in ['GridPosition', 'FinishPosition', 'QualifyingPosition']:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

    print(f"\nDataset ready: {len(final_df)} rows across {final_df['Year'].nunique()} years")
    print(f"   NaN summary:\n{final_df.isnull().sum()}")

    if failed_years:
        print(f"\n  These years failed and are NOT in the CSV: {failed_years}")

    final_df.to_csv('monaco_data_clean.csv', index=False)
    print("\nSaved → monaco_data_clean.csv")
    print(final_df.to_string(index=False))