# Data Dictionary

## Raw / Aggregated Variables

| Variable | Description | Type | Notes |
|----------|-------------|------|-------|
| YearMonth | Monthly period of observation | datetime | Aggregated at month level |
| Reporting_Airline | Airline carrier code | categorical | Example: DL, AA, UA |
| Origin | Origin airport code | categorical | Example: JFK, ATL, LAX |
| total_scheduled_flights | Total flights scheduled for the airline-airport-month | integer | Aggregated count |
| cancelled_flights | Number of cancelled flights | integer | Aggregated count |
| diverted_flights | Number of diverted flights | integer | Aggregated count |
| disrupted_flights | Number of disrupted flights | integer | Includes delayed, cancelled, or diverted flights |
| pct_disrupted | Proportion of disrupted flights | float | disrupted_flights / total_scheduled_flights |
| pct_cancelled | Proportion of cancelled flights | float | cancelled_flights / total_scheduled_flights |
| pct_diverted | Proportion of diverted flights | float | diverted_flights / total_scheduled_flights |
| avg_dep_delay | Average departure delay in minutes | float | Monthly average |
| avg_arr_delay | Average arrival delay in minutes | float | Monthly average |

## Engineered Features

| Variable | Description | Type | Notes |
|----------|-------------|------|-------|
| lag_1_pct_disrupted | Previous month disruption rate | float | Grouped by airline and airport |
| lag_2_pct_disrupted | Disruption rate two months prior | float | Grouped by airline and airport |
| lag_3_pct_disrupted | Disruption rate three months prior | float | Grouped by airline and airport |
| rolling_3_pct_disrupted | Rolling 3-month average disruption rate | float | Uses previous months only |
| rolling_3_std_pct_disrupted | Rolling 3-month standard deviation of disruption rate | float | Measures instability |
| delta_pct_disrupted | Month-to-month change in disruption rate | float | lag_1 - lag_2 |
| delta2_pct_disrupted | Acceleration in disruption trend | float | Second-order change |
| lag_1_total_scheduled_flights | Previous month scheduled flight volume | float | Grouped by airline and airport |
| rolling_3_total_scheduled_flights | Rolling 3-month average flight volume | float | Uses previous months only |
| interaction_disruption_delay | Interaction between disruption and delay | float | Engineered feature |
| interaction_volume_delay | Interaction between flight volume and delay | float | Engineered feature |

## Target Variable

| Variable | Description | Type | Notes |
|----------|-------------|------|-------|
| HighDisruptionMonth | Binary target indicating whether monthly disruption exceeded threshold | binary | 1 = high disruption month, 0 = otherwise |