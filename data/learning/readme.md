# PJM data for IL and RL

We have already processed the raw PJM data to suit home energy management. Further here, each day is
viewed as a *scenario*, and we split the data into many scenarios for training and testing of the
learning methods like IL and RL.

## MILP as the expert in IL

Each scenario (including the RT price and the solar generation) is stored in a `DataFrame`.
Next, we run the MILP for each scenario and write the optimal solutions into the same `DataFrame`
as separate columns.

The unit of price is $/kWh and the PV generation's unit is kW.

The collection of scenarios in each year is stored as `ss` in BSON.
