# Scenario data

Each day is viewed as a scenario that contains 24 steps. The scenarios in 2019 and 2020 are used for
training, and scenarios in 2021 form the test set.

Each scenario is composed of two columns: price $\rho$ and PV generation $P^{PV}$. The unit of price is $/kWh and the PV generation's unit is kW.

- In BSON files, each scenario is represented by a `DataFrame`.
- In a npy file, there is a 3-dim array `data`, and `data[:, :, i]` (a $24\times 2$ matrix) denotes
  the `i`-th scenario.
