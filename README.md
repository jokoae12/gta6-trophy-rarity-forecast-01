# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jokoae12/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jokoae12/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An earners/players decomposition forecaster for Grand Theft Auto VI's
PlayStation 5 trophy earn rates on Exophase. A printed rate is a ratio:
the program collapses the case's history to its distinct Exophase
snapshots, turns rates back into earner counts, fits a damped growth
rate to the player count and to each trophy's earners separately
(each shrunk toward the total trophies-earned growth), and divides the
two forecasts to get each trophy's median rate at the target, with a
logit-scale spread that widens with the horizon, the series' residual
scatter and the dilution the player forecast implies. A history with a
single distinct snapshot has seen no refresh yet, so that branch
centres below the last rate with a wide spread instead of persisting a
stale page. Python 3 standard library only, no network, deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` case in the input directory and writes
`<id>.out` for each: five quantiles (5%, 25%, 50%, 75%, 95%) of every
trophy's earn rate at the case's target.

---

<sub>Entered on [make.fun](https://make.fun/?ref=wwpam0xj&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
