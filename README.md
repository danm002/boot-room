[README.md](https://github.com/user-attachments/files/31596268/README.md)
# The Boot Room

A Streamlit football probability and value-bet analysis dashboard using football-data.org for fixtures and historical results.

## Important modelling choice

**xG is entered manually.** The app does not turn goals scored/conceded into fake xG because the selected data source does not provide fixture-level xG.

Historical results are displayed as context only and do not silently modify the manual xG inputs.

## Supported competitions

- Premier League (`PL`)
- League Cup (`FLC`)
- FA Cup (`FAC`)
- Champions League (`CL`)

## Probability model

The app uses a Poisson score model with a conservative Dixon-Coles correction for the four low-scoring cells (0-0, 0-1, 1-0 and 1-1).

Markets produced from the score matrix:

- Home / Draw / Away
- Over / Under 2.5 goals
- BTTS Yes / No
- Most likely scorelines

Cards are a separate Poisson calculation based on a manually entered expected total cards figure.

## Betting calculations

For a complete market, inverse decimal odds are normalised to remove the bookmaker overround:

`de-vig probability = (1 / odds) / sum(1 / all market odds)`

Edge:

`model probability - de-vig market probability`

Expected value:

`model probability × decimal odds - 1`

A selection is labelled **VALUE** only when it meets both the configured edge and EV thresholds.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `FOOTBALL_DATA_KEY` using Streamlit secrets or an environment variable.
