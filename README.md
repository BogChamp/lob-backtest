# Lob-backtest

Code for backtesting MM strategies. Algorithm explanation here: https://app.diagrams.net/#G19Ys3Yt7-e_gShX8aNkQzAYD70ox6wMYJ

## To install all dependencies

pip install -e .

## Structure of this repo

```bash
├── lobio
│   ├── accounting
│   │   └── pnl_counter.py
│   ├── lob
│   │   ├── limit_order.py
│   │   ├── order_book.py
│   │   └── price_level.py
│   ├── simulator.py
│   ├── strategies
│   │   ├── avellaneda_stoikov_model.py
│   │   └── base_model.py
│   ├── stream
│   │   └── run_streams.py
│   └── utils
│       ├── check_compress_data.py
│       ├── data_preparation.py
│       └── utils.py
├── notebooks
│   ├── ac.ipynb
│   ├── backtest.ipynb
│   ├── data_prep.ipynb
│   ├── exp_pick.ipynb
│   └── research.ipynb
├── queue_dynamic
│   ├── a2c_runs.py
│   ├── find_price_levels.py
│   ├── losses.py
│   ├── models
│   │   └── models.py
│   ├── reinforce_runs.py
│   └── simulation.py
├── setup.py
```
## How to load data

1) To download data from the exchange run 
```
python3 lobio/stream/run_streams.py
```

2) To prepare and preprocess data, run following commands:
```
python3 lobio/utils/check_compress_data.py
python3 lobio/utils/data_preparation.py
```
## Repo description

* [`lobio`](lobio) — folder contains code for data downloading, strategy, limit order book structure and accounting implementation. 

* [`notebooks`](notebooks) — folder contains all python notebooks with experimental code.

* [`queue_dynamic`](queue_dynamic) — folder contains code for price level queue dynamic and reinforcement agent training.
