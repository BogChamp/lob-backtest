# lob-backtest

Code for backtesting MM strategies. Algorithm explanation here: https://app.diagrams.net/#G19Ys3Yt7-e_gShX8aNkQzAYD70ox6wMYJ

Data contains initial orderbook snapshot, sequential trades and incremental orderbook updates.

Files:
1) stream_diffs.py - to read ob updates from exchange
2) stream_trades.py - to read trades from exchange
3) stream_lobs.py - to read ob states from exchange
4) get_lob.py - get 1 ob state
5) backtest.py - run backtest
