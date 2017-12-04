# tf-chess


A very simple Chess AI Generator based on https://github.com/erikbern/deep-pink and (to less extent) on https://github.com/elc1798/chessley-tan. It's written with tensorflow. When it works better, I'd like the AI to adjust its playing strength to the human.

But for now, speed and accuracy are tough challenges... Feel free to point out any improvements.

In order to play against my AI, do the following:
- Create and activate a `venv` with python3.
- Install the modules in `requirements.txt` with `pip`.
- `cd netz`
- `python play.py`

In order to train a model on existing `.hdf5` files:
- Create and activate a `venv` with python3.
- Install the modules in `requirements.txt` with `pip`.
- `cd netz`
- `python fast_trainer.py`

In order to create appropriate `.hdf5` files:
- Download `.pgn` files from `http://www.ficsgames.org/download.html` and save them in `netz/game-files`.
- `source p2bin/activate`
- `cd netz`
- `python parse_games.py`

To do:
[ ] Make `play` really stop after checkmate.
[ ] Add pretty frontend ?
[ ] Save model in checkpoints.