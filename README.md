# tf-chess


A very simple Chess AI Generator based on https://github.com/erikbern/deep-pink, written with tensorflow. When it works better, I'd like the AI to adjust its playing strength to the human.

But for now, speed and accuracy are tough challenges... Feel free to point out any improvements.

- In order to play against an AI, do the following:
`source bin/activate`
`cd netz`
`python play.py`

- In order to train a model on existing `.hdf5` files:
`source bin/activate`
`cd netz`
`python trainer.py`

- In order to create appropriate `.hdf5` files:
Download `.pgn` files from `http://www.ficsgames.org/download.html` and save them in `netz/game-files`.
`source p2bin/activate`
`cd netz`
`python parse_games.py`
