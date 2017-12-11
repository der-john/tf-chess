# tf-chess


A very simple Chess AI Generator based on https://github.com/erikbern/deep-pink and (to less extent) on https://github.com/elc1798/chessley-tan. It's written with tensorflow. When it works better, I'd like the AI to adjust its playing strength to the human.

But for now, speed and accuracy are tough challenges... Feel free to point out any improvements.

In order to play against the AI, you can simply run:

    make play

In order to improve the model, you can get `.pgn` files from <http://www.ficsgames.org/download.html>. Simply put them into `netz/game-files/` and run `make play` again. Note that updating the model will take a lot of time, RAM, and CPU.

- Create and activate a `venv` with python3.
- `cd netz`
- `python parse_games.py`

To do:
- [ ] Add pretty frontend ?
- [ ] Save model in checkpoints ?
