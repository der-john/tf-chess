VIRTUAL_ENV ?= .env
HDF_FILES := $(shell find 'netz/game-files' | sed 's/\.pgn$$/.hdf5/g' | grep '\.hdf5$$' | uniq )

.PHONY: play
play: netz/model.tfc $(VIRTUAL_ENV)
	. $(VIRTUAL_ENV)/bin/activate && cd netz && python play.py

netz/model.tfc: $(HDF_FILES) $(VIRTUAL_ENV)
	. $(VIRTUAL_ENV)/bin/activate && cd netz && python fast_trainer.py

netz/game-files/%.hdf5: netz/game-files/%.pgn
	. $(VIRTUAL_ENV)/bin/activate && python netz/parse_games.py $<

$(VIRTUAL_ENV):
	python -m venv $(VIRTUAL_ENV)
	$(VIRTUAL_ENV)/bin/python -m pip install -r requirements.txt
