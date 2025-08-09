.PHONY: venv install train infer

venv:
	python -m venv venv
	. venv/bin/activate; python -m pip install -U pip

install: venv
	. venv/bin/activate; pip install -r requirements.txt

train:
	. venv/bin/activate; python scripts/train.py --config configs/train.yaml

infer:
	. venv/bin/activate; python scripts/infer.py
