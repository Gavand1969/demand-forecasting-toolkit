.PHONY: install install-dev test bench clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

bench:
	python src/benchmark.py

clean:
	rm -rf reports/charts/*.png reports/*.csv data/demand_panel.csv
