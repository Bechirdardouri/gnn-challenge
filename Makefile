.PHONY: help check train-sota validate-sota render-leaderboard compile

help:
	@echo "Common commands:"
	@echo "  make check              - run environment checks"
	@echo "  make train-sota         - generate SOTA baseline submission CSV"
	@echo "  make validate-sota      - validate generated submission against data/test.csv"
	@echo "  make render-leaderboard - regenerate leaderboard markdown/json artifacts"
	@echo "  make compile            - compile Python files as a quick syntax check"

check:
	python check_setup.py

train-sota:
	python starter_code/sota_graph_ensemble.py --output submissions/sota_ensemble_submission.csv

validate-sota:
	python competition/validate_submission.py submissions/sota_ensemble_submission.csv data/test.csv

render-leaderboard:
	python competition/render_leaderboard.py

compile:
	python -m compileall competition scripts encryption starter_code submissions check_setup.py
