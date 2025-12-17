# Options Research Project Makefile
# For Unix/Mac systems

.PHONY: help status status-push test backtest

help:
	@echo "Available targets:"
	@echo "  status       - Generate status report files"
	@echo "  status-push  - Generate status and commit/push to GitHub"
	@echo "  backtest     - Run the baseline backtest"
	@echo "  test         - Run unit tests"

# Generate status reports
status:
	python src/status_report.py

# Generate status and push to GitHub
status-push: status
	git add reports/STATUS.md reports/status.json
	git commit -m "chore: update status report"
	git push

# Run baseline backtest
backtest:
	python -c "from src.backtest.engine import run_baseline_backtest; run_baseline_backtest()"

# Run tests (when implemented)
test:
	python -m pytest tests/ -v
