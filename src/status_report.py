"""
Status Report Generator

Generates STATUS.md and status.json for GitHub visibility.
ChatGPT can read these files to understand current project state.
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yml"
STATE_PATH = ROOT / "state.json"
REPORT_DIR = ROOT / "reports"


def get_git_commit():
    """Get current git commit hash and short message."""
    try:
        # Get short hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT
        )
        commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else None

        # Get commit message (first line)
        msg_result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            capture_output=True, text=True, cwd=ROOT
        )
        commit_msg = msg_result.stdout.strip() if msg_result.returncode == 0 else None

        # Get branch name
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=ROOT
        )
        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None

        return {
            "hash": commit_hash,
            "message": commit_msg,
            "branch": branch
        }
    except Exception:
        return {"hash": None, "message": None, "branch": None}


def load_config():
    """Load project configuration."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    return {}


def load_state():
    """Load pipeline state."""
    if STATE_PATH.exists():
        with open(STATE_PATH, 'r') as f:
            return json.load(f)
    return {"last_run_utc": None, "version": 1}


def find_latest_backtest():
    """Find most recent backtest results."""
    results_dir = ROOT / "results"
    if not results_dir.exists():
        return None

    # Look for summary.json files
    summaries = list(results_dir.glob("**/summary.json"))
    if not summaries:
        return None

    # Get most recent by modification time
    latest = max(summaries, key=lambda p: p.stat().st_mtime)
    try:
        with open(latest, 'r') as f:
            data = json.load(f)
        data["summary_path"] = str(latest.relative_to(ROOT))
        return data
    except Exception:
        return None


def get_backtest_modules():
    """Check which backtest modules exist."""
    backtest_dir = ROOT / "src" / "backtest"
    modules = {}

    expected = ["config.py", "strategy.py", "costs.py", "risk.py", "engine.py"]
    for mod in expected:
        mod_path = backtest_dir / mod
        modules[mod.replace(".py", "")] = mod_path.exists()

    return modules


def render_markdown(config, state, backtest_summary, modules, git_info=None):
    """Generate STATUS.md content."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = []
    lines.append("# Project Status")
    lines.append("")
    lines.append(f"**Project:** {config.get('project', 'dissertation-options-beta-neutral')}")
    lines.append(f"**Last update (UTC):** {now}")
    lines.append(f"**Pipeline last run:** {state.get('last_run_utc', '—')}")
    if git_info and git_info.get("hash"):
        lines.append(f"**Git:** `{git_info['hash']}` ({git_info.get('branch', 'unknown')}) - {git_info.get('message', '')}")
    lines.append("")

    # Data summary
    lines.append("## Data Pipeline")
    lines.append("")
    data_sum = state.get("data_summary", {})
    if data_sum:
        lines.append("| Dataset | Rows | Location |")
        lines.append("|---------|------|----------|")
        for name, info in data_sum.items():
            rows = info.get("rows", "—")
            loc = info.get("location", "—")
            lines.append(f"| {name} | {rows:,} | `{loc}` |")
    lines.append("")

    # Available data periods
    years = state.get("years_available", [])
    months = state.get("months_by_year", {})
    lines.append("## Available Data Periods")
    lines.append("")
    for year in years:
        month_list = months.get(str(year), [])
        month_names = [["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1] for m in month_list]
        lines.append(f"- **{year}:** {', '.join(month_names)}")
    lines.append("")

    # Backtest modules
    lines.append("## Backtest Framework")
    lines.append("")
    lines.append("| Module | Status |")
    lines.append("|--------|--------|")
    for mod, exists in modules.items():
        status = "✅ Ready" if exists else "❌ Missing"
        lines.append(f"| {mod} | {status} |")
    lines.append("")

    # Latest backtest results
    if backtest_summary:
        lines.append("## Latest Backtest Results")
        lines.append("")
        for key in ["run_id", "structure", "start_date", "end_date"]:
            if key in backtest_summary:
                lines.append(f"- **{key}:** {backtest_summary[key]}")
        lines.append("")

        # Performance metrics
        perf_keys = ["total_return", "net_cagr", "net_sharpe", "sharpe_ratio",
                     "max_drawdown", "win_rate", "total_trades"]
        perf_found = [k for k in perf_keys if k in backtest_summary]
        if perf_found:
            lines.append("### Performance")
            lines.append("")
            for key in perf_found:
                val = backtest_summary[key]
                if isinstance(val, float):
                    if "pct" in key or "rate" in key or "return" in key or "drawdown" in key:
                        lines.append(f"- **{key}:** {val:.2%}")
                    else:
                        lines.append(f"- **{key}:** {val:.4f}")
                else:
                    lines.append(f"- **{key}:** {val}")
        lines.append("")
        lines.append(f"*Summary path:* `{backtest_summary.get('summary_path', '—')}`")
        lines.append("")

    # Next steps from state
    next_steps = state.get("next_steps", [])
    if next_steps:
        lines.append("## Next Steps")
        lines.append("")
        for step in next_steps:
            lines.append(f"- {step}")
        lines.append("")

    lines.append("---")
    lines.append(f"*Auto-generated by `src/status_report.py` at {now}*")
    lines.append("")

    return "\n".join(lines)


def main():
    """Generate status report files."""
    # Ensure reports directory exists
    REPORT_DIR.mkdir(exist_ok=True)

    # Load data
    config = load_config()
    state = load_state()
    backtest_summary = find_latest_backtest()
    modules = get_backtest_modules()
    git_info = get_git_commit()

    # Generate markdown
    md_content = render_markdown(config, state, backtest_summary, modules, git_info)

    # Write STATUS.md
    status_md_path = REPORT_DIR / "STATUS.md"
    with open(status_md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    # Generate JSON snapshot (git_info already loaded above)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snapshot = {
        "generated_utc": now,
        "project": config.get("project", "dissertation-options-beta-neutral"),
        "git": git_info,
        "last_run_utc": state.get("last_run_utc"),
        "data_summary": state.get("data_summary", {}),
        "years_available": state.get("years_available", []),
        "months_by_year": state.get("months_by_year", {}),
        "backtest_modules": modules,
        "latest_backtest": backtest_summary or {},
        "next_steps": state.get("next_steps", [])
    }

    # Write status.json
    status_json_path = REPORT_DIR / "status.json"
    with open(status_json_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, indent=2, fp=f)

    print(f"Generated: {status_md_path}")
    print(f"Generated: {status_json_path}")

    return status_md_path, status_json_path


if __name__ == "__main__":
    main()
