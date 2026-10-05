"""
Recompute the repository figures quoted in the JSS paper (Sections 3 and 5).

  python scripts/repo_metrics.py            # from the repo root, inside the server venv
  python scripts/repo_metrics.py --json out.json

Reports: module and line counts, route handlers, test count and coverage, which
modules import concrete provider adapters, capability probes in the orchestrator,
the declared experiment vocabulary, and the numstat of the two capability commits
(if they are in this clone's history). Also re-executes change scenario C4 (adding
an expressive category, posture, in config.yaml only) on a temporary copy of the repo.

Author: Alexander Barquero Elizondo, Ph.D. -- UCR, ECCI/CITIC
License: MIT
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, OVARP_TESTING="1", OPENAI_API_KEY="sk-dummy", GEMINI_API_KEY="dummy")


def sh(cmd, cwd=ROOT):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=ENV, shell=isinstance(cmd, str)).stdout


def lines(paths):
    return sum(len(Path(p).read_text(encoding="utf-8", errors="ignore").splitlines()) for p in paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    m = {}
    src = [p for p in ROOT.glob("src/**/*.py")]
    m["python_modules"] = len([p for p in src if p.name != "__init__.py"])
    m["python_lines"] = lines(src)
    static = [p for p in ROOT.glob("src/static/**/*") if p.suffix in (".js", ".html", ".css")]
    m["console_lines"] = lines(static)
    routes = re.findall(r"@(?:router|results_router|app)\.(get|post|put|patch|delete|websocket)\(",
                        "\n".join(p.read_text(encoding="utf-8") for p in src))
    m["route_handlers"] = len(routes)
    m["routers"] = len([p for p in ROOT.glob("src/api/routers/*.py") if p.name != "__init__.py"])
    tests = list(ROOT.glob("tests/**/test_*.py"))
    m["test_files"], m["test_lines"] = len(tests), lines(tests)

    cov = sh([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--cov=src", "--cov-report=term"])
    m["tests_passed"] = int(re.search(r"(\d+) passed", cov).group(1)) if "passed" in cov else None
    tot = re.search(r"TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%", cov)
    if tot:
        m["statements"], m["coverage_pct"] = int(tot.group(1)), int(tot.group(3))

    concrete = re.compile(r"(openai_provider|gemini_provider|custom_provider)")
    m["modules_importing_concrete_adapters"] = sorted(
        str(p.relative_to(ROOT)) for p in src
        if "src/providers/" not in str(p) and any(concrete.search(l) for l in p.read_text(encoding="utf-8").splitlines()
                                                   if l.lstrip().startswith(("from ", "import ")) or "import " in l and "from src" in l))
    orch = (ROOT / "src/core/orchestrator.py").read_text(encoding="utf-8")
    m["orchestrator_capability_probes"] = re.findall(r"hasattr\(self\.(?:llm|tts|stt), \"(\w+)\"\)", orch)

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cats = cfg.get("custom_commands") or {}
    m["vocabulary"] = {
        "devices": len(cfg.get("devices") or []), "agents": len(cfg.get("agents") or []),
        "categories": len(cats), "category_values": sum(len(c["values"]) for c in cats.values()),
        "tts_voices": sum(len(v) for v in (cfg.get("tts") or {}).values()),
        "marker_presets": len(cfg.get("event_markers") or []),
        "profiles": len(list(ROOT.glob("profiles/*.yaml"))),
        "scenarios": len(list(ROOT.glob("scenarios/*.yaml"))),
        "scenario_steps": sum(len(yaml.safe_load(p.read_text(encoding="utf-8"))["steps"]) for p in ROOT.glob("scenarios/*.yaml")),
        "instruments": {p.stem: len(yaml.safe_load(p.read_text(encoding="utf-8")).get("items", [])) for p in ROOT.glob("surveys/*.yaml")},
    }

    m["capability_commits"] = {}
    for tag in ("OPA-303", "OPA-335"):
        h = sh(["git", "log", "--format=%h", f"--grep={tag}"]).split()
        if h:
            m["capability_commits"][tag] = sh(["git", "show", "--numstat", "--format=", h[0]]).strip().splitlines()

    # Scenario C4, executed on a temporary copy
    with tempfile.TemporaryDirectory() as d:
        cp = Path(d) / "repo"
        shutil.copytree(ROOT, cp, ignore=shutil.ignore_patterns(".git", "venv", "data", "bench_results", "__pycache__"))
        c = (cp / "config.yaml").read_text(encoding="utf-8")
        added = '  posture:\n    description: "Body posture of the agent"\n    values:\n      - "upright"\n      - "leaning_forward"\n      - "leaning_back"\n\n'
        (cp / "config.yaml").write_text(c.replace("custom_commands:\n", "custom_commands:\n" + added, 1), encoding="utf-8")
        probe = r'''
from src.core.config import config_manager; config_manager.load_config()
from src.core.schemas import BaseCommand
ok = BaseCommand(sender="w", target_device="web_01", target_agent="agent_alpha", command_type="action", command="execute_state", subcommand={"posture": "leaning_forward"})
try:
    BaseCommand(sender="w", target_device="web_01", target_agent="agent_alpha", command_type="action", command="execute_state", subcommand={"posture": "lying_down"}); rej = False
except Exception: rej = True
from src.providers.gemini_provider import GeminiLLMProvider
from src.providers.openai_provider import OpenAILLMProvider
print(rej, "posture" in str(GeminiLLMProvider()._build_tools_schema()), "posture" in str(OpenAILLMProvider()._build_tools_schema()))
'''
        out = sh([sys.executable, "-c", probe], cwd=cp).split()
        t = sh([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=cp)
        m["scenario_C4"] = {"lines_added": added.count("\n"), "files_changed": 1,
                            "invalid_value_rejected": out[0:1] == ["True"],
                            "in_gemini_tool_schema": out[1:2] == ["True"], "in_openai_tool_schema": out[2:3] == ["True"],
                            "tests": re.search(r"\d+ passed[^\n]*", t).group(0) if "passed" in t else t[-300:]}

    print(json.dumps(m, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(m, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
