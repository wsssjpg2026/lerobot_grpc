"""Server-log gate-event capture and parsing (implementation-gap #2 tooling).

The safety stack's state (rejected / held / stale / collided / jumped) has
no RPC-visible channel — it surfaces only as throttled log records.  This
module turns those logs back into structured data:

- :class:`GateLogCapture` — a logging.Handler attached to the
  ``lerobot_robot_grpc.follower`` logger subtree during an in-process (sim)
  run; the harness slices the stream per sequence and the report writer
  merges it into every baseline.
- :func:`parse_gate_line` — pattern-matcher for the four gate lines the
  law/adapters emit (works identically on live records and on tee'd files).
- CLI merge for real runs (B-3): the runbook tees the server process::

    conda run -n lerobot-grpc-serve python examples/serve_so101_follower.py \\
        --robot.port=<SERIAL_PORT> --action_mode=pose_delta \\
        --address=127.0.0.1:5556 2>&1 | tee follower.log

  then, after the suite wrote its report::

    conda run -n lerobot-grpc-serve python -m tests.response.gate_events \\
        follower.log outputs/response_baseline/<run>_real/report.json

  which appends a ``gate_events_from_log`` section (counts + samples) to
  the report and prints a summary.  Sequence correlation on real runs is
  wall-clock approximate (the sim path slices exactly by record time).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

FOLLOWER_LOGGER = "lerobot_robot_grpc.follower"

# ---------------------------------------------------------------------------
# Line parsing (identical grammar for live records and tee'd files)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateEvent:
    kind: str            # reject | jump | hold | relatch
    detail: str          # reason / flags as logged
    pos_err_mm: float | None = None
    t_wall: float | None = None  # record.created (epoch) when known
    t_text: str = ""     # raw asctime when parsed from a file

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "pos_err_mm": self.pos_err_mm,
            "t_wall": self.t_wall,
            "t_text": self.t_text,
        }


_RE_REJECT = re.compile(r"SOLVE rejected \((?P<reason>[\w-]+)\): pos_err=(?P<err>[\d.]+)mm")
_RE_JUMP = re.compile(r"IK jump: solution (?P<deg>[\d.]+)° from the previous")
_RE_HOLD = re.compile(
    r"pose_delta hold: stale=(?P<stale>\w+) rejected=(?P<rejected>\w+) "
    r"collided=(?P<collided>\w+) pos_err=(?P<err>-?[\d.]+)mm"
)
_RE_RELATCH = re.compile(r"SetReference: T_arm_ref re-locked at current FK")


def parse_gate_line(text: str, t_wall: float | None = None,
                    t_text: str = "") -> GateEvent | None:
    """One log line -> GateEvent, or None for non-gate lines."""
    if (m := _RE_REJECT.search(text)):
        return GateEvent("reject", m["reason"], float(m["err"]), t_wall, t_text)
    if (m := _RE_JUMP.search(text)):
        return GateEvent("jump", f"{m['deg']}deg", None, t_wall, t_text)
    if (m := _RE_HOLD.search(text)):
        flags = f"stale={m['stale']} collided={m['collided']}"
        if m["rejected"] == "True":
            flags = "rejected=True " + flags
        return GateEvent("hold", flags, float(m["err"]), t_wall, t_text)
    if _RE_RELATCH.search(text):
        return GateEvent("relatch", "T_arm_ref re-locked", None, t_wall, t_text)
    return None


# ---------------------------------------------------------------------------
# In-process capture (sim)
# ---------------------------------------------------------------------------


class GateLogCapture(logging.Handler):
    """Collects gate events from the follower logger subtree.

    Attaching to ``lerobot_robot_grpc.follower`` catches every child logger
    (pose_delta_law, both servicers) — records propagate up to it.  The
    subtree's level is lifted to INFO while attached (the ``pose_delta
    hold`` line is INFO; default root WARNING would drop it) and restored
    on detach.
    """

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.events: list[GateEvent] = []
        self._logger = logging.getLogger(FOLLOWER_LOGGER)
        self._attached = False

    def attach(self) -> None:
        if self._attached:
            return
        self._prev_level = self._logger.level
        if self._prev_level == logging.NOTSET or self._prev_level > logging.INFO:
            self._logger.setLevel(logging.INFO)
        self._logger.addHandler(self)
        self._attached = True

    def detach(self) -> None:
        if not self._attached:
            return
        self._logger.removeHandler(self)
        self._logger.setLevel(self._prev_level)
        self._attached = False

    def emit(self, record: logging.LogRecord) -> None:
        event = parse_gate_line(record.getMessage(), t_wall=record.created)
        if event is not None:
            self.events.append(event)


def slice_events(events: list[GateEvent], t_start: float, t_end: float) -> list[GateEvent]:
    """Events with wall times inside [t_start, t_end] (epoch seconds)."""
    return [e for e in events if e.t_wall is not None and t_start <= e.t_wall <= t_end]


def summarize(events: list[GateEvent]) -> dict:
    """Counts per kind + up to three sample lines each (report payload)."""
    out: dict = {"total": len(events)}
    for kind in ("reject", "jump", "hold", "relatch"):
        of_kind = [e for e in events if e.kind == kind]
        if not of_kind:
            continue
        out[f"{kind}_count"] = len(of_kind)
        out[f"{kind}_samples"] = [e.to_json() for e in of_kind[:3]]
    return out


# ---------------------------------------------------------------------------
# Tee'd-file parsing + report merge (real runbook)
# ---------------------------------------------------------------------------

# The real launcher's basicConfig prefix: "%(asctime)s %(levelname)s %(name)s: %(message)s"
_RE_LOGFILE_LINE = re.compile(
    r"^(?P<asctime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?)\s+"
    r"(?P<level>\w+)\s+(?P<name>[\w.]+):\s+(?P<msg>.*)$"
)


def parse_log_file(path: Path) -> list[GateEvent]:
    events: list[GateEvent] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = _RE_LOGFILE_LINE.match(raw)
        if not m:
            continue
        event = parse_gate_line(m["msg"], t_text=m["asctime"])
        if event is not None:
            events.append(event)
    return events


def merge_into_report(log_path: Path, report_path: Path) -> dict:
    """Parse a tee'd server log and append its gate events to a report JSON."""
    events = parse_log_file(log_path)
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    payload["gate_events_from_log"] = {
        "log_file": str(log_path),
        "note": "parsed from a tee'd server log (real runbook); sequence "
        "correlation is wall-clock approximate",
        **summarize(events),
    }
    Path(report_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload["gate_events_from_log"]


def _cli(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    summary = merge_into_report(Path(argv[1]), Path(argv[2]))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv))
