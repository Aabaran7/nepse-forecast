"""The restore path, which only ever runs when something has gone wrong.

Every CI run starts on a wiped machine, so `archive_restore.py` is not a
disaster-recovery nicety -- it is in the normal daily path, and a bug in it is
indistinguishable from data loss. Two failures are worth testing directly:

  A RESTORE THAT SHRINKS. If the archive on disk is ahead of the mirror (a pull
  ran, the backup did not), overwriting it destroys sessions that NEPSE will not
  serve again (§3.4). It must refuse.

  A RESTORE THAT CHANGES TYPES. This one is nastier because it does not raise.
  CSV has no types; if securityId comes back as int 131 where the archive holds
  the string "131", archive.merge() stops matching keys and appends a second
  copy of the entire history. The archive is then corrupt and still looks fine.

Run: .venv/bin/pytest tests/test_restore.py -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nepselab.ingest import archive  # noqa: E402

RESTORE = REPO / "scripts" / "archive_restore.py"
BACKUP = REPO / "scripts" / "archive_backup.py"


def run(script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args],
                          cwd=cwd, capture_output=True, text=True)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A throwaway project tree with a small but type-diverse archive."""
    (tmp_path / "data" / "archive").mkdir(parents=True)

    # securityId as STRING is the case that broke the first implementation --
    # it is genuinely stored that way in securities.parquet.
    # snapshot_date at MILLISECOND resolution, also for real: pd.to_datetime
    # always produces ns, so a restore that ignores the recorded unit drifts here
    # even though every value compares equal.
    pd.DataFrame({
        "snapshot_date": pd.to_datetime(["2026-08-01", "2026-08-01"]).astype("datetime64[ms]"),
        "securityId": ["131", "132"],
        "symbol": ["NABIL", "SCB"],
    }).to_parquet(tmp_path / "data/archive/securities.parquet", index=False)

    # ...and as INT in today_price, in the same archive, on purpose.
    pd.DataFrame({
        "businessDate": pd.to_datetime(["2026-08-01"] * 2),
        "securityId": [131, 132],
        "closePrice": [500.0, 610.5],
        "totalTrades": [12, 340],
    }).to_parquet(tmp_path / "data/archive/today_price.parquet", index=False)
    return tmp_path


def mirror_of(project: Path) -> Path:
    mirror = project / "data-mirror"
    mirror.mkdir(exist_ok=True)
    subprocess.run(["git", "init", "-q", str(mirror)], check=True)
    r = run(BACKUP, "--dir", str(mirror), "--no-push", cwd=project)
    assert r.returncode == 0, r.stderr
    return mirror


class TestRoundTrip:
    def test_types_survive_and_the_archive_still_merges(self, project: Path):
        mirror = mirror_of(project)
        before = {p.stem: pd.read_parquet(p)
                  for p in (project / "data/archive").glob("*.parquet")}

        # Wipe the archive the way a fresh CI runner would, then restore.
        for p in (project / "data/archive").glob("*.parquet"):
            p.unlink()
        r = run(RESTORE, "--from", str(mirror), cwd=project)
        assert r.returncode == 0, r.stderr

        for name, orig in before.items():
            got = pd.read_parquet(project / f"data/archive/{name}.parquet")
            assert got[orig.columns].dtypes.astype(str).tolist() == \
                   orig.dtypes.astype(str).tolist(), f"{name} dtypes drifted"

            # The test that actually matters: re-merging the original must be a
            # no-op. If keys stopped matching, this silently doubles the table.
            res = archive.merge(name, orig, root=project / "data/archive")
            assert res.added == 0, f"{name} re-merge added {res.added} rows"
            assert len(res.conflicts) == 0, f"{name} re-merge saw conflicts"

    def test_string_ids_do_not_come_back_as_integers(self, project: Path):
        mirror = mirror_of(project)
        (project / "data/archive/securities.parquet").unlink()
        run(RESTORE, "--from", str(mirror), cwd=project)
        got = pd.read_parquet(project / "data/archive/securities.parquet")
        assert got["securityId"].tolist() == ["131", "132"]


class TestRefusesToDestroy:
    def test_will_not_shrink_an_archive_that_is_ahead(self, project: Path):
        mirror = mirror_of(project)

        # A pull adds a session; the backup does not run. The mirror is now stale.
        grown = pd.concat([
            pd.read_parquet(project / "data/archive/today_price.parquet"),
            pd.DataFrame({"businessDate": pd.to_datetime(["2026-08-04"]),
                          "securityId": [131], "closePrice": [505.0],
                          "totalTrades": [9]}),
        ], ignore_index=True)
        grown.to_parquet(project / "data/archive/today_price.parquet", index=False)

        r = run(RESTORE, "--from", str(mirror), cwd=project)
        assert r.returncode != 0
        assert "RestoreWouldShrink" in r.stderr or "Refusing" in r.stderr
        # and the extra session is still there
        assert len(pd.read_parquet(project / "data/archive/today_price.parquet")) == 3

    def test_force_overrides_deliberately(self, project: Path):
        mirror = mirror_of(project)
        pd.concat([
            pd.read_parquet(project / "data/archive/today_price.parquet"),
            pd.DataFrame({"businessDate": pd.to_datetime(["2026-08-04"]),
                          "securityId": [131], "closePrice": [505.0],
                          "totalTrades": [9]}),
        ], ignore_index=True).to_parquet(
            project / "data/archive/today_price.parquet", index=False)

        r = run(RESTORE, "--from", str(mirror), "--force", cwd=project)
        assert r.returncode == 0, r.stderr
        assert len(pd.read_parquet(project / "data/archive/today_price.parquet")) == 2

    def test_a_logged_prediction_is_never_overwritten(self, project: Path):
        """§7: the forward log is append-only, and a restore is not an exception."""
        (project / "predictions").mkdir()
        (project / "predictions/2026-08-03.csv").write_text("as_of,horizon\nreal\n")
        mirror = mirror_of(project)
        (project / "predictions/2026-08-03.csv").write_text("as_of,horizon\nLOCAL\n")

        run(RESTORE, "--from", str(mirror), cwd=project)
        assert "LOCAL" in (project / "predictions/2026-08-03.csv").read_text()


class TestFirstRun:
    def test_a_missing_mirror_is_not_an_error(self, project: Path):
        # The very first CI run has nothing to restore. Failing here would block
        # the run that creates the archive in the first place.
        r = run(RESTORE, "--from", str(project / "nope"), cwd=project)
        assert r.returncode == 0

    def test_an_old_mirror_without_dtypes_still_restores_dates(self, project: Path):
        mirror = mirror_of(project)
        (mirror / "archive/_dtypes.json").unlink()
        (project / "data/archive/today_price.parquet").unlink()

        r = run(RESTORE, "--from", str(mirror), cwd=project)
        assert r.returncode == 0, r.stderr
        got = pd.read_parquet(project / "data/archive/today_price.parquet")
        assert str(got["businessDate"].dtype).startswith("datetime")


def test_dtypes_sidecar_records_every_column(project: Path):
    mirror = mirror_of(project)
    schema = json.loads((mirror / "archive/_dtypes.json").read_text())
    assert schema["securities"]["securityId"] == "object"
    assert schema["today_price"]["securityId"] == "int64"
