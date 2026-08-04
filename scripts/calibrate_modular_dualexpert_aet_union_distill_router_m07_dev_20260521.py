import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.calibrate_modular_dualexpert_aet_rankstable_router_m04a_dev_20260520 as cal  # noqa: E402


cal.cal.BRANCH = "aet_union_distill_router_m07_routecls_noauxwarm_lr2e6_save50"
cal.cal.SCORE_ROOT = cal.cal.REPO / "outputs/stage2_modular_dualexpert/aet_union_distill_router_m07_20260521/route_likelihood" / cal.cal.BRANCH
cal.cal.OUT_JSON = cal.cal.REPO / "reports/artifacts/2026-05-21_stage2_modular_dualexpert_aet_union_distill_router_m07_dev.json"
cal.cal.OUT_MD = cal.cal.REPO / "reports/2026-05-21_stage2_modular_dualexpert_aet_union_distill_router_m07_dev.md"
BASE_RENDER_REPORT = cal.render_report


def render_report(payload):
    return BASE_RENDER_REPORT(payload).replace(
        "# A/E/T Rank-Stable Router M04A Dev Sweep",
        "# A/E/T Union-Distill Router M07 Dev Sweep",
        1,
    )


if __name__ == "__main__":
    cal.render_report = render_report
    cal.main()
