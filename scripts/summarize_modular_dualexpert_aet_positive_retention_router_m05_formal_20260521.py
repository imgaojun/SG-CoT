import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.summarize_modular_dualexpert_aet_rankstable_router_m04b_formal_20260520 as summ


summ.summ.BRANCH = "aet_positive_retention_router_m05_routecls_noauxwarm_lr2e6_save50"
summ.summ.SCORE_ROOT = summ.summ.REPO / "outputs/stage2_modular_dualexpert/aet_positive_retention_router_m05_20260521/formal_route_likelihood"
summ.summ.DEV_JSON = summ.summ.REPO / "reports/artifacts/2026-05-21_stage2_modular_dualexpert_aet_positive_retention_router_m05_dev.json"
summ.summ.OUT_JSON = summ.summ.REPO / "reports/artifacts/2026-05-21_stage2_modular_dualexpert_aet_positive_retention_router_m05_formal.json"
summ.summ.OUT_MD = summ.summ.REPO / "reports/2026-05-21_stage2_modular_dualexpert_aet_positive_retention_router_m05_formal.md"
BASE_RENDER_REPORT = summ.render_report


def render_report(payload):
    return BASE_RENDER_REPORT(payload).replace(
        "# A/E/T Rank-Stable Router M04B Formal Replay",
        "# A/E/T Positive-Retention Router M05 Formal Replay",
        1,
    )


if __name__ == "__main__":
    summ.render_report = render_report
    summ.summ.render_report = render_report
    summ.summ.main()
