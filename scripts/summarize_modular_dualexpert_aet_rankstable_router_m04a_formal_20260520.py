import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.summarize_modular_dualexpert_aet_stable_router_m02_formal_20260520 as summ


summ.BRANCH = "aet_rankstable_router_m04a_routecls_noauxwarm_lr2e6_save50"
summ.SCORE_ROOT = summ.REPO / "outputs/stage2_modular_dualexpert/aet_rankstable_router_m04a_20260520/formal_route_likelihood"
summ.DEV_JSON = summ.REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_rankstable_router_m04a_dev.json"
summ.OUT_JSON = summ.REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_rankstable_router_m04a_formal.json"
summ.OUT_MD = summ.REPO / "reports/2026-05-20_stage2_modular_dualexpert_aet_rankstable_router_m04a_formal.md"
BASE_RENDER_REPORT = summ.render_report


def render_report(payload):
    return BASE_RENDER_REPORT(payload).replace(
        "# A/E/T Stable Router M02 Formal Replay",
        "# A/E/T Rank-Stable Router M04A Formal Replay",
        1,
    )


if __name__ == "__main__":
    summ.render_report = render_report
    summ.main()
