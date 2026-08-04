import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.summarize_modular_dualexpert_aet_stable_router_m02_formal_20260520 as summ


summ.BRANCH = "aet_trigger_safe_router_m03_routecls_noauxwarm_lr2e6_save50"
summ.SCORE_ROOT = summ.REPO / "outputs/stage2_modular_dualexpert/aet_trigger_safe_router_m03_20260520/formal_route_likelihood"
summ.DEV_JSON = summ.REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_trigger_safe_router_m03_dev.json"
summ.OUT_JSON = summ.REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_trigger_safe_router_m03_formal.json"
summ.OUT_MD = summ.REPO / "reports/2026-05-20_stage2_modular_dualexpert_aet_trigger_safe_router_m03_formal.md"
BASE_RENDER_REPORT = summ.render_report


def render_report(payload):
    return BASE_RENDER_REPORT(payload).replace(
        "# A/E/T Stable Router M02 Formal Replay",
        "# A/E/T Trigger-Safe Stable Router M03 Formal Replay",
        1,
    )


if __name__ == "__main__":
    summ.render_report = render_report
    summ.main()
