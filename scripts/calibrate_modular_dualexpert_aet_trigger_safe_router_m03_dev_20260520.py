import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.calibrate_modular_dualexpert_aet_stable_router_m02_dev_20260520 as cal


cal.BRANCH = "aet_trigger_safe_router_m03_routecls_noauxwarm_lr2e6_save50"
cal.SCORE_ROOT = cal.REPO / "outputs/stage2_modular_dualexpert/aet_trigger_safe_router_m03_20260520/route_likelihood" / cal.BRANCH
cal.OUT_JSON = cal.REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_trigger_safe_router_m03_dev.json"
cal.OUT_MD = cal.REPO / "reports/2026-05-20_stage2_modular_dualexpert_aet_trigger_safe_router_m03_dev.md"
BASE_RENDER_REPORT = cal.render_report


def render_report(payload):
    return BASE_RENDER_REPORT(payload).replace(
        "# A/E/T Stable Router M02 Dev Sweep",
        "# A/E/T Trigger-Safe Stable Router M03 Dev Sweep",
        1,
    )


if __name__ == "__main__":
    cal.render_report = render_report
    cal.main()
