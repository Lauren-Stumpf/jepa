#!/usr/bin/env bash
# Add a crontab entry that runs launch_all.sh two minutes after every reboot, so the
# baselines resume on their own. Run once:  bash scripts/install_reboot_hook.sh
# Remove later with:  crontab -l | grep -v launch_all.sh | crontab -
set -euo pipefail
LINE='@reboot sleep 120 && bash /storage3/lauren/code/jepa/scripts/launch_all.sh >> /storage3/lauren/code/jepa/results/launch_all.log 2>&1'
# with no existing crontab, "crontab -l" fails and grep finds nothing; neither is an error here
existing=$(crontab -l 2>/dev/null | grep -v launch_all.sh || true)
printf '%s\n%s\n' "$existing" "$LINE" | sed '/^$/d' | crontab -
echo "installed:"; crontab -l | grep launch_all.sh
