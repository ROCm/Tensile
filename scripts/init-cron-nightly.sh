H1='\033[0;31m'
H2='\033[0;32m'
NC='\033[0m' # No color

if crontab -l | grep -q 'run-profile'; then
    echo -e "$H1> Error: cron job with same command already exists.$NC"
    echo -e "$H1> Clean your crontab manually with 'crontab -e' and rerun.$NC"
    echo -e "$H1> Conflict line:\n>   `crontab -l | grep run-profile`$NC"
    exit 1
else
    (crontab -l 2>/dev/null; echo "00 22 * * 0-4 $HOME/automation/Tensile/scripts/run-profile.sh --build-id=14354") | crontab -
    echo -e "${H2}Added cron job:\n  `crontab -l | tail -n 1`$NC"
fi

