H1='\033[0;31m'
H2='\033[0;32m'
NC='\033[0m' # No color

usage() {
    echo "Set up cron table for nightly Tensile profiling reports"
    echo ""
    echo "Usage: $0 --build-id=<build-id>"
    echo ""
    echo "Parameters:"
    echo "  --build-id: The target docker build ID"
    echo ""
    echo "Example:"
    echo "  $0 --build-id=14354"
}

# Variables
build_id=""

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --build-id=*) build_id="${arg#*=}" ;;
        --help) usage; exit 0 ;;
        *) echo "Invalid option: $arg"; usage; exit 1 ;;
    esac
done

# Check if all parameters are provided
if [ -z "$build_id" ]; then
    usage
    exit 1
fi

if crontab -l | grep -q 'run-profile'; then
    echo -e "$H1> Error: cron job with same command already exists.$NC"
    echo -e "$H1> Clean your crontab manually with 'crontab -e' and rerun.$NC"
    echo -e "$H1> Conflict line:\n>   `crontab -l | grep run-profile`$NC"
    exit 1
else
    (crontab -l 2>/dev/null; echo "00 22 * * 0-4 $HOME/automation/Tensile/scripts/run-profile.sh --build-id=$build_id") | crontab -
    echo -e "${H2}Added cron job:\n  `crontab -l | tail -n 1`$NC"
fi

