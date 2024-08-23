H1='\033[0;31m'
H2='\033[0;32m'
NC='\033[0m' # No color

# Variables
build_id="14543"
tensile_path="`pwd`"

# Constants
script_to_run="profile-tcl.sh"

usage() {
    echo "Set up cron table for nightly Tensile profiling reports"
    echo ""
    echo "Usage: $0 [--build-id=<id>] [--tensile-path=<path>]"
    echo ""
    echo "Parameters:"
    echo "  --build-id: The target docker build ID [default: $build_id]"
    echo "  --tensile-path: Path to root directory of Tensile [default: $tensile_path]"
    echo ""
    echo "Example:"
    echo "  $0 --build-id=14354 --tensile-path='path/to/tensile'"
}


# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --build-id=*) build_id="${arg#*=}" ;;
        --tensile-path=*) tensile_path="${arg#*=}" ;;
        --help) usage; exit 0 ;;
        *) echo "Invalid option: $arg"; usage; exit 1 ;;
    esac
done

# Check if all parameters are provided
if [ -z "$build_id" ] || [ -z "$tensile_path" ]; then
    usage
    exit 1
fi

if [ ! -e "$tensile_path" ]; then
    echo -e "$H1+ Error: path to Tensile does not exist.$NC"
    echo -e "$H1+ Cannot find: $tensile_path$NC"
    exit 1
fi

if crontab -l | grep -q "$script_to_run"; then
    echo -e "$H1+ Error: cron job with same command already exists.$NC"
    echo -e "$H1+ Clean your crontab manually with 'crontab -e' and rerun.$NC"
    echo -e "$H1+ Conflict line:\n+   `crontab -l | grep $script_to_run`$NC"
    exit 1
else
    cron_log="$tensile_path/tcl-profile-$(date +'%Y-%m-%dT%H-%M-%S').log.cron"
    (crontab -l 2>/dev/null; echo "00 22 * * 0-4 $tensile_path/scripts/$script_to_run --build-id=$build_id | tee $cron_log") | crontab -
    echo -e "${H2}Added cron job:\n  `crontab -l | tail -n 1`$NC"
fi

