########################################
# Usage: awk -f winners.awk out.txt
########################################

BEGIN {
  problemLine = "  ---Problem---"
  winnerLine = "  ---Winner---"
}

/Problem\[/ {
  printf "%-40s - %s\n", problemLine, winnerLine
  problemLine = $0
}

/\*,/ {
  winnerLine = $0
}

