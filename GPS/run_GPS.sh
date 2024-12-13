if [ -z "$1" ]; then
  echo "Usage: $0 <goal-state>"
  exit 1
fi

# Capture the goal state from the command-line argument
GOAL_STATE=$1

# Start Common Lisp and load the gps.lisp file
pwd
clisp <<EOF

(load "GPS/gps.lisp")

(sleep 1)
# Establish operators
(use *dragon-ops*)

(sleep 1)
# Run GPS in the context of the problem
(gps *dragon-world* '($GOAL_STATE))

(sleep 1)
(exit)
EOF