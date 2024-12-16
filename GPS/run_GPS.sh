#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <world-state> <goal-state>"
  exit 1
fi

# Capture the goal state from the command-line argument
WORLD_STATE=$1
GOAL_STATE=$2

# Start Common Lisp and load the gps.lisp file
clisp <<EOF
(load "GPS/gps.lisp")

(defparameter *dragon-world* '( $WORLD_STATE ))

(sleep 1)
# Establish operators
(use *dragon-ops*)

(sleep 1)
# Run GPS in the context of the problem
(gps *dragon-world* '( $GOAL_STATE ))

(sleep 1)
(exit)
EOF