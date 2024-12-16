;;; -*- Mode: Lisp; Syntax: Common-Lisp; -*-
;;; Module: gps.lisp: a Lisp version of GPS.
;;; bugs to vladimir kulyukin in canvas
;;; =========================================

(in-package :user)

;;; get the auxfuns and ops
(require :auxfuns "auxfuns")
(require :ops "ops")

;;; ==============================

(defvar *ops* nil "A list of available operators.")

(defun gps (state goals &optional (*ops* *ops*))
  "General Problem Solver: from state, achieve goals using *ops*."
  (remove-if #'atom (achieve-all (cons '(start) state) goals nil)))

;;; ==============================

(defun achieve-all (state goals goal-stack)
  "Achieve each goal, and make sure they still hold at the end."
  (let ((current-state state))
    (if (and (every #'(lambda (g)
                        (setf current-state
                              (achieve current-state g goal-stack)))
                    goals)
             (subsetp goals current-state :test #'equal))
        current-state)))

(defun achieve (state goal goal-stack)
  "A goal is achieved if it already holds,
  or if there is an appropriate op for it that is applicable."
  (dbg-indent :gps (length goal-stack) "Goal: ~a" goal)
  (cond ((member-equal goal state) state)
        ((member-equal goal goal-stack) nil)
        (t (some #'(lambda (op) (apply-op state goal op goal-stack))
                 (find-all goal *ops* :test #'appropriate-p)))))

;;; ==============================

(defun member-equal (item list)
  (member item list :test #'equal))

;;; ==============================

(defun apply-op (state goal op goal-stack)
  "Return a new, transformed state if op is applicable."
  (dbg-indent :gps (length goal-stack) "Consider: ~a" (op-action op))
  (let ((state2 (achieve-all state (op-preconds op) 
                             (cons goal goal-stack))))
    (unless (null state2)
      ;; Ensure no locked facts are removed
      (let ((updated-state
             (append (remove-if #'(lambda (x) 
                                    (and (member-equal x (op-del-list op))
                                         (not (member-equal x '(fire-extinguished on-boat)))))
                                state2)
                     (op-add-list op))))
        (dbg-indent :gps (length goal-stack) "Action: ~a" (op-action op))
        updated-state))))
(defun appropriate-p (goal op)
  "An op is appropriate to a goal if it is in its add list."
  (member-equal goal (op-add-list op)))

;;; ==============================

(defun use (oplist)
  "Use oplist as the default list of operators."
  ;; Return something useful, but not too verbose: 
  ;; the number of operators.
  (length (setf *ops* oplist)))

;;; ==============================

(defun trace-gps ()
  (do-debug :gps))

(defun untrace-gps ()
  (do-undebug))

