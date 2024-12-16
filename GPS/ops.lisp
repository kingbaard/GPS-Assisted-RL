;;; -*- Mode: Lisp; Syntax: Common-Lisp; -*-
;;; Module: ops.lisp
;;; different worlds and operators for the GPS planner.
;;; bugs to vladimir kulyukin in canvas
;;; =========================================

(in-package :user)

(defstruct op "An GPS operator"
  (action nil) 
  (preconds nil) 
  (add-list nil) 
  (del-list nil))

(defun executing-p (x)
  "Is x of the form: (execute ...) ?"
  (starts-with x 'execute))

(defun convert-op (op)
  "Make op conform to the (EXECUTING op) convention."
  (unless (some #'executing-p (op-add-list op))
    (push (list 'execute (op-action op)) (op-add-list op)))
  op)

(defun op (action &key preconds add-list del-list)
  "Make a new operator that obeys the (EXECUTING op) convention."
  (convert-op
    (make-op :action action :preconds preconds
             :add-list add-list :del-list del-list)))

;;; ================= village-safe ====================

;;;(defparameter *dragon-world* '(dragon-alive forest-on-fire))

(defparameter *dragon-ops*
  (list

    (make-op 
      :action 'goto-boat
      :add-list '(boat-boarded))
      
    (make-op 
      :action 'goto-mermaid-beach
      :preconds '(mermaid-on-beach)
      :add-list '(fire-extinguished)
      :del-list '(forest-on-fire))

    (make-op
      :action 'goto-mermaid-ocean
      :preconds '(boat-boarded mermaid-in-ocean)
      :add-list '(fire-extinguished)
      :del-list '(forest-on-fire))

    (make-op 
      :action 'goto-sword
      :preconds '(fire-extinguished)
      :add-list '(has-sword)
      :del-list '(sword-in-forest))

    (make-op 
      :action 'goto-dragon
      :preconds '(has-sword)
      :add-list '(no-dragon)
      :del-list '(dragon-alive))
    )
  ) (mapc #'convert-op *dragon-ops*)

(provide :ops)
