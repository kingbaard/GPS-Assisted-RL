;;; -*- Mode: Lisp; Syntax: Common-Lisp; -*-
#|
===========================================
file: auxfuns.lisp
description: some aux funs
===========================================
|#

(in-package :user)

(defconstant fail nil)
(defconstant no-bindings '((t . t)))

(defvar *dbg-ids* nil "Identifiers used by dbg")

(defun pat-match (pattern input &optional (bindings no-bindings))
   "Match pattern against input in the context of the bindings"
   (cond ((eq bindings fail) fail)
	 ((variable-p pattern)
	  (match-variable pattern input bindings))
	 ((eql pattern input) bindings)
	 ((segment-pattern-p pattern)                ; ***
	  (segment-match pattern input bindings))    ; ***
	 ((and (consp pattern) (consp input)) 
	  (pat-match (rest pattern) (rest input)
		     (pat-match (first pattern) (first input) 
				bindings)))
	 (t fail)))

(defun match-variable (var input bindings)
   "Does VAR match input?  Uses (or updates) and returns bindings."
   (let ((binding (get-binding var bindings)))
     (cond ((not binding) (extend-bindings var input bindings))
	   ((equal input (binding-val binding)) bindings)
	   (t fail))))

(defun make-binding (var val) (cons var val))

(defun binding-var (binding)
  "Get the variable part of a single binding."
  (car binding))

(defun binding-val (binding)
  "Get the value part of a single binding."
  (cdr binding))

(defun get-binding (var bindings)
  "Find a (variable . value) pair in a binding list."
  (assoc var bindings))

(defun lookup (var bindings)
  "Get the value part (for var) from a binding list."
  (binding-val (get-binding var bindings)))

(defun extend-bindings (var val bindings)
  "Add a (var . value) pair to a binding list."
  (cons (cons var val)
	(if (eq bindings no-bindings)
	     nil
	     bindings)))

(defun variable-p (x)
   "Is x a variable (a symbol beginning with `?')?"
   (and (symbolp x) (equal (elt (symbol-name x) 0) #\?)))

(defun segment-pattern-p (pattern)
   "Is this a segment matching pattern: ((?* var) . pat)"
   (and (consp pattern)
	(starts-with (first pattern) '?*)))

(eval-when (load eval compile)
   (defmacro once-only (variables &rest body)
     "Returns the code built by BODY.  If any of VARIABLES
   might have side effects, they are evaluated once and stored
   in temporary variables that are then passed to BODY."
     (assert (every #'symbolp variables))
     (let ((temps nil))
       (dotimes (i (length variables)) (push (gensym) temps))
       `(if (every #'side-effect-free? (list .,variables))
	 (progn .,body)
	 (list 'let
	  ,`(list ,@(mapcar #'(lambda (tmp var)
				`(list ',tmp ,var))
			    temps variables))
	  (let ,(mapcar #'(lambda (var tmp) `(,var ',tmp))
			variables temps)
	    .,body)))))

   (defun side-effect-free? (exp)
     "Is exp a constant, variable, or function,
   or of the form (THE type x) where x is side-effect-free?"
     (or (atom exp) (constantp exp)
	 (starts-with exp 'function)
	 (and (starts-with exp 'the)
	      (side-effect-free? (third exp)))))

   (defmacro funcall-if (fn arg)
     (once-only (fn)
		`(if ,fn (funcall ,fn ,arg) ,arg)))

   (defmacro read-time-case (first-case &rest other-cases)
     "Do the first case, where normally cases are
   specified with #+ or possibly #- marks."
     (declare (ignore other-cases))
     first-case)

   (defun rest2 (x)
     "The rest of a list after the first TWO elements."
     (rest (rest x)))

   (defun find-anywhere (item tree)
     "Does item occur anywhere in tree?"
     (if (atom tree)
	 (if (eql item tree) tree)
	 (or (find-anywhere item (first tree))
	     (find-anywhere item (rest tree)))))

   (defun starts-with (list x)
     "Is x a list whose first element is x?"
     (and (consp list) (eql (first list) x)))
   )


 ;;; ==============================

(defun segment-match (pattern input bindings &optional (start 0))
   "Match the segment pattern ((?* var) . pat) against input."
   (let ((var (second (first pattern)))
	 (pat (rest pattern)))
     (if (null pat)
	 (match-variable var input bindings)
	 ;; We assume that pat starts with a constant
	 ;; In other words, a pattern can't have 2 consecutive vars
	 (let ((pos (position (first pat) input
			      :start start :test #'equal)))
	   (if (null pos)
	       fail
	       (let ((b2 (pat-match pat (subseq input pos) bindings)))
		 ;; If this match failed, try another longer one
		 ;; If it worked, check that the variables match
		 (if (eq b2 fail)
		     (segment-match pattern input bindings (+ pos 1))
		     (match-variable var (subseq input 0 pos) b2))))))))


(defun flatten (the-list)
  "Append together elements (or lists) in the list."
  (mappend #'mklist the-list))

(defun mklist (x)
  "Return x if it is a list, otherwise (x)."
  (if (listp x)
      x
      (list x)))

(defun mappend (fn the-list)	
  "Apply fn to each element of list and append the results."
  (apply #'append (mapcar fn the-list)))

(defun random-elt (choices)
  "Choose an element from a list at random."
  (elt choices (random (length choices))))

(setf (symbol-function 'find-all-if) #'remove-if-not)

(defun find-all (item sequence &rest keyword-args
                 &key (test #'eql) test-not &allow-other-keys)
  "Find all those elements of sequence that match item,
  according to the keywords.  Doesn't alter sequence."
  (if test-not
      (apply #'remove item sequence 
             :test-not (complement test-not) keyword-args)
      (apply #'remove item sequence
             :test (complement test) keyword-args)))

(defun dbg-indent (id indent format-string &rest args)
  "Print indented debugging info if (DEBUG ID) has been specified."
  (when (member id *dbg-ids*)
    (fresh-line *debug-io*)
    (dotimes (i indent) (princ "  " *debug-io*))
    (apply #'format *debug-io* format-string args)))

(defun dbg (id format-string &rest args)
  "Print debugging info if (DEBUG ID) has been specified."
  (when (member id *dbg-ids*)
    (fresh-line *debug-io*)
    (apply #'format *debug-io* format-string args)))

(defun do-debug (&rest ids)
  "Start dbg output on the given ids."
  (setf *dbg-ids* (union ids *dbg-ids*)))

(defun do-undebug (&rest ids)
  "Stop dbg on the ids.  With no ids, stop dbg altogether."
  (setf *dbg-ids* (if (null ids) nil
                      (set-difference *dbg-ids* ids))))

(provide :auxfuns)
