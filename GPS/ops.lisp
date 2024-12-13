;;; -*- Mode: Lisp; Syntax: Common-Lisp; -*-
;;; Module: ops.lisp
;;; different worlds and operators for the GPS planner.
;;; bugs to vladimir kulyukin in canvas
;;; =========================================

;;; *block-ops Plan

;; Goal: A-ON-B
;; Consider: PUT-A-FROM-T-ON-B
;;   Goal: CLEAR-A
;;   Consider: PUT-C-FROM-A-ON-T
;;     Goal: CLEAR-C
;;     Goal: C-ON-A
;;   Action: PUT-C-FROM-A-ON-T
;;   Goal: CLEAR-B
;;   Goal: B-ON-C
;;   Consider: PUT-B-FROM-T-ON-C
;;     Goal: CLEAR-B
;;     Goal: CLEAR-C
;;     Goal: B-ON-T
;;     Goal: A-ON-T
;;   Action: PUT-B-FROM-T-ON-C
;;   Goal: A-ON-T
;; Action: PUT-A-FROM-T-ON-B                                                                                                                                                                   Goal: B-ON-C
;; ((START) (EXECUTE PUT-C-FROM-A-ON-T) (EXECUTE PUT-B-FROM-T-ON-C) (EXECUTE PUT-A-FROM-T-ON-B))

;;; *banana-ops* Plan

;; Goal: NOT-HUNGRY
;; Consider: EAT-BANANAS
;;   Goal: HAS-BANANAS                                                                                                                                                                           Consider: GRAB-BANANAS
;;     Goal: ON-CHAIR
;;     Consider: CLIMB-ON-CHAIR                                                                                                                                                                      Goal: ON-FLOOR
;;       Goal: CHAIR-AT-ROOM-CENTER
;;       Consider: MOVE-CHAIR-FROM-DOORWAY-TO-ROOM-CENTER
;;         Goal: ON-FLOOR
;;         Goal: CHAIR-AT-DOOR
;;       Action: MOVE-CHAIR-FROM-DOORWAY-TO-ROOM-CENTER
;;     Action: CLIMB-ON-CHAIR
;;     Goal: NOT-HAS-BALL
;;     Consider: DROP-BALL
;;       Goal: HAS-BALL
;;     Action: DROP-BALL
;;     Goal: CHAIR-AT-ROOM-CENTER
;;   Action: GRAB-BANANAS
;;   Goal: BANANAS-OPEN
;;   Consider: OPEN-BANANAS
;;     Goal: HAS-BANANAS
;;   Action: OPEN-BANANAS
;; Action: EAT-BANANAS
;; ((START) (EXECUTE MOVE-CHAIR-FROM-DOORWAY-TO-ROOM-CENTER) (EXECUTE CLIMB-ON-CHAIR) (EXECUTE DROP-BALL) (EXECUTE GRAB-BANANAS) (EXECUTE OPEN-BANANAS) (EXECUTE EAT-BANANAS))

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

;;; ================= Son At School ====================

(defparameter *school-world* '(son-at-home car-needs-battery
					   have-money have-phone-book))

(defparameter *school-ops*
  (list
    ;;; operator 1
   (make-op :action 'drive-son-to-school
	    :preconds '(son-at-home car-works)
	    :add-list '(son-at-school)
	    :del-list '(son-at-home))
   ;;; operator 2
   (make-op :action 'shop-installs-battery
	    :preconds '(car-needs-battery shop-knows-problem shop-has-money)
	    :add-list '(car-works))
   ;;; operator 3
   (make-op :action 'tell-shop-problem
	    :preconds '(in-communication-with-shop)
	    :add-list '(shop-knows-problem))
   ;;; operator 4
   (make-op :action 'telephone-shop
	    :preconds '(know-phone-number)
	    :add-list '(in-communication-with-shop))
   ;;; operator 5
   (make-op :action 'look-up-number
	    :preconds '(have-phone-book)
	    :add-list '(know-phone-number))
   ;;; operator 6
   (make-op :action 'give-shop-money
	    :preconds '(have-money)
	    :add-list '(shop-has-money)
	    :del-list '(have-money))))

;;; ================= Sussman's Anomaly ====================

(defparameter *blocks-world* '(a-on-t b-on-t c-on-a clear-c clear-b))

(defparameter *blocks-ops*
  (list
    (make-op 
      :action 'put-a-from-t-on-b
      :preconds '(clear-a clear-b b-on-c a-on-t)
      :add-list '(a-on-b)
      :del-list '(clear-b a-on-t))
    
    (make-op 
      :action 'put-c-from-a-on-t
      :preconds '(clear-c c-on-a)
      :add-list '(c-on-t clear-a)
      :del-list '(c-on-a))
    
    (make-op 
      :action 'put-b-from-t-on-c
      :preconds '(clear-b clear-c b-on-t a-on-t)
      :add-list '(b-on-c)
      :del-list '(b-on-t clear-c))
  ))
           
	    


;;; ================= village-safe ====================

(defparameter *dragon-world* '(dragon-alive forest-on-fire))

(defparameter *dragon-ops*
  (list
    (make-op 
      :action 'goto-mermaid
      :preconds '()
      :add-list '(fire-extingished)
      :del-list '(forest-on-fire))
    (make-op 
      :action 'goto-sword
      :preconds '(fire-extingished)
      :add-list '(has-sword)
      :del-list '(sword-in-forest))

    (make-op 
      :action 'goto-dragon
      :preconds '(has-sword)
      :add-list '(dragon-slain)
      :del-list '(dragon-alive))

    )
  ) (mapc #'convert-op *school-ops*) (mapc #'convert-op *dragon-ops*)

(provide :ops)
