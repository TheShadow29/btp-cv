(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(ansi-color-faces-vector
   [default default default italic underline success warning error])
 '(ansi-color-names-vector
   ["black" "#d55e00" "#009e73" "#f8ec59" "#0072b2" "#cc79a7" "#56b4e9" "white"])
 '(custom-enabled-themes (quote (deeper-blue)))
 '(inhibit-startup-screen t))
;; '(echo-keystrokes 0.1))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 )
(require 'package)

(set-face-attribute 'default nil :height 150)

;; If you want to use latest version
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/"))
(package-initialize)
(ac-config-default)
(add-to-list 'ac-modes 'latex-mode)
;; (add-to-list 'ac-modes 'python-mode)
(add-to-list 'ac-modes 'cmake-mode)
(add-to-list 'ac-modes 'xml-mode)
(add-to-list 'package-archives
			 '("elpy" . "http://jorgenschaefer.github.io/packages/"))
(global-auto-complete-mode t)
;(require 'sr-speedbar)
(setq c-default-style "linux"
		  c-basic-offset 4)

(setq backup-directory-alist '(("." . "~/.emacs.d/backup"))
  backup-by-copying t    ; Don't delink hardlinks
  version-control t      ; Use version numbers on backups
  delete-old-versions t  ; Automatically delete excess backups
  kept-new-versions 20   ; how many of the newest versions to keep
  kept-old-versions 5    ; and how many of the old
  )
(global-set-key (kbd "M-;") 'comment-dwim-2)


(setq TeX-auto-save t)
(setq TeX-parse-self t)

(setq custom-file "~/.emacs-custom.el")
(load custom-file)
(put 'upcase-region 'disabled nil)

(setq ido-enable-flex-matching t)
(setq ido-everywhere t)
(ido-mode 1)

(require 'package)
(add-to-list 'package-archives
			 '("elpy" . "http://jorgenschaefer.github.io/packages/"))
;;Py configurations
(elpy-enable)
(when (require 'elpy nil t)
  (elpy-enable))
;; (pyvenv-activate "/home/arka_s/internship_files/Viterbi-Internship/my_proj")
;; (pyvenv-activate "/home/arktheshadow/ARK-Linux/Programming/MachineLearning/ml_proj")
;; (pyvenv-activate "/home/arktheshadow/ARK-Linux/Programming/scikit/sk_venv_p3")
(pyvenv-activate "/home/arktheshadow/anaconda2/envs/pyt_venv")
(setq python-shell-interpreter "ipython"
	  python-shell-interpreter-args "--simple-prompt -i")
(add-hook 'before-save-hook 'whitespace-cleanup)
(setq elpy-rpc-backend "jedi")
;; (require 'python)
;; (setq python-shell-interpreter "ipython")
;; (setq python-shell-interpreter-args "--pylab")

(global-set-key (kbd "M-m") 'magit-status) ; Alt+m
(global-auto-revert-mode t)
(custom-set-variables
  ;; custom-set-variables was added by Custom.
  ;; If you edit it by hand, you could mess it up, so be careful.
  ;; Your init file should contain only one such instance.
  ;; If there is more than one, they won't work right.
 '(tab-stop-list (quote (4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96 100 104 108 112 116 120))))
