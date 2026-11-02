# Bibliographie

Publications à partir desquelles les méthodes ont été implémentées, et
références des choix de conception. Aucun code tiers n'a été consulté
pour écrire les méthodes de ce paquet (voir
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) §3).

## Méthodes linéaires

- N. Wiener, *Extrapolation, Interpolation, and Smoothing of Stationary
  Time Series*, MIT Press, 1949. — `wiener`.
- D. J. Field, "Relations between the statistics of natural images and
  the response properties of cortical cells", *JOSA A* 4(12),
  2379-2394, 1987. — spectre en 1/f² des images naturelles (`wiener`).
- F. Orieux, J.-F. Giovannelli, T. Rodet, "Bayesian estimation of
  regularization and point spread function parameters for Wiener-Hunt
  deconvolution", *JOSA A* 27(7), 1593-1607, 2010. —
  `wiener_unsupervised` (implémentation de scikit-image).
- A. N. Tikhonov, V. Y. Arsenin, *Solutions of Ill-Posed Problems*,
  Winston, 1977. — `tikhonov`.
- R. C. Gonzalez, R. E. Woods, *Digital Image Processing*, 4ᵉ éd.,
  Pearson, 2018, chap. 5. — filtre inverse tronqué (`inverse`).

## Choix du paramètre de régularisation

- G. H. Golub, M. Heath, G. Wahba, "Generalized cross-validation as a
  method for choosing a good ridge parameter", *Technometrics* 21(2),
  215-223, 1979. — GCV.
- V. A. Morozov, "On the solution of functional equations by the method
  of regularization", *Soviet Math. Dokl.* 7, 414-417, 1966. — principe
  de discrépance.
- P. C. Hansen, "Analysis of discrete ill-posed problems by means of the
  L-curve", *SIAM Review* 34(4), 561-580, 1992. — L-curve.
- G. Blanchet, L. Moisan, "An explicit sharpness index related to global
  phase coherence", *ICASSP*, 2012. — indice SI (eoqual-metrics).
- A. Leclaire, L. Moisan, "No-reference image quality assessment and
  blind deblurring with sharpness metrics exploiting Fourier phase
  information", *JMIV* 52(1), 145-172, 2015. — SI comme critère de
  réglage sans référence.

## Méthodes itératives

- W. H. Richardson, "Bayesian-based iterative method of image
  restoration", *JOSA* 62(1), 55-59, 1972 ; L. B. Lucy, "An iterative
  technique for the rectification of observed distributions", *AJ*
  79(6), 745-754, 1974. — `richardson_lucy`.
- D. S. C. Biggs, M. Andrews, "Acceleration of iterative image
  restoration algorithms", *Applied Optics* 36(8), 1766-1775, 1997. —
  accélération de `richardson_lucy`.
- N. Dey, L. Blanc-Féraud, C. Zimmer, P. Roux, Z. Kam, J.-C.
  Olivo-Marin, J. Zerubia, "Richardson-Lucy algorithm with total
  variation regularization for 3D confocal microscope deconvolution",
  *Microscopy Research and Technique* 69(4), 260-266, 2006. — option
  `tv_weight`.
- L. Landweber, "An iteration formula for Fredholm integral equations of
  the first kind", *American Journal of Mathematics* 73(3), 615-624,
  1951 ; M. Bertero, P. Boccacci, *Introduction to Inverse Problems in
  Imaging*, IOP, 1998. — `landweber`.
- J. Barzilai, J. M. Borwein, "Two-point step size gradient methods",
  *IMA J. Numer. Anal.* 8(1), 141-148, 1988. — `landweber(acceleration=True)`.
- S. Bonettini, R. Zanella, L. Zanni, "A scaled gradient projection
  method for constrained image deblurring", *Inverse Problems* 25(1),
  015002, 2009. — piste pour Richardson-Lucy (REFERENCE_TECHNIQUE.md §12).

## Méthodes variationnelles

- A. Beck, M. Teboulle, "A fast iterative shrinkage-thresholding
  algorithm for linear inverse problems", *SIAM J. Imaging Sci.* 2(1),
  183-202, 2009. — `fista_wavelet`, pas en PSF de `blind_l0`.
- I. Daubechies, M. Defrise, C. De Mol, "An iterative thresholding
  algorithm for linear inverse problems with a sparsity constraint",
  *Comm. Pure Appl. Math.* 57(11), 1413-1457, 2004. — ISTA.
- M. A. T. Figueiredo, R. D. Nowak, "An EM algorithm for wavelet-based
  image restoration", *IEEE TIP* 12(8), 906-916, 2003. — décalages
  aléatoires (*cycle spinning*).
- L. Condat, "A primal-dual splitting method for convex optimization
  involving Lipschitzian, proximable and linear composite terms",
  *JOTA* 158(2), 460-479, 2013 ; B. C. Vũ, "A splitting algorithm for
  dual monotone inclusions involving cocoercive operators", *Adv.
  Comput. Math.* 38(3), 667-681, 2013. — `tv_primal_dual`.
- A. Chambolle, T. Pock, "A first-order primal-dual algorithm for convex
  problems with applications to imaging", *JMIV* 40(1), 120-145, 2011.
- L. I. Rudin, S. Osher, E. Fatemi, "Nonlinear total variation based
  noise removal algorithms", *Physica D* 60, 259-268, 1992.

## Plug-and-Play

- D. Geman, C. Yang, "Nonlinear image recovery with half-quadratic
  regularization", *IEEE TIP* 4(7), 932-946, 1995. — HQS.
- S. V. Venkatakrishnan, C. A. Bouman, B. Wohlberg, "Plug-and-play
  priors for model based reconstruction", *IEEE GlobalSIP*, 2013. —
  `pnp_admm`.
- S. H. Chan, X. Wang, O. A. Elgendy, "Plug-and-play ADMM for image
  restoration: fixed-point convergence and applications", *IEEE TCI*
  3(1), 84-98, 2017. — continuation de `pnp_admm`.
- K. Zhang, W. Zuo, S. Gu, L. Zhang, "Learning deep CNN denoiser prior
  for image restoration", *CVPR*, 2017. — `pnp_hqs`.
- K. Zhang, Y. Li, W. Zuo, L. Zhang, L. Van Gool, R. Timofte,
  "Plug-and-play image restoration with deep denoiser prior", *IEEE
  TPAMI* 44(10), 6360-6376, 2021. — calendrier de σ et λ = 0.23 de
  `pnp_hqs`.

## Déconvolution aveugle

- L. Xu, S. Zheng, J. Jia, "Unnatural L0 sparse representation for
  natural image deblurring", *CVPR*, 2013. — `blind_l0`.
- J. Pan, Z. Hu, Z. Su, M.-H. Yang, "Deblurring text images via
  L0-regularized intensity and gradient prior", *CVPR*, 2014. —
  `blind_l0` (λ, seuil de la PSF, calendrier).
- L. Xu, C. Lu, Y. Xu, J. Jia, "Image smoothing via L0 gradient
  minimization", *ACM TOG* 30(6), 2011. — pas en image (HQS).
- S. Cho, S. Lee, "Fast motion deblurring", *ACM TOG* 28(5), 2009. —
  estimation de la PSF dans le domaine des gradients.
- D. Krishnan, T. Tay, R. Fergus, "Blind deconvolution using a
  normalized sparsity measure", *CVPR*, 2011.
- D. Perrone, P. Favaro, "Total variation blind deconvolution: the devil
  is in the details", *CVPR*, 2014. — approche multi-échelle.
- J. Bolte, S. Sabach, M. Teboulle, "Proximal alternating linearized
  minimization for nonconvex and nonsmooth problems", *Math. Program.*
  146, 459-494, 2014. — PALM (essayé et écarté, REFERENCE_TECHNIQUE.md
  annexe A).
- J. Duchi, S. Shalev-Shwartz, Y. Singer, T. Chandra, "Efficient
  projections onto the ℓ1-ball for learning in high dimensions", *ICML*,
  2008. — projection sur le simplexe.

## Données incomplètes et bords

- M. S. C. Almeida, M. A. T. Figueiredo, "Deconvolving images with
  unknown boundaries using the alternating direction method of
  multipliers", *IEEE TIP* 22(8), 3074-3086, 2013. — `boundary="unknown"`.
- A. P. Dempster, N. M. Laird, D. B. Rubin, "Maximum likelihood from
  incomplete data via the EM algorithm", *J. R. Stat. Soc. B* 39(1),
  1-38, 1977. — remplissage des données manquantes (`mask`, `saturation`).
- M. Bertero, P. Boccacci, "A simple method for the reduction of boundary
  effects in the Richardson-Lucy approach to image deconvolution",
  *Astronomy & Astrophysics* 437, 369-374, 2005. — Richardson-Lucy à
  bords inconnus.
- N. Antipa, G. Kuo, R. Heckel, B. Mildenhall, E. Bostan, R. Ng, L. Waller,
  "DiffuserCam: lensless single-exposure 3D imaging", *Optica* 5(1), 1-9,
  2018. — ADMM avec recadrage (bords inconnus).

## Bords et PSF

- L. Moisan, "Periodic plus smooth image decomposition", *JMIV* 39(2),
  161-179, 2011. — `boundary="perdecomp"` (eoqual-core).
- D. L. Donoho, I. M. Johnstone, "Ideal spatial adaptation by wavelet
  shrinkage", *Biometrika* 81(3), 425-455, 1994. — estimation de σ
  (eoqual-core).
- M. Born, E. Wolf, *Principles of Optics*, 7ᵉ éd., Cambridge University
  Press, 1999. — tache d'Airy.
- I. Trujillo, J. A. L. Aguerri, J. Cepa, C. M. Gutiérrez, "The effects
  of seeing on Sérsic profiles — II. The Moffat PSF", *MNRAS* 328(3),
  977-985, 2001. — β = 4.765 de `moffat_psf`.
- V. N. Mahajan, *Optical Imaging and Aberrations, Part II: Wave
  Diffraction Optics*, SPIE Press, 2001. — pupille annulaire
  (`annular_airy`, `system_psf`).

## Validation

- S. Diamond, S. Boyd, "CVXPY: a Python-embedded modeling language for
  convex optimization", *JMLR* 17(83), 1-5, 2016. — solution de référence
  de `tests/test_reference_cvxpy.py`.

## Pistes (REFERENCE_TECHNIQUE.md §12)

- Y. Zhu, K. Zhang, J. Liang, J. Cao, B. Wen, R. Timofte, L. Van Gool,
  "Denoising diffusion models for plug-and-play image restoration",
  *CVPR Workshops*, 2023. — DiffPIR.
- H. Chung, J. Kim, M. T. McCann, M. L. Klasky, J. C. Ye, "Diffusion
  posterior sampling for general noisy inverse problems", *ICLR*, 2023. —
  DPS.
- M. A. T. Figueiredo, J. M. Bioucas-Dias, "Restoration of Poissonian
  images using alternating direction optimization", *IEEE TIP* 19(12),
  3133-3145, 2010. — PIDAL (Poisson + TV).
- S. Prigent et al., "SPITFIR(e): a supermaneuverable algorithm for
  fast denoising and deconvolution of 3D fluorescence microscopy images
  and videos", *Scientific Reports* 13, 1489, 2023.

## Pour aller plus loin (cours et tutoriels)

Ressources publiques qui ont guidé la progression de
[TUTORIEL.md](TUTORIEL.md) ; aucun contenu n'en est repris.

- C. Chaux, TP « Méthodes variationnelles pour l'imagerie », École
  GEOMDATA 2018 — https://www.i2m.univ-amu.fr/~caroline.chaux/enseignements.html
- G. Peyré, Numerical Tours, « Image Deconvolution using Variational
  Method » — https://github.com/gpeyre/numerical-tours
- H. Talbot, scripts du manuel d'optimisation —
  https://github.com/hugues-talbot/optimization_textbook
- M. Ponti, cours de traitement d'images (ICMC-USP) —
  https://github.com/maponti/imageprocessing_course_icmc
- Waller Lab, tutoriel DiffuserCam — https://github.com/Waller-Lab/DiffuserCam-Tutorial
- Stanford EE367 « Computational Imaging » — https://stanford.edu/class/ee367/
- Dépôt BBii-Decon (gradient projeté de Barzilai-Borwein, microscopie) —
  https://github.com/gdellaire/BBii-Decon
- S. Prigent, sdeconv (SPITFIR(e), Wiener, Richardson-Lucy, PyTorch) —
  https://github.com/sylvainprigent/sdeconv
