# Jeremy, 19 septembre 2022

- Reprise en main scripts, lancer XPs.
- Script Delta Choice:
  - XP alpha_comparisons déjà faite
  - lancement delta-choice. Choix de mettre que la petite dim mais de bien regarder une grille fine pour delta. SNR 100 et 30 (au lieu de 20 où on voit rien)
  - Change grosse dim à 1000,100 rang 20
  - Ajout (TODO pm) courbes moyennes depuis shootout
  - enlevé inner iters, pour tracer SNR
  - seed_idx --> seed
  - Résultats 15h00: (à plot correctement) à 100dB et 30dB, essentiellement pas de différence entre 0.05 et 0.4 pour delta. Accélération wrt delta=0 
  et delta=0.6 ou 0.9. TODO: redo with more iterations, convergence not reached for SNR=100 and delta too large.
  - Plot median cnt (subsample cnt sinon too big)
  - Observation: on peut juste mettre 100=iter_inner since 0.9 is virtually inner=2
  - Rerun with only 100 inner iter max
  - Final plots jolis :) (cnt un peu dense mais OK?) (error/time à couper en horizontal SNR=30 pas très informatif)
  - Refaire une dernière fois avec règle iter internes/outer reliées! et SNR=100 only.
  - Done --> C BEAU :) :)
  
- Script Alpha_comparisons:
  - Adding median plots
  - tweaking parameters --> relaunch? (delta, outer iters max)
  - Relaunched, only dims and alpha choice are changing. 
  - Conclusion Plots --> keep datasum.
  
- Script Frobenius, KL comparisons:
  - Freshening up
  - Adding delta input and cnt output (already supported in nnls) for HALS
  - Prepping the script plots.
  - TODO: run, add cnt plots. -> cnt not useful?
  - Done

- Trying to fix shootout so that dimensions can be stored and manipulated in format e.g. 100,100,5 directly in dataframe --> FAILED lol, too hard
  - solution: only group mnr after conv df :( 
  - TODO found fix: don't split, store whole list as string ^^

# Jeremy, 20 septembre 2022

- Reduce rank for big sizes, and single run at a time.
- Ran KL and Frobenius comparisons, KL very good Frobenius not so much
- Running audio_test script:
  - update script
  - remove Fevotte KL (same as Lee and Seung), added Proposed Frobenius without extrapolation (gamma1.8)
  - no median plots, show all runs (split Frobenius and KL)

- Adding NLS_Frobenius and NLS_KL to have scripts that only update H, to test the regression problem.
- Adding test script on audio (TODO add nnls from nn_fac)
- Adding simulated tests
- Ran script, stored results and figure
- TODO: regression test on hyperspectral (or audio?)
- TODO: same regression algorithms on KL

# Jeremy, 21 septembre 2022

- Add hals nnls to audio, rerun
- Adaptive grid for interpolation for each algorithm implemented
- bug in median/interp? Large variability wrt time but not iterations? I think not, just intrinsic time variabilityA
- Rerun HSI (and audio?) with larger noise for initial values, avg H0 sol is 250 so all inits were basically 0
  - Ended up normalizing data but not Wgt, then H is reasonably valued (mean H0 is 0.02)
- Rerender all plots with indiv algs grids, and fixed interpolation. Also 1200x900 figsize with final hand customization
- TODO: hyperspectral NMF et Faces experiment.

# Jeremy, 29 Novembre 2022

- Reprise du taf sur ce sujet, faire la partie XP du papier (rédaction et figures)
- TODOS:
  - write xp in article
  - benchmark nmf --> use to compare with toolboxes? (more code...)

# Jeremy, 05 Janvier 2023

- Checks de gammas : c le bordel
  - gamma manquant dans NMF KL non min
  - extrapol avec un max avec 1/L pas utile
# Jeremy, 06 décembre 2022

- Quyen changed the update (KL only), no more alpha, have to rerun tests
- TODO for Frobenius as well
- For real data, does not improve --> why
  - Testing the impact of sparsity of factors on improvements. My guess: improvement when SNR high and dense factors
  - Testing done. 
    - Synthetic: Sparsity has an impact on initialization, but not SNR. I cannot reproduce the behavior of the real data however.
    - Audio with fake data: sparsity has no impact (using max update). Low SNR reproduces somewhat the results, indicating that the problem I am solving with real data is just not a good NLS/NMF problem (no good solution).
- Replaced handmade KL div by scipy kldiv which works for x=0 y>=0

# Jeremy, 07 décembre 2022

- Spotted problem in manuscript: we use gamma 1 and 1.9 with pointwise max, not extrapol. Extrapol happens always with LeeSeung
- corrected bug with time on unix systems... do not use time.time() !!
- NNLS first start not the same as other first start... same problem !
- Making runs NLS fairer with tic before big computations
- Problem with hyperspcral nls --> was a weird problem with error computation in NLS. Wondering if same problem happens in NMF... quick verif says code is OK nevertheless, and never seen this problem elsewhere. Have no idea why error was bugged only for hyperspectral. Fixed anyway.
