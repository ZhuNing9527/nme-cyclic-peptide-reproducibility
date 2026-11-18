# NMe-Ala joint acquisition benchmark precommit v1

## Status

FROZEN AFTER INTERNAL VALIDATION SPLIT

FROZEN BEFORE ANY NEW VALIDATION OR ACQUISITION QM

FROZEN BEFORE ANY ACTIVE-LEARNING SELECTION

## Scientific purpose

Test whether periodic Gaussian-process active learning
can resolve the joint phi-psi-omega residual surface
more efficiently than geometry-only sampling.

Active learning is therefore treated as a testable
sampling method, not as an AI label added after the
fact.

## Fixed data roles

Initial labeled training seeds:

54 existing identifiability QM/M0 states.

Locked internal validation:

144 new grid states.

Acquisition pool:

1566 new grid states.

PPII:

six separately sealed off-grid validation states.

The 144 internal-validation states are forbidden from
all acquisition calculations.

## Acquisition arms

Three acquisition strategies are compared under matched
new-QM point budgets.

ARM 1:

PERIODIC_GP_ACTIVE_LEARNING

ARM 2:

PERIODIC_DISTANCE_MAXIMIN

ARM 3:

FIXED_SEED_RANDOM

## Cumulative budgets

New acquisition labels per arm:

24

48

96

192

The cumulative budget checkpoints are fixed before any
new QM value is observed.

## Batch size

Periodic GP active learning acquires:

24

states per sequential batch.

Thus the 192-point maximum contains exactly eight
active-learning batches.

## Common fitting target

Every acquisition arm ultimately receives labels:

R(phi,psi,omega)
=
E_QM
-
E_M0

on the exact same accepted QM geometry.

Raw QM energy alone is not the learning target.

For numerical conditioning only, a training-set constant
mean may be removed and added back.

A constant shift does not alter the physical residual
surface or acquisition uncertainty.

## Periodic GP kernel

The active-learning surrogate uses a periodic
squared-exponential kernel on the three-dimensional
torus:

k(x,x')
=
sigma_f^2
*
exp[
-2 * (
sin^2((phi-phi')/2) / ell_phi^2
+
sin^2((psi-psi')/2) / ell_psi^2
+
sin^2((omega-omega')/2) / ell_omega^2
)
].

Angles entering this expression are in radians.

The three length scales are independent.

## GP numerical noise

The target data are deterministic QM/MM evaluations.

No observation-noise model is fitted.

A fixed numerical nugget may be used only for stable
linear algebra.

It must not exceed:

1e-6 kcal^2/mol^2.

## GP hyperparameters

At the start of every active-learning batch:

sigma_f;

ell_phi;

ell_psi;

ell_omega

are re-estimated using only labels currently belonging
to that active-learning arm.

The 144 internal-validation labels are never used.

The 1566 unqueried acquisition labels are unavailable.

Bounds:

sigma_f:

0.10 to 20.0 kcal/mol.

ell_phi:

0.15 to 3.0.

ell_psi:

0.15 to 3.0.

ell_omega:

0.15 to 3.0.

Optimization is deterministic multi-start
log-marginal-likelihood minimization.

The exact implementation and deterministic initial
hyperparameter starts must be frozen before the first
active-learning selection.

## Joint-aware uncertainty

For candidate:

x = (phi,psi,omega),

define the corresponding trans-reference coordinate:

x_T = (phi,psi,180 degrees).

The GP posterior provides:

Var[R(x)]

and the contrast variance:

Var[J(x)]
=
Var[
R(x)
-
R(x_T)
].

The primary active-learning acquisition score is:

score(x)
=
SD[R(x)]
+
SD[J(x)].

Therefore both:

the trans-reference backbone surface A(phi,psi)

and

the omega-dependent joint correction J(phi,psi,omega)

can drive sampling.

For omega=180 degrees:

SD[J] = 0

by definition of the contrast, while SD[R] remains
available to sample the A surface.

## GP batch selection

Within each 24-state batch:

GP hyperparameters remain fixed.

Candidate points are selected greedily.

After every within-batch selection, posterior covariance
is updated as if the selected coordinate were added with
unknown value.

Because GP posterior variance conditioning depends only
on coordinates, this fantasy variance update uses no
invented energy label.

This prevents a batch from collapsing into one local
region.

Tie breaking:

1. larger acquisition score;

2. larger minimum periodic angular distance to the
   arm's current labeled-plus-selected set;

3. lexicographically smaller state ID.

After the 24 QM/M0 labels are obtained, GP
hyperparameters may be refitted for the next batch.

## Maximin baseline

The complete 192-state maximin sequence is frozen before
new QM.

Starting reference coordinates are the 54 existing
identifiability training seeds.

At every step select the acquisition-pool state that
maximizes its minimum 3D periodic angular Euclidean
distance to:

all 54 initial seed coordinates;

plus all previously selected maximin states.

Ties are broken lexicographically by state ID.

No energy is used.

## Random baseline

The complete 192-state random sequence is frozen before
new QM.

Procedure:

sort all 1566 acquisition-pool state IDs
lexicographically;

apply Python Mersenne-Twister shuffle with integer seed:

20260822

take the first 192 states.

The resulting explicit sequence is frozen to file, so
future software-version behavior cannot change the
benchmark.

## Computational de-duplication

The same physical state may be selected by more than
one arm.

Its QM/M0 calculation may be performed only once and
reused computationally.

However arm-specific learning curves remain logically
separate.

A label obtained because another arm selected the same
state is unavailable to an arm until that state appears
within that arm's own frozen/selected sequence.

Thus computational de-duplication cannot create
cross-arm information leakage.

## Acquisition benchmark evaluation

At cumulative budgets:

24

48

96

192

each arm is evaluated on the same locked 144-state
internal validation set.

The evaluation model and representation-selection rules
will be frozen in a separate precommit before
validation labels are opened.

No acquisition arm is declared superior in the present
step.

## Barrier-state boundary

No acquisition strategy may censor a candidate merely
because its constrained QM energy is high.

The candidate pool includes cis, trans, and
intermediate-omega barrier regions by construction.

## Energy-relevance boundary

The first acquisition benchmark is uncertainty /
geometry driven.

No Boltzmann or low-energy weighting enters the primary
acquisition score.

Energy relevance may later be analyzed secondarily, but
it cannot erase high-energy or barrier-region coverage.

## PPII boundary

PPII remains sealed from:

GP fitting;

hyperparameter fitting;

active-learning scoring;

maximin selection;

random selection;

benchmark evaluation.

## Current authorization

Authorized after this precommit:

implement the deterministic periodic GP algorithm;

freeze its exact numerical implementation;

generate validation QM/M0;

generate matched acquisition-arm QM/M0 labels
sequentially.

Not authorized:

inspect PPII;

fit a final production potential;

change the acquisition budgets after seeing validation;

move validation states into training.
