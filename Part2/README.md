# Part 2 — Ensembles of RWNNs for Uncertainty Quantification

**Source article:** [On Randomly Wired and Optimally Wired Feed-Forward Neural Networks — Part 2](https://medium.com/@stylianoskyriacou/on-randomly-wired-and-optimally-wired-feed-forward-neural-networks-part-2-2e1fbd4166a0)

## From a single RWNN to an ensemble

Part 1 showed that a single randomly wired network can fit a smooth
function. Part 2 asks a different question: **how confident is the network
about its predictions, especially outside the training region?**

A single deterministic network gives one number per input — no error bar,
no honest admission that an input is far from anything it has seen. The
trick in Part 2 is to use the **natural diversity of random wiring** as a
source of epistemic uncertainty: if many independently wired networks,
trained on overlapping but different data, agree on a prediction, we can
be confident; if they disagree, we should not be.

This is conceptually the same idea behind **Random Forests**: many weak,
decorrelated learners voted together produce both a better mean prediction
and a usable measure of spread.

## Experimental setup

- **Architecture:** each network has 1 input node, 2 bias nodes,
  10 hidden nodes, and 1 output node. Within that skeleton, the wiring is
  random and different for every network.
- **Ensemble size:** 100 RWNNs, each with a unique topology and a unique
  random weight initialisation.
- **Data bagging:** each network is trained on a random **90%** subsample
  of the training set — the same bootstrap-style trick used in Random
  Forests to decorrelate members of the ensemble.
- **Training:** standard backpropagation on each member independently.
- **Inference:** at test time, every network produces its own prediction
  for the same input, giving a distribution of 100 outputs per point.

## Turning 100 predictions into uncertainty

For each test input, the ensemble yields a small empirical distribution.
The article summarises it with percentile statistics:

- **P50 (median)** — the central prediction of the ensemble.
- **P10–P90 band** — a "likely range" capturing the middle 80% of members.
- **Min–Max band** — the full envelope across all 100 networks.

The shape of the P10–P90 band and the min–max envelope as a function of
the input tells the story:

- **Inside the training range** the bands are tight — the networks
  converge on the same answer. This is *interpolation confidence*.
- **Outside the training range** the bands fan out dramatically — each
  network extrapolates in its own idiosyncratic way, and they disagree.
  This is *extrapolation uncertainty*, and it is exactly what we want the
  model to report.

## Findings

- The ensemble produces a sensible median prediction that tracks the
  underlying function inside the training distribution.
- The spread across members is a **faithful signal of uncertainty**: narrow
  where the data supports the prediction, wide where it does not.
- The observation from the article is blunt: the networks do *poorly*
  extrapolating outside the training range, and the ensemble correctly
  flags that region with large variance rather than overconfident
  predictions.

## Why this is useful

- It is a cheap Bayesian-flavoured uncertainty estimator — no variational
  inference, no MC dropout, no Gaussian process kernel. Just train many
  cheaply-wired small networks and look at the spread.
- The random wiring provides structural diversity that weight-init
  randomness alone would not. Two networks with identical architecture and
  different seeds tend to agree more than two networks with different
  architectures.
- Bagging (90% subsamples) adds data diversity on top of topology
  diversity, further decorrelating the members.

## Limits the article is honest about

- Performance degrades quickly outside the training range — the models
  don't know how to extrapolate, and no ensemble can rescue a model from
  data it has never seen. The virtue of the ensemble is that it **knows it
  doesn't know**.
- 100 members is an arbitrary choice; tighter bands require more members
  or more data, and the compute scales linearly.

## Why this matters for the rest of the series

Part 2 uses random wiring as a feature, not a bug — the randomness is a
source of useful diversity. Part 3 flips the logic one more time: instead
of embracing randomness, it *searches* over topologies to find a single,
small, accurate one. That transition sets up the move from **Randomly**
Wired to **Optimally** Wired networks.

## Key takeaways

- An ensemble of RWNNs gives both a median prediction and a usable
  uncertainty band for free.
- Percentile statistics (P10/P50/P90 + min/max) translate ensemble spread
  into interpretable intervals.
- Topology diversity + data bagging = decorrelated learners, the same
  principle that makes Random Forests work.
- Extrapolation remains hard — but the ensemble makes that hardness
  visible instead of hiding it.
