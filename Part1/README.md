# Part 1 — Randomly Wired Neural Networks (RWNNs)

**Source article:** [On Randomly Wired and Optimally Wired Feed-Forward Neural Networks — Part 1](https://medium.com/@stylianoskyriacou/on-randomly-wired-and-optimally-wired-feed-forward-neural-networks-6969e9c929b4)

## Motivation

Most neural networks we use in practice are **hand-designed**: a human
researcher chooses the number of layers, the number of neurons per layer,
and the pattern of connections between them (fully connected, convolutional,
residual, etc.). Facebook AI Research (FAIR) published a result showing that
**randomly wired neural networks** can match — and sometimes beat —
painstakingly designed architectures like ResNet and ShuffleNet on ImageNet.

That is a surprising result. It suggests the detailed topology of a network
matters less than we tend to assume, as long as a few structural constraints
are respected. Part 1 of this series reproduces that idea in the simplest
possible setting so the behaviour can be inspected directly: a small
feed-forward regression problem.

## The RWNN construction

A Randomly Wired Neural Network (RWNN) here is a directed acyclic graph of
neurons with the following constraints:

- **Input nodes** (drawn red in the original article) have **only outgoing**
  connections. They receive the raw features.
- **Bias nodes** (drawn purple) likewise have **only outgoing** connections
  and always output the value `1`.
- **Output nodes** (drawn blue) have **only incoming** connections.
- **Hidden nodes** are indexed. A hidden node with index `i` may only
  receive input from nodes with index `< i`. This ordering guarantees the
  graph is a DAG — no cycles, and therefore a well-defined forward pass.

Within those rules, each synaptic connection is sampled randomly. Two
networks generated this way will almost never share the same wiring, yet
both are valid feed-forward networks that can be trained with ordinary
backpropagation.

Each connection carries a trainable weight. Each hidden and output neuron
applies a nonlinearity to the weighted sum of its inputs.

## Toy experiment

To make the behaviour easy to visualise, Part 1 uses a two-dimensional
regression task:

$$f(x_1, x_2) = x_1^2 + x_2^2$$

- **Training set:** 100 points sampled uniformly from `[-1, 1] × [-1, 1]`.
- **Test set:** 5,000 points from the same range.
- **Training:** standard backpropagation on the mean squared error.

The RWNN has 1 or 2 input nodes, a small number of bias nodes, a handful of
hidden nodes, and 1 output node. Because the topology is random, the number
and location of synapses varies — but the DAG constraint keeps the forward
pass deterministic and differentiable.

## Findings

- A **single randomly wired network** successfully learns the quadratic
  bowl `x₁² + x₂²`.
- Test-set predictions track the ground truth closely inside the training
  range — the random topology is not a handicap for this problem.
- The experiment establishes a baseline: RWNNs are trainable, stable, and
  expressive enough for smooth function approximation.

## Why this matters for the rest of the series

Part 1 is the foundation. It shows that if we wire neurons randomly (under
the DAG + role constraints), backprop still works. That licence to
randomise the topology is what the next two parts exploit:

- **Part 2** treats the randomness as a source of *diversity* and builds an
  ensemble of RWNNs to estimate predictive uncertainty.
- **Part 3** replaces the randomness with a *search*: evolving the topology
  to find small, accurate networks — Optimally Wired Neural Networks.

## Key takeaways

- Topology can be random as long as it is a DAG with well-defined input,
  bias, hidden and output roles.
- Such networks are trainable with vanilla backpropagation.
- On a simple regression task, random wiring is sufficient to learn and
  generalise — no hand-designed architecture required.
