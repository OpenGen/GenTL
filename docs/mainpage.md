# Gen Inference and Learning Template Library (GenTL}   {#mainpage}

## Concepts

Using GenTL typically involves defining your own types that adhere to a set of concepts.
GenTL does not provide any abstract classes that define these concepts.

### GenerativeFunction

An object whose call operator `()` returns a BoundGenerativeFunction.

### BoundGenerativeFunction

An object that is capable of generating a Trace, given a random number generator, a parameter object, and optionally a ChoiceBuffer that describes constraints that the resulting Trace should satisfy.

There are two member functions that are used to generate Traces:

- `simulate` (two variants)

- `generate` (two variants)

### Trace

A possible sample from the probability distribution of a BoundGenerativeFunction.

Traces typically do not have public constructors, and are instead obtained from the `simulate` and `generate` functions of a BoundGenerativeFunction.

Member functions:

- `update`

- `revert`

- `fork`

- `choices`

- `score`

- `parameter_gradient`

- `choice_gradient`

### ChoiceBuffer

Storage for the values of random choices.

### ChoiceSelection

A set of random choices.

### Change

A change to the input or output of a generative function.

### ParameterStore

Storage for parameters that support gradient-based learning.

### ParameterGradientAccumulator

Member functions:

- `update_gradient`

## Functionality

### Markov chain Monte Carlo (MCMC)

- gentl::mcmc::mh

- gentl::mcmc::hmc

- gentl::mcmc::mala

### Sequential Monte Carlo (SMC)

- gentl::smc::ParticleSystem

### Sampling importance resampling (SIR)

- gentl::sir::rolling_importance_resampling

### Maximum likelihood

- gentl::sgd::train_supervised
