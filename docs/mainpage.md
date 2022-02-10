# Gen Inference and Learning Template Library (GenTL}   {#mainpage}

This documentation is a work-in-progress.

## Concepts

See include/gentl/concepts.h for a formal definition of these concepts.

Using GenTL typically involves defining your own types that adhere to a set of concepts.
Note that there is no abstract class for each of these concepts in GenTL (i.e. there is no general abstract `Trace` class).

### GenerativeFunction

An object whose call operator `()` returns a BoundGenerativeFunction.

The type of the argument to this operator is related to the InputGradient and InputChange types.

### BoundGenerativeFunction

An object that is capable of generating a Trace, given a random number generator (RNG), a ParameterStore object, and optionally a ChoiceBuffer that describes constraints that the resulting Trace should satisfy.

There are two template member functions that are used to obtain Traces:

Sample from the joint distribution, and optionally compute and cache gradients with respect to trainable parameters.
```
template <typename RNG, typename ParameterStore>
std::unique_ptr<Trace> simulate(RNG&, ParameterStore&, const SimulateOptions&) const
```
See gentl::SimulateOptions.

Sample from an alternate distribution, and return an importance weight accounting for the difference,
and optionally compute and cache gradients with respect to trainable parameters.
```
template <typename RNG, typename ParameterStore>
std::pair<std::unique_ptr<Trace>, double> generate(RNG&, ParameterStore&, const ChoiceBuffer&, const GenerateOptions&) const
```
See gentl::GenerateOptions.

There are two variants that write into a reference to a trace object instead of returning a new trace object:

```
template <typename RNG, typename ParameterStore>
void simulate(RNG&, ParameterStore&, const SimulateOptions&, Trace&) const
```


Equivalent to `generate` but does not need to return a trace:
```
template <typename RNG, typename ParameterStore>
double assess(RNG&, ParameterStore&, const ChoiceBuffer&) const
```


### Trace

A possible sample from the probability distribution of a BoundGenerativeFunction.

Traces typically do not have public constructors, and are instead obtained from the `simulate` and `generate` functions of a BoundGenerativeFunction.


Member functions:

Modify the state of a trace:
```
template <typename RNG>
double update(RNG&, const InputChange&, const ForwardChoiceBuffer&, const UpdateOptions&)
```
See gentl::UpdateOptions.
Note that `BackwardChoiceBuffer` and `ValueChange` references returned by `update` will become undefined after the next call to `update` or `revert` (below) or `fork` (below).

Return the constraints, that, if passed to `update`, would revert the most recent `update` (only valid after a call to `update` and before a call to `revert` or `fork`):
```
const ChoiceBuffer& backward_constraints()
```

Return an object representing the change in return value.
```
const ValueChange& return_value_change()
```

Revert to the state before the most recent `update` call for which the `save` flag is set.
```
void revert()
```

Return a new trace that is observationally independent from `this`:
```
std::unique_ptr<Trace> fork()
```

Return a view into the value of some subset of the random choices in the trace:
```
const ChoiceBuffer& choices(const Selection&) const
```

NOTE: gentl::mcmc::mh requires that the proposal implement `choices(const gentl::selections:all)`.

Return the log joint probability density:
```
double score() const
```

Return the return value of the trace, which is in general different from the choices.
```
Output return_value() const
```

Compute the gradient of the log joint density with respect to the parameters, and accumulate this gradient into the given GradientAccumulator object.
```
void parameter_gradient(GradientAccumulator&, double scaler=1.0)
```

Compositional variant:
```
InputGradient parameter_gradient(GradientAccumulator&, const OutputGradient&, double scaler=1.0)
```

Return the gradient of the log joint density with respect to the choices:
```
const ChoiceBuffer& choice_gradient(const Selection&)
```

Compositional variant:
```
std::pair<InputGradient, const ChoiceBuffer&> choice_gradient(const Selection&, const OutputGradient&)
```

### ChoiceBuffer

Stores the values of random choices.

To support MALA and HMC, it must support the following elementwise arithmetic operators: `+`, `-`, `*` with an element of the same type as well as with `double`, `+=`, unary `-`.

### ChoiceSelection

A set of random choices.

### Change

A change to the input or output of a generative function.

### ParameterStore

Storage for parameters that support gradient-based learning.
They maintain a current value for the parameters and a value for the gradient with respect to the parameters that is used by gradient-based learning algorithms including stochastic gradient descent.

### ParameterGradientAccumulator

ParameterGradientAccumulators are constructed from a ParameterStore object.
Separate ParameterGradientAccumulator objects can be used to accumulate gradients with respect to parameters in parallel.

The following function accumulates the gradients of the source ParameterStore object (and resets the ParameterGradientAccumulator):
```
void update_gradient()
```

## Functionality

### Modeling

#### Sequential and Temporal Models

- gentl::modeling::sequence::Model

- gentl::modeling::sequence::Trace

### Inference and Learning

#### Markov chain Monte Carlo (MCMC)

- gentl::mcmc::mh

- gentl::mcmc::hmc

- gentl::mcmc::mala

#### Sequential Monte Carlo (SMC)

- gentl::smc::ParticleSystem

#### Sampling importance resampling (SIR)

- gentl::sir::rolling_importance_resampling

#### Maximum likelihood

- gentl::sgd::train_supervised

## Types

gentl::change::NoChange and singleton gentl::change::no_change

gentl::change::UnknownChange
