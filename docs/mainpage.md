# Gen Inference and Learning Template Library (GenTL}   {#mainpage}

This documentation is a work-in-progress.

## Concepts

Using GenTL typically involves defining your own types that adhere to a set of concepts.
GenTL does not provide any abstract classes that define these concepts.

### GenerativeFunction

An object whose call operator `()` returns a BoundGenerativeFunction.

The input to the call is called the Input and related to the InputGradient and InputChange types.

### BoundGenerativeFunction

An object that is capable of generating a Trace, given a random number generator, a parameter object, and optionally a ChoiceBuffer that describes constraints that the resulting Trace should satisfy.

There are two template member functions that are used to obtain Traces:

```
template <typename RNG, typename Parameters>
std::unique_ptr<Trace> simulate(RNG& rng, Parameters&, bool gradient) const
```

```
template <typename RNG, typename Parameters>
std::pair<std::unique_ptr<Trace>, double> generate(RNG& rng, Parameters& parameters, bool gradient) const
```

There are two variants that write into a reference to a trace object instead of returning a new trace object:

```
template <typename RNG, typename Parameters>
void simulate(Trace& trace, RNG& rng, Parameters&, bool gradient) const
```

```
template <typename RNG, typename Parameters>
double generate(Trace& trace, RNG& rng, Parameters& parameters, bool gradient) const
```

### Trace

A possible sample from the probability distribution of a BoundGenerativeFunction.

Traces typically do not have public constructors, and are instead obtained from the `simulate` and `generate` functions of a BoundGenerativeFunction.

Member functions:

```
template <typename RNG>
std::tuple<double, const BackwardChoiceBuffer&, const ValueChange&> update(RNG&, const InputChange&, const ForwardChoiceBuffer&, bool save, bool gradient)
```

Note that the `BackwardChoiceBuffer` and `ValueChange` references will become undefined after the next call to `update` or `revert` or `fork`.

```
void revert()
```

```
void fork()
```

```
const ChoiceBuffer& choices(const Selection&) const
```

```
double score() const
```

```
Output get_return_value() const
```

```
InputGradient parameter_gradient(const OutputGradient&, double scaler, GradientAccumulator&)
```

```
std::pair<InputGradient choice_gradient(const Selection&, const OutputGradient&)
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

- gentl::modeling::sequence

#### Inference and Learning

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
