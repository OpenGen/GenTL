/* Copyright 2021-2022 Massachusetts Institute of Technology

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#ifndef GENTL_MCMC_H
#define GENTL_MCMC_H

#include <cmath>
#include <random>
#include <gentl/types.h>
#include <gentl/concepts.h>

using gentl::SimulateOptions;
using gentl::GenerateOptions;
using gentl::UpdateOptions;

namespace gentl::mcmc {



double mh_accept_prob(double model_log_weight, double proposal_forward_score, double proposal_backward_score) {
    return std::min(1.0, std::exp(model_log_weight + proposal_backward_score - proposal_forward_score));
}

// ***********************
// *** Involutive MCMC ***
// ***********************

template<typename ProposalTrace, typename ModelTrace, typename Proposal, typename ProposalParams,
         typename Involution, typename RNG>
bool involutive_mcmc(RNG& rng, ModelTrace& trace, const Proposal& proposal, const ProposalParams& params,
                     const Involution& involution, bool round_trip_check = false) {
    // TODO add version that does simulation in-place (and accepts a ProposalTrace argument)
    auto forward_proposal_trace = proposal(trace).simulate(rng, params, SimulateOptions());
    auto forward_proposal_choices = forward_proposal_trace.choices();
    double forward_proposal_score = forward_proposal_trace.score();
    std::unique_ptr<ModelTrace> original_trace = nullptr;
    if (round_trip_check) {
        original_trace = trace.fork();
    }
    // NOTE: involution must save, so it can be reverted
    auto [backward_proposal_choices, model_log_weight] = involution(trace, forward_proposal_choices);
    auto [retval, backward_proposal_score] = proposal(trace).assess(rng, params);
    double prob_accept = mh_accept_prob(model_log_weight, forward_proposal_score, backward_proposal_score);
    bool accept = std::bernoulli_distribution{prob_accept}(rng);
    if (round_trip_check) {
        auto [forward_proposal_choices_rt, model_log_weight_rt] = involution(trace, backward_proposal_choices);
        // TODO check that forward_proposal_choices_rt matches forward_proposal_choices (NOTE this is not possible if the value type in the choice map is std::any; should we have users use std::variant instead?)
        // TODO check that trace.choices() matches original_trace->choices()
        // TODO check that model_log_weight_rt = -model_log_weight
        if (accept)
            trace.revert();
    } else {
        if (!accept)
            trace.revert();
    }
    return accept;

    // TODO what is the implementation oft he transform? Just a lambda function that takes
    // (model_trace, constraints) -> pair<constraints, double>
    // can we provide basic implemntation for the case when Jacobian is 1? i.e. when we just arte changing the address of choices or changing discrete choices?
    // we can provide an implementation when the jacobian is 1 that works for ChoiceTrie:
    // 1. a user function that takes the forward constraints, and the choices() from the trace, and the return value (?)
    // and returns a new choice trie (that represents constraints to be passed to update), and a backward constraints
    // then, the backward constraints from update are merged with their backwrad constraints
    // there are separate namespaces ("model" and "propsoal")
}


// ***********************************************************************
// *** Metropolis-Hastings using a generative function as the proposal ***
// ***********************************************************************


#ifdef __cpp_concepts
template<typename ModelTrace, typename ProposalTrace>
concept MHProposalCompatible = requires(ModelTrace trace, ProposalTrace proposal_trace, std::mt19937 rng) {
    { trace.update(rng, gentl::change::no_change, proposal_trace.choices(), UpdateOptions()) };
};
#endif

template<typename ModelTrace, typename ProposalTrace, typename Proposal,
         typename ProposalParams, typename RNGType>
#ifdef __cpp_concepts
requires
    InPlaceSimulatableGenerativeFunction<Proposal,ModelTrace,ProposalParams,ProposalTrace> &&
    MHProposalCompatible<ModelTrace, ProposalTrace> &&
    Revertible<ModelTrace>
#endif
bool mh(RNGType &rng, ModelTrace &model_trace,
        const Proposal &proposal, ProposalParams &proposal_params, ProposalTrace &proposal_trace,
        bool precompute_gradient = false) {
    proposal(model_trace).simulate(rng, proposal_params, SimulateOptions(), proposal_trace);
    double proposal_forward_score = proposal_trace.score();
    const auto &forward_constraints = proposal_trace.choices();
    double model_log_weight = model_trace.update(
            rng, gentl::change::no_change,
            forward_constraints,
            UpdateOptions().save(true).precompute_gradient(precompute_gradient));
    const auto& backward_constraints = model_trace.backward_constraints();
    auto [proposal_retval, proposal_backward_score] = proposal(model_trace).assess(rng, proposal_params, backward_constraints);
    double prob_accept = mh_accept_prob(model_log_weight, proposal_forward_score, proposal_backward_score);
    bool accept = std::bernoulli_distribution{prob_accept}(rng);
    if (!accept)
        model_trace.revert();
    return accept;
}

template<typename ProposalTrace, typename ModelTrace, typename Proposal, typename ProposalParams, typename RNGType>
#ifdef __cpp_concepts
requires
SimulatableGenerativeFunction<Proposal,ModelTrace,ProposalParams,ProposalTrace> &&
MHProposalCompatible<ModelTrace, ProposalTrace> &&
Revertible<ModelTrace>
#endif
bool mh(RNGType &rng, ModelTrace &model_trace,
        const Proposal &proposal, ProposalParams &proposal_params,
        bool precompute_gradient = false) {
    auto proposal_trace = proposal(model_trace).simulate(rng, proposal_params, SimulateOptions());
    double proposal_forward_score = proposal_trace->score();
    const auto &forward_constraints = proposal_trace->choices();
    double model_log_weight = model_trace.update(
            rng, gentl::change::no_change,
            forward_constraints,
            UpdateOptions().save(true).precompute_gradient(precompute_gradient));
    const auto& backward_constraints = model_trace.backward_constraints();
    auto [proposal_retval, proposal_backward_score] = proposal(model_trace).assess(rng, proposal_params, backward_constraints);
    double prob_accept = mh_accept_prob(model_log_weight, proposal_forward_score, proposal_backward_score);
    bool accept = std::bernoulli_distribution{prob_accept}(rng);
    if (!accept)
        model_trace.revert();
    return accept;
}

// *******************************************
// *** Metropolis-Adjusted Langevin (MALA) ***
// *******************************************

template <typename ChoiceBuffer, typename RNGType>
double mala_propose_values(ChoiceBuffer& proposal, const ChoiceBuffer& current,
                           const ChoiceBuffer& gradient, double tau, RNGType& rng) {
    double stdev = std::sqrt(2 * tau);
    static std::normal_distribution<double> standard_normal {0.0, 1.0};
    static double logSqrt2Pi = 0.5*std::log(2*M_PI);
    double log_density = (current.cend() - current.cbegin()) * (-std::log(stdev) - logSqrt2Pi);

    // first compute the mean in-place
    proposal = current + (tau * gradient);

    // then sample new values (over-writing the mean), and finish computing the log density
    for (auto proposal_it = proposal.begin(); proposal_it != proposal.end(); proposal_it++) {
        double standard_normal_increment = standard_normal(rng);
        *proposal_it += (standard_normal_increment * stdev);
        log_density += -0.5 * (standard_normal_increment * standard_normal_increment);
    }
    return log_density;
}

template <typename ChoiceBuffer>
double mala_assess(const ChoiceBuffer& proposed, const ChoiceBuffer& current,
                   const ChoiceBuffer& gradient, double tau, ChoiceBuffer& storage) {
    double stdev = std::sqrt(2 * tau);
    static double logSqrt2Pi = 0.5*std::log(2*M_PI);
    double log_density = (current.cend() - current.cbegin()) * (-std::log(stdev) - logSqrt2Pi);

    ChoiceBuffer& proposal_mean = storage; // rename it
    proposal_mean = current + (tau * gradient);

    auto proposal_mean_it = proposal_mean.cbegin();
    for (auto proposed_it = proposed.cbegin(); proposed_it != proposed.cend(); proposed_it++) {
        double standard_normal_increment = (*proposed_it - *(proposal_mean_it++)) / stdev;
        log_density += -0.5 * (standard_normal_increment * standard_normal_increment);
    }
    return log_density;
}



template<typename Trace, typename Selection, typename RNG, typename ChoiceBuffer>
#ifdef __cpp_concepts
requires TopLevelGradientBasedChoiceUpdates<Trace,Selection,ChoiceBuffer> && Revertible<Trace>
#endif
bool mala(Trace &trace, const Selection &selection, double tau,
          ChoiceBuffer &storage1, ChoiceBuffer& storage2, RNG &rng) {

    // NOTE: these buffers are only valid up until the next call to update.
    ChoiceBuffer& proposed_values = storage1;
    double forward_log_density = mala_propose_values(proposed_values, trace.choices(selection),
                                                     trace.choice_gradient(selection), tau, rng);
    double log_weight = trace.update(
            rng, gentl::change::no_change, proposed_values,
            UpdateOptions().save(true).precompute_gradient(true));

    // compute backward log density
    const auto& previous_values = trace.backward_constraints();
    double backward_log_density = mala_assess(previous_values, proposed_values,
                                              trace.choice_gradient(selection), tau,
                                              storage2);

    double prob_accept = std::min(1.0, std::exp(log_weight + backward_log_density - forward_log_density));
    bool accept = std::bernoulli_distribution{prob_accept}(rng);
    if (!accept)
        trace.revert();
    return accept;
}

// *************************************
// *** Hamiltonian Monte Carlo (HMC) ***
// *************************************


template <typename ChoiceBufferType, typename RNGType>
void sample_momenta(ChoiceBufferType& momenta, RNGType& rng) {
    static std::normal_distribution<double> standard_normal{0.0, 1.0};
    for (auto& momentum : momenta)
        momentum = standard_normal(rng);
}

template <typename ChoiceBufferType>
double assess_momenta(const ChoiceBufferType& momenta) {
    static double logSqrt2Pi = 0.5*std::log(2*M_PI);
    double sum = 0.0;
    for (const auto& momentum : momenta)
        sum += -0.5 * momentum * momentum;
    return sum - (momenta.cend() - momenta.cbegin()) * logSqrt2Pi;
}

template<typename Trace, typename Selection, typename RNG, typename ChoiceBuffer>
#ifdef __cpp_concepts
requires TopLevelGradientBasedChoiceUpdates<Trace,Selection,ChoiceBuffer> && Revertible<Trace>
#endif
bool hmc(Trace &trace, const Selection& selection,
         size_t leapfrog_steps, double eps,
         ChoiceBuffer& momenta_buffer,
         ChoiceBuffer& values_buffer,
         RNG &rng) {

    // NOTE: this read-only buffer is only valid up until the next call to update.
    const ChoiceBuffer* gradient_buffer = &trace.choice_gradient(selection);

    // this overwrites the memory in the buffer
    sample_momenta(momenta_buffer, rng);
    double prev_momenta_score = assess_momenta(momenta_buffer);

    double log_weight = 0.0;
    for (size_t step = 0; step < leapfrog_steps; step++) {

        // half step on momenta
        momenta_buffer += (eps / 2.0) * (*gradient_buffer);

        // full step on positions
        values_buffer += eps * momenta_buffer;

        // get incremental log weight and new gradient
        double log_weight_increment = trace.update(
                rng, gentl::change::no_change, values_buffer,
                UpdateOptions().save(step == 0).precompute_gradient(true));
        log_weight += log_weight_increment;
        gradient_buffer = &trace.choice_gradient(selection);

        // half step on momenta
        momenta_buffer += (eps / 2.0) * (*gradient_buffer);
    }

    double new_momenta_score = assess_momenta(momenta_buffer);

    double prob_accept = std::min(1.0, std::exp(log_weight + new_momenta_score - prev_momenta_score));
    bool accept = std::bernoulli_distribution{prob_accept}(rng);
    if (!accept)
        trace.revert();
    return accept;
}


// TODO elliptical slice sampling
// TODO involutive MCMC...

}
#endif //GENTL_MCMC_H
