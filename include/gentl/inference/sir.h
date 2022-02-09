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

#ifndef GENTL_SIR_H
#define GENTL_SIR_H

#include <random>
#include <gentl/util/mathutils.h>

namespace gentl::sir {

template <typename TraceType, typename ModelType, typename ParametersType, typename ConstraintsType, typename RNGType>
std::pair<std::unique_ptr<TraceType>,double> rolling_importance_resampling(
        const ModelType& model, ParametersType& parameters,
        const ConstraintsType& constraints, RNGType& rng,
        size_t num_particles) {
    if (num_particles == 0)
        throw std::logic_error("num_particles == 0");
    auto [trace_ptr, log_total_weight] = model.generate(constraints, rng, parameters, false);
    // generate another trace that will be used to generate all future traces in-place
    auto [candidate_trace_ptr, unused] = model.generate(constraints, rng, parameters, false);
    for (size_t i = 1; i < num_particles; i++) {
        // overwrite candidate trace
        double log_weight_increment = model.generate(*candidate_trace_ptr, constraints, rng, parameters, false);
        log_total_weight = gentl::mathutils::logsumexp(log_total_weight, log_weight_increment);
        double prob_replace = std::exp(log_weight_increment - log_total_weight);
        std::bernoulli_distribution dist{prob_replace};
        if (dist(rng)) {
            // swap the pointers
            std::swap(trace_ptr, candidate_trace_ptr);
        }
    }
    double log_ml_estimate = log_total_weight - std::log(num_particles);
    return {std::move(trace_ptr), log_ml_estimate};
}

}

#endif //GENTL_SIR_H
