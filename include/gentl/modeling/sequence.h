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

#ifndef GENTL_SEQUENCE_H
#define GENTL_SEQUENCE_H

#include <variant>
#include <utility>
#include <unordered_map>

namespace gentl::modeling::sequence {

    // model change types
    namespace modelchange {
        class Extend {};
        class None {};
        constexpr Extend extend{};
        constexpr None none{};
    }

    // value change types
    namespace valuechange {
        class Unknown {
            explicit operator bool() const { return true; }
        };
        class None {
            explicit operator bool() const { return false; }
        };
        constexpr Unknown unknown{};
        constexpr None none{};
    }

    // constraint types
    namespace constraints {

        template <typename T>
        struct NewStepOnly {
            NewStepOnly(T constraints_) : constraints(constraints_) {}
            T constraints;
        };

        template <typename T>
        class General {
            size_t min_t_;
            std::unordered_map<size_t,T> constraints_;
        public:
            General() : min_t_(-1) {}
            General& add(size_t t, T&& constraint) {
                auto it = constraints_.find(t);
                if (it == constraints_.end()) {
                    constraints_.insert({t, std::forward<T>(constraint)});
                } else {
                    throw std::logic_error("cannot overwrite constraints");
                }
                min_t_ = std::min(min_t_, t);
            }
            size_t min_t() const {
                return min_t_;
            }
            typename std::unordered_map<size_t,T>::const_iterator find(const size_t& key) const {
                return constraints_.find();
            }
            typename std::unordered_map<size_t,T>::const_iterator cend() const {
                return constraints_.cend();
            }
        };

        struct Empty {};
        constexpr Empty empty;
    }

    // TimeSlice
    template <typename SubtraceType>
    class TimeSlice {
        using prev_ptr_t = std::shared_ptr<TimeSlice<SubtraceType>>;
        prev_ptr_t prev_;
        size_t t_;
        std::unique_ptr<SubtraceType> subtrace_;
    public:
        TimeSlice(std::unique_ptr<SubtraceType>&& subtrace) :
            t_(0), subtrace_(std::forward<std::unique_ptr<SubtraceType>>(subtrace)) {}
        TimeSlice(std::unique_ptr<SubtraceType>&& subtrace, const prev_ptr_t& prev) :
            t_(prev->t_ + 1), subtrace_(std::forward<std::unique_ptr<SubtraceType>>(subtrace)), prev_(prev) {}
        SubtraceType& subtrace() const { return *subtrace_; }
        prev_ptr_t& prev() const { return *prev_; }
        size_t t() const { return t_; }
    };

    template <typename SubmodelType, typename SubtraceType, typename ValueType, typename ParametersType>
    class Trace;

    template <typename SubmodelType, typename SubtraceType, typename ValueType, typename ParametersType>
    class Model {
        const SubmodelType& submodel_;
        ValueType init_value_;
    public:
        Model(const SubmodelType& submodel, const ValueType& init_value) : submodel_(submodel), init_value_(init_value) {}
        typedef Trace<SubmodelType, SubtraceType, ValueType, ParametersType> trace_type;
        template <typename RNGType>
        std::pair<std::unique_ptr<trace_type>,double> generate(const constraints::Empty& obs, RNGType& rng,
                                                               ParametersType& parameters, bool gradient) const;
    };

    // TODO implement save_previous
    //        size_t private_t_alternate_; // NOTE: this can be different!
    //        std::shared_ptr<TimeSlice<SubtraceType> last_alternate_;
    // TODO add configuration for how many recent states to store pointers to
    // to avoid having to reconstruct history each time we update
    // TODO: add optimization that uses private_t_ to avoid copying non-shared subtraces

    template <typename SubmodelType, typename SubtraceType, typename ValueType, typename ParametersType>
    class Trace {
    private:
        size_t t_; // the number of time steps (0 means there are no subtraces)
//        size_t last_private_t_; // the time step (starting from 0) of the last non-shared time slice
        std::shared_ptr<TimeSlice<SubtraceType>> last_; // initially a nullptr
        const SubmodelType& submodel_;
        ParametersType& parameters_;
        ValueType init_value_;
        // void pointer because we don't know the type (see the ConstraintsType update implementation)
        std::shared_ptr<void> backward_void_ptr_;
        Trace(const Trace& other) = default;
    public:
        Trace(const SubmodelType& submodel, ValueType init_value, ParametersType& parameters) :
                t_(0),
//                last_private_t_(0),
                last_{nullptr},
                submodel_(submodel), parameters_(parameters), init_value_(init_value),
                backward_void_ptr_(nullptr) {}

        // extend (does not mutate any state)
        template<typename RNGType, typename ConstraintsType>
        std::tuple<double, const constraints::Empty&, const valuechange::Unknown&> update(
                RNGType& rng, const modelchange::Extend&, const constraints::NewStepOnly<ConstraintsType>& constraints,
                bool save_previous, bool gradient) {
            if (gradient)
                throw std::logic_error("gradient not implemented");
            if (save_previous)
                throw std::logic_error("save_previous not yet implemented");
            ValueType args = (t_ == 0) ? init_value_ : last_->subtrace().return_value();
            auto [subtrace, log_weight] = submodel_(args).generate(constraints.constraints, rng, parameters_, false);
            if (t_ == 0) {
                last_ = std::make_shared<TimeSlice<SubtraceType>>(std::move(subtrace));
            } else {
                last_ = std::make_shared<TimeSlice<SubtraceType>>(std::move(subtrace), last_);
            }
            assert(last_->t() == t_);
            t_++;
            return {log_weight, constraints::empty, valuechange::unknown};
        }

        // mutates history (copy-on-write)
        // TODO the trace owns the backwards object, but there are several types of backwards objects we could get
        template<typename RNGType, typename ConstraintsType>
        std::tuple<double, const constraints::General<ConstraintsType>&, const valuechange::Unknown&> update(
                RNGType &rng, const modelchange::None&, const constraints::General<ConstraintsType>& constraints,
                bool save_previous, bool gradient) {
            if (gradient)
                throw std::logic_error("gradient not implemented");
            if (save_previous)
                throw std::logic_error("save_previous not yet implemented");

            using slice_t = TimeSlice<SubtraceType>;

            // walk through the time slices backward to obtain a vector ('history') of pointers
            // to all time slices between min_t (inclusive) and t_ (inclusive)
            size_t min_t = constraints.min_t();
            size_t history_length = t_ - min_t + 1;
            std::vector<const slice_t*> history(history_length);
            {
                const slice_t *slice_ptr = last_.get();
                for (auto it = history.rbegin(); it != history.rend(); it++) {
                    *it = slice_ptr;
                    slice_ptr = slice_ptr->prev().get();
                }
            }
            assert(history.front()->t() == min_t);
            assert(history.back()->t() == t_);

            // walk forward through the history, forking each trace and creating new time slices
            std::shared_ptr<constraints::General<ConstraintsType>> backward;
            std::shared_ptr<slice_t> prev {history.begin()->prev()};
            std::variant<valuechange::Unknown,valuechange::None> input_change {valuechange::none};
            double log_weight = 0.0;
            for (const auto& slice_ptr : history) {
                size_t t = slice_ptr->t();
                auto forked_subtrace = slice_ptr->subtrace().fork();
                auto constraints_it = constraints.find(t);
                std::variant<constraints::Empty,ConstraintsType> sub_backward;
                double sub_log_weight;
                if (constraints_it == constraints.cend()) {
                    // did not find constraints, pass in empty constraints
                    std::tie(sub_log_weight, sub_backward, input_change) = forked_subtrace->update(
                            rng, input_change, constraints::empty, false, false);
                    if (!std::holds_alternative<constraints::empty>(sub_backward)) {
                        // this requirement could be relaxed
                        throw std::logic_error("backward constraint must have same type as forward constraints");
                    }
                } else {
                    // found constraints
                    std::tie(sub_log_weight, sub_backward, input_change) = forked_subtrace->update(
                            rng, input_change, *constraints_it, false, false);
                    if (!std::holds_alternative<ConstraintsType>(sub_backward)) {
                        // this requirement could be relaxed
                        throw std::logic_error("backward constraint must have same type as forward constraints");
                    }
                    backward->add(t, sub_backward); // copy
                }
                log_weight += sub_log_weight;
                prev = std::make_shared<TimeSlice<SubtraceType>>(std::move(forked_subtrace), std::move(prev));
            }

            // store shared pointer to last step (increments reference count)
            last_ = prev;

            backward_void_ptr_ = std::move(backward);
            const auto& backward_ref = *static_cast<constraints::General<ConstraintsType>*>(backward_void_ptr_);
            return {log_weight, backward_ref, valuechange::unknown};
        }

        std::unique_ptr<Trace> fork() {
            return std::unique_ptr<Trace>(new Trace(*this));
        }
    };

    template <typename SubmodelType, typename SubtraceType, typename ValueType, typename ParametersType>
    template <typename RNGType>
    std::pair<std::unique_ptr<typename Model<SubmodelType,SubtraceType,ValueType,ParametersType>::trace_type>,double> Model<SubmodelType,SubtraceType,ValueType,ParametersType>::generate(
            const constraints::Empty& obs, RNGType& rng, ParametersType& parameters, bool gradient) const {
        // create an initial trace with zero time steps
        auto trace = std::make_unique<trace_type>(submodel_, init_value_, parameters);
        return {std::move(trace), 0.0};
    }
}
#endif //GENTL_SEQUENCE_H
