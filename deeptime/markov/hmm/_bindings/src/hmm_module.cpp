//
// Created by mho on 2/3/20.
//

#include "common.h"
#include "utils.h"
#include "OutputModelUtils.h"
#include "docs.h"

using namespace pybind11::literals;

PYBIND11_MODULE(_hmm_bindings, m) {
    auto outputModels = m.def_submodule("output_models");
    outputModels.def("handle_outliers", &hmm::output_models::handleOutliers<float>);
    outputModels.def("handle_outliers", &hmm::output_models::handleOutliers<double>);
    {
        auto discreteModule = outputModels.def_submodule("discrete");
        discreteModule.def("generate_observation_trajectory",
                          &hmm::output_models::discrete::generateObservationTrajectory<float, std::int16_t>);
        discreteModule.def("generate_observation_trajectory",
                           &hmm::output_models::discrete::generateObservationTrajectory<float, std::int32_t>);
        discreteModule.def("generate_observation_trajectory",
                           &hmm::output_models::discrete::generateObservationTrajectory<float, std::int64_t>);
        discreteModule.def("generate_observation_trajectory",
                           &hmm::output_models::discrete::generateObservationTrajectory<double, std::int32_t>);
        discreteModule.def("generate_observation_trajectory",
                           &hmm::output_models::discrete::generateObservationTrajectory<double, std::int64_t>);
        discreteModule.def("to_output_probability_trajectory",
                           &hmm::output_models::discrete::toOutputProbabilityTrajectory<float, std::int32_t>);
        discreteModule.def("to_output_probability_trajectory",
                           &hmm::output_models::discrete::toOutputProbabilityTrajectory<float, std::int64_t>);
        discreteModule.def("to_output_probability_trajectory",
                           &hmm::output_models::discrete::toOutputProbabilityTrajectory<double, std::int32_t>);
        discreteModule.def("to_output_probability_trajectory",
                           &hmm::output_models::discrete::toOutputProbabilityTrajectory<double, std::int64_t>);
        discreteModule.def("sample", &hmm::output_models::discrete::sample<float, std::int16_t>);
        discreteModule.def("sample", &hmm::output_models::discrete::sample<float, std::int32_t>);
        discreteModule.def("sample", &hmm::output_models::discrete::sample<float, std::int64_t>);
        discreteModule.def("sample", &hmm::output_models::discrete::sample<double, std::int16_t>);
        discreteModule.def("sample", &hmm::output_models::discrete::sample<double, std::int32_t>);
        discreteModule.def("sample", &hmm::output_models::discrete::sample<double, std::int64_t>);
        discreteModule.def("update_p_out", &hmm::output_models::discrete::updatePOut<float, std::int16_t>);
        discreteModule.def("update_p_out", &hmm::output_models::discrete::updatePOut<float, std::int32_t>);
        discreteModule.def("update_p_out", &hmm::output_models::discrete::updatePOut<float, std::int64_t>);
        discreteModule.def("update_p_out", &hmm::output_models::discrete::updatePOut<double, std::int16_t>);
        discreteModule.def("update_p_out", &hmm::output_models::discrete::updatePOut<double, std::int32_t>);
        discreteModule.def("update_p_out", &hmm::output_models::discrete::updatePOut<double, std::int64_t>);
    }
    {
        auto gaussian = outputModels.def_submodule("gaussian");
        gaussian.def("p_o", &hmm::output_models::gaussian::pO<double>, "o"_a, "mus"_a, "sigmas"_a,
                     "out"_a = py::none());
        gaussian.def("p_o", &hmm::output_models::gaussian::pO<float>, "o"_a, "mus"_a, "sigmas"_a, "out"_a = py::none());
        gaussian.def("to_output_probability_trajectory",
                     &hmm::output_models::gaussian::toOutputProbabilityTrajectory<double>);
        gaussian.def("to_output_probability_trajectory",
                     &hmm::output_models::gaussian::toOutputProbabilityTrajectory<float>);
        gaussian.def("generate_observation_trajectory",
                     &hmm::output_models::gaussian::generateObservationTrajectory<float>);
        gaussian.def("generate_observation_trajectory",
                     &hmm::output_models::gaussian::generateObservationTrajectory<double>);
        gaussian.def("fit32", &hmm::output_models::gaussian::fit<float>);
        gaussian.def("fit64", &hmm::output_models::gaussian::fit<double>);
    }
    {
        auto util = m.def_submodule("util");
        util.def("viterbi", &viterbiPath<float>, "transition_matrix"_a, "state_probability_trajectory"_a, "initial_distribution"_a, docs::VITERBI);
        util.def("viterbi", &viterbiPath<double>, "transition_matrix"_a, "state_probability_trajectory"_a, "initial_distribution"_a, docs::VITERBI);
        util.def("forward", &forward<float>, "transition_matrix"_a, "state_probability_trajectory"_a, "initial_distribution"_a, "alpha_out"_a, "T"_a = py::none(), docs::FORWARD);
        util.def("forward", &forward<double>, "transition_matrix"_a, "state_probability_trajectory"_a, "initial_distribution"_a, "alpha_out"_a, "T"_a = py::none(), docs::FORWARD);
        util.def("backward", &backward<float>, "transition_matrix"_a, "state_probability_trajectory"_a, "beta_out"_a, "T"_a = py::none(), docs::BACKWARD);
        util.def("backward", &backward<double>, "transition_matrix"_a, "state_probability_trajectory"_a, "beta_out"_a, "T"_a = py::none(), docs::BACKWARD);
        util.def("state_probabilities", &stateProbabilities<float>, "alpha"_a, "beta"_a, "gamma_out"_a, "T"_a = py::none(), docs::STATE_PROBS);
        util.def("state_probabilities", &stateProbabilities<double>, "alpha"_a, "beta"_a, "gamma_out"_a, "T"_a = py::none(), docs::STATE_PROBS);
        util.def("transition_counts", &transitionCounts<float>, "alpha"_a, "beta"_a, "transition_matrix"_a, "state_probability_trajectory"_a, "counts_out"_a, "T"_a = py::none(), docs::TRANSITION_COUNTS);
        util.def("transition_counts", &transitionCounts<double>, "alpha"_a, "beta"_a, "transition_matrix"_a, "state_probability_trajectory"_a, "counts_out"_a, "T"_a = py::none(), docs::TRANSITION_COUNTS);
        util.def("sample_path", &samplePath<float>, "alpha"_a, "transition_matrix"_a, "T"_a , "seed"_a = -1, docs::SAMPLE_PATH);
        util.def("sample_path", &samplePath<double>, "alpha"_a, "transition_matrix"_a, "T"_a, "seed"_a = -1, docs::SAMPLE_PATH);
        util.def("count_matrix", &countMatrix<std::int32_t>, "dtrajs"_a, "lag"_a, "n_states"_a);
        util.def("forward_backward", &forwardBackward<float>, "transition_matrix"_a, "pObs"_a, "pi"_a, "alpha"_a, "beta"_a, "gamma"_a, "counts"_a, "T"_a);
        util.def("forward_backward", &forwardBackward<double>, "transition_matrix"_a, "pObs"_a, "pi"_a, "alpha"_a, "beta"_a, "gamma"_a, "counts"_a, "T"_a);
    }
}
