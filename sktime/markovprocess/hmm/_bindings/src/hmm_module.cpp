//
// Created by mho on 2/3/20.
//

#include "common.h"
#include "OutputModelUtils.h"

using namespace pybind11::literals;

PYBIND11_MODULE(_hmm_bindings, m) {
    auto outputModels = m.def_submodule("output_models");
    outputModels.def("handle_outliers", &hmm::output_models::handleOutliers<float>);
    outputModels.def("handle_outliers", &hmm::output_models::handleOutliers<double>);
    {
        auto discreteModule = outputModels.def_submodule("discrete");
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
        discreteModule.def("sample", &hmm::output_models::discrete::sample<float, std::int32_t>);
        discreteModule.def("sample", &hmm::output_models::discrete::sample<float, std::int64_t>);
        discreteModule.def("sample", &hmm::output_models::discrete::sample<double, std::int32_t>);
        discreteModule.def("sample", &hmm::output_models::discrete::sample<double, std::int64_t>);
        discreteModule.def("update_p_out", &hmm::output_models::discrete::updatePOut<float, std::int32_t>);
        discreteModule.def("update_p_out", &hmm::output_models::discrete::updatePOut<float, std::int64_t>);
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
        gaussian.def("generate_observation_trajectory", &hmm::output_models::gaussian::generateObservationTrajectory<float>);
        gaussian.def("generate_observation_trajectory", &hmm::output_models::gaussian::generateObservationTrajectory<double>);
        gaussian.def("fit", &hmm::output_models::gaussian::fit<float>);
        gaussian.def("fit", &hmm::output_models::gaussian::fit<double>);
    }
}
