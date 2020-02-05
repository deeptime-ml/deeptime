//
// Created by mho on 2/3/20.
//

#include "common.h"
#include "OutputModels.h"

using namespace pybind11::literals;

PYBIND11_MODULE(_hmm_bindings, m) {
    using dtype = double;
    {
        auto outputModels = m.def_submodule("output_models");
        using DOM = hmm::output_models::discrete::DiscreteOutputModel<dtype, std::int64_t>;
        py::class_<DOM> dm(outputModels, "DiscreteOutputModel");
        dm.def(py::init<np_array<dtype>, py::object, bool>(), "output_probability_matrix"_a, "prior"_a = py::none(),
               "ignore_outliers"_a = true);
        dm.def_property_readonly("n_hidden_states", &DOM::nHiddenStates);
        dm.def_property_readonly("n_observable_states", &DOM::nObservableStates);
        dm.def_property("ignore_outliers", &DOM::ignoreOutliers, &DOM::setIgnoreOutliers);
        dm.def_property_readonly("prior", &DOM::prior);
        dm.def_property_readonly("output_probabilities", &DOM::outputProbabilities);
        dm.def("output_probability_trajectory", &DOM::outputProbabilityTrajectory);
        dm.def("generate_observation_trajectory", &DOM::generateObservationTrajectory);
        dm.def("sample", &DOM::sample);
    }
    {
        {
            auto discrete = m.def_submodule("discrete");
            discrete.def("update_p_out", &hmm::output_models::discrete::updatePOut<float, std::int32_t>, "obs"_a, "weights"_a, "pout"_a);
            discrete.def("update_p_out", &hmm::output_models::discrete::updatePOut<float, std::int64_t>, "obs"_a, "weights"_a, "pout"_a);
            discrete.def("update_p_out", &hmm::output_models::discrete::updatePOut<double, std::int32_t>, "obs"_a, "weights"_a, "pout"_a);
            discrete.def("update_p_out", &hmm::output_models::discrete::updatePOut<double, std::int64_t>, "obs"_a, "weights"_a, "pout"_a);
        }

        {
            auto gaussian = m.def_submodule("gaussian");
            gaussian.def("p_o", &hmm::output_models::gaussian::pO<double>, "o"_a, "mus"_a, "sigmas"_a, "out"_a = py::none());
            gaussian.def("p_o", &hmm::output_models::gaussian::pO<float>, "o"_a, "mus"_a, "sigmas"_a, "out"_a = py::none());
            gaussian.def("p_obs", &hmm::output_models::gaussian::pObs<double>, "obs"_a, "mus"_a, "sigmas"_a, "out"_a = py::none());
            gaussian.def("p_obs", &hmm::output_models::gaussian::pObs<float>, "obs"_a, "mus"_a, "sigmas"_a, "out"_a = py::none());
        }
    }
}
