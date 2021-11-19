// author: maaike

#include "tram.h"

using namespace pybind11::literals;

PYBIND11_MODULE(_tram_bindings, m) {
    using namespace deeptime::tram;
    {
        auto tramMod = m.def_submodule("tram");

        tramMod.def("init_lagrangian_mult", &init_lagrangian_mult<double>, "count_matrices"_a, "n_therm_states"_a,
                    "n_conf_states"_a, "log_lagrangian_mult"_a);

        tramMod.def("update_lagrangian_mult", &update_lagrangian_mult<double>, "log_lagrangian_mult"_a,
                    "biased_conf_energies"_a, "count_matrices"_a,
                    "state_counts"_a, "n_therm_states"_a, "n_conf_states"_a, "scratch_M"_a,
                    "new_log_lagrangian_mult"_a);

        tramMod.def("get_log_Ref_K_i", &get_log_Ref_K_i<double>, "log_lagrangian_mult"_a, "biased_conf_energies"_a,
                    "count_matrices"_a, "state_counts"_a, "n_therm_states"_a, "n_conf_states"_a, "scratch_M"_a,
                    "log_R_K_i"_a);
        /*tramMod.def("get_log_Ref_K_i_mbar", &get_log_Ref_K_i<double, true>, "log_lagrangian_mult"_a, "biased_conf_energies"_a,
                    "count_matrices"_a, "state_counts"_a, "n_therm_states"_a, "n_conf_states"_a, "scratch_M"_a,
                    "log_R_K_i"_a);*/

        tramMod.def("update_biased_conf_energies", &update_biased_conf_energies<double>, "bias_energy_sequence"_a,
                    "state_sequence"_a, "seq_length"_a, "log_R_K_i"_a, "n_therm_states"_a, "n_conf_states"_a,
                    "scratch_T"_a, "new_biased_conf_energies"_a, "return_log_L"_a);

        tramMod.def("get_conf_energies", &get_conf_energies<double>, "bias_energy_sequence"_a, "state_sequence"_a,
                    "seq_length"_a, "log_R_K_i"_a, "n_therm_states"_a, "n_conf_states"_a, "scratch_T"_a,
                    "conf_energies"_a);

        tramMod.def("get_therm_energies", &get_therm_energies<double>, "biased_conf_energies"_a, "n_therm_states"_a,
                    "n_conf_states"_a, "scratch_M"_a, "therm_energies"_a);

        tramMod.def("normalize", &normalize<double>, "conf_energies"_a, "biased_conf_energies"_a, "therm_energies"_a,
                    "n_therm_states"_a, "n_conf_states"_a, "scratch_M"_a);

        tramMod.def("estimate_transition_matrix", &estimate_transition_matrix<double>, "log_lagrangian_mult"_a,
                    "conf_energies"_a, "count_matrix"_a, "n_conf_states"_a, "scratch_M"_a, "transition_matrix"_a);

        tramMod.def("discrete_log_likelihood_lower_bound", &discrete_log_likelihood_lower_bound<double>,
                    "log_lagrangian_mult"_a, "biased_conf_energies"_a, "count_matrices"_a, "state_counts"_a,
                    "n_therm_states"_a, "n_conf_states"_a, "scratch_M"_a, "scratch_MM"_a);

        tramMod.def("get_pointwise_unbiased_free_energies", &get_pointwise_unbiased_free_energies<double>, "k"_a,
                    "bias_energy_sequence"_a, "therm_energies"_a, "state_sequence"_a, "seq_length"_a, "log_R_K_i"_a,
                    "n_therm_states"_a, "n_conf_states"_a, "scratch_T"_a, "pointwise_unbiased_free_energies"_a);

        tramMod.def("_bar_df", &_bar_df<double>, "db_IJ"_a, "L1"_a, "db_JI"_a, "L2"_a, "scratch"_a);
    }
}
