#include <pybind11/pybind11.h>

#include "aggr.h"
#include "edge_softmax.h"
#include "schedule.h"
namespace py = pybind11;
using namespace pybind11::literals;

void init_compute(py::module &m) {
  m.def("sage_mean_forward_edge_value", &sage_mean_forward_edge_value,
        "Sage mean forward edge value");
  m.def("sage_sum_forward_edge_value", &sage_sum_forward_edge_value,
        "Sage sum forward edge value");
  m.def("sage_mean_forward", &sage_mean_forward, "Sage mean forward");
  m.def("sage_sum_forward", &sage_sum_forward, "Sage sum forward");
  m.def("aggr_rel", &aggr_rel, "");
  m.def("aggr_rel_direct", &aggr_rel_direct, "");
  m.def("gather", &gather, "gather");
  m.def("edge_attention", &edge_softmax_forward, "ptr"_a, "idx"_a, "att_dst"_a,
        "att_src"_a, "num_edge"_a, "relu_l"_a, "Edge attention");
  m.def("edge_value_degree", &edge_value_degree, "ptr"_a, "num_dst"_a,
        "num_edge"_a, "Edge value degree");
  m.def("target_aggr", &target_sage_sum_forward, "target aggr");
  m.def("rel_schedule", &rel_schedule, "rel schedule");
  m.def("selective_aggr", &selective_aggr_fwd, "selective aggr");
  m.def("selective_aggr_bwd", &selective_aggr_bwd, "selective aggr");
}

PYBIND11_MODULE(cxgnncomp_backend, m) {
  m.doc() = "A Supa Fast Graph GNN compute library";
  init_compute(m);
}
