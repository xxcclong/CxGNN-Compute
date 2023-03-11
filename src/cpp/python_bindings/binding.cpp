#include <pybind11/pybind11.h>

#include "aggr.h"
#include "dgsparse.h"
#include "edge_softmax.h"
#include "rel_spmm.h"
#include "schedule.h"
#include "sddmm_host.h"
#include "spmm_host.h"
#include "stitch_kernel.h"
namespace py = pybind11;
using namespace pybind11::literals;

void init_compute(py::module &m) {
  m.def("sage_mean_forward_edge_value", &sage_mean_forward_edge_value,
        "Sage mean forward edge value");
  m.def("sage_sum_forward_edge_value", &sage_sum_forward_edge_value,
        "Sage sum forward edge value");
  m.def("sage_mean_forward", &sage_mean_forward, "Sage mean forward");
  m.def("sage_sum_forward", &sage_sum_forward, "Sage sum forward");
  m.def("gather", &gather, "gather");
  m.def("edge_attention", &edge_softmax_forward, "ptr"_a, "idx"_a, "att_dst"_a,
        "att_src"_a, "num_edge"_a, "relu_l"_a, "Edge attention");
  m.def("edge_value_degree", &edge_value_degree, "ptr"_a, "num_dst"_a,
        "num_edge"_a, "Edge value degree");
  m.def("target_aggr", &target_sage_sum_forward, "target aggr");
  m.def("rel_schedule", &rel_schedule, "rel schedule");
  m.def("selective_aggr", &selective_aggr_fwd, "selective aggr");
  m.def("selective_aggr_bwd", &selective_aggr_bwd, "selective aggr");
  m.def("gen_edge_type_mag240m", &gen_edge_type_mag240m, "");
  m.def("spmv", &spmv, "");
}

void init_spmm(py::module &m) {
  m.def("run_spmm_configurable", &run_spmm_configurable, "run spmm");
  m.def("run_spmm_configurable_int32", &run_spmm_configurable_int32,
        "run spmm");
}

void init_rel_spmm(py::module &m) {
  m.def("aggr_rel", &aggr_rel, "");
  m.def("aggr_rel_direct", &aggr_rel_direct, "");
  m.def("aggr_rgcn_direct_func", &aggr_rgcn_direct_func, "");
  m.def("typed_linear", &run_typed_linear, "");
  m.def("typed_linear_s2e", &run_typed_linear_s2e, "");
  m.def("typed_linear_s2d", &run_typed_linear_s2d, "");
  m.def("pad_rel", &pad_rel_gpu, "");
  m.def("pad_rel_idx", &pad_rel_idx_gpu, "");
}

void init_spmm_multihead(py::module &m) {
  py::enum_<SPMM_MULTIHEAD_SCHEDULE>(m, "SPMM_MULTIHEAD_SCHEDULE")
      .value("Naive", SPMM_MULTIHEAD_SCHEDULE::Naive)
      .value("PreReduce", SPMM_MULTIHEAD_SCHEDULE::PreReduce)
      .value("Optimal", SPMM_MULTIHEAD_SCHEDULE::Optimal)
      .export_values();
  m.def("spmm_multihead", &spmm_multihead, py::arg("ptr"), py::arg("idx"),
        py::arg("val"), py::arg("vin"), py::arg("num_node"),
        py::arg("schedule") = SPMM_MULTIHEAD_SCHEDULE::Optimal,
        py::arg("block_size") = 256);
  m.def("spmm_multihead_bwd", &spmm_multihead_bwd, py::arg("ptr"),
        py::arg("idx"), py::arg("val"), py::arg("grad_output"),
        py::arg("grad_x"), py::arg("num_node"),
        py::arg("schedule") = SPMM_MULTIHEAD_SCHEDULE::Optimal,
        py::arg("block_size") = 256);
  m.def("run_spmm_multihead_configurable", &run_spmm_multihead_configurable,
        "");
}

void init_sddmm(py::module &m) { m.def("run_sddmm", &run_sddmm, ""); }

void assertTensor(torch::Tensor &T, torch::ScalarType type) {
  assert(T.is_contiguous());
  assert(T.device().type() == torch::kCUDA);
  assert(T.dtype() == type);
}

torch::Tensor GSpMM(torch::Tensor A_rowptr, torch::Tensor A_colind,
                    torch::Tensor A_csrVal, torch::Tensor B, REDUCEOP re_op,
                    COMPUTEOP comp_op) {
  assertTensor(A_rowptr, torch::kInt32);
  assertTensor(A_colind, torch::kInt32);
  assertTensor(A_csrVal, torch::kFloat32);
  assertTensor(B, torch::kFloat32);
  return GSpMM_cuda(A_rowptr, A_colind, A_csrVal, B, re_op, comp_op);
}

torch::Tensor GSpMM_nodata(torch::Tensor A_rowptr, torch::Tensor A_colind,
                           torch::Tensor B, REDUCEOP op) {
  assertTensor(A_rowptr, torch::kInt32);
  assertTensor(A_colind, torch::kInt32);
  assertTensor(B, torch::kFloat32);
  return GSpMM_no_value_cuda(A_rowptr, A_colind, B, op);
}

void init_dgsparse(py::module &m) {
  m.def("GSpMM_u_e", &GSpMM, "CSR SPMM");
  m.def("GSpMM_u", &GSpMM_nodata, "CSR SPMM NO EDGE VALUE");
  py::enum_<REDUCEOP>(m, "REDUCEOP")
      .value("SUM", REDUCEOP::SUM)
      .value("MAX", REDUCEOP::MAX)
      .value("MIN", REDUCEOP::MIN)
      .value("MEAN", REDUCEOP::MEAN)
      .export_values();
  py::enum_<COMPUTEOP>(m, "COMPUTEOP")
      .value("ADD", COMPUTEOP::ADD)
      .value("MUL", COMPUTEOP::MUL)
      .value("DIV", COMPUTEOP::DIV)
      .value("SUB", COMPUTEOP::SUB)
      .export_values();
}

PYBIND11_MODULE(cxgnncomp_backend, m) {
  m.doc() = "A Supa Fast Graph GNN compute library";
  init_spmm(m);
  init_rel_spmm(m);
  init_spmm_multihead(m);
  init_compute(m);
  init_dgsparse(m);
  init_sddmm(m);
}
