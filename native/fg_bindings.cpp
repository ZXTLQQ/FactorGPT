// pybind11 绑定：**同一份** ``fg_kernels.cpp``，只是换一层 Python 入口。
//
// 为什么不把所有入口都写成 pybind11：那样就只能在"能把扩展链到 CPython"的
// 编译器上验证（Windows 上即 MSVC），而大多数开发机没有。现在热核是纯 C 接口，
// 既能被这里直接包成扩展，也能被 ctypes 加载（``scripts/build_native.py`` 用
// zig 就能编出来），数值验证与打包分发互不阻塞。

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>

extern "C" {
void fg_cs_rank_pct(const double* x, double* out, int64_t rows, int64_t cols,
                    int64_t min_count);
void fg_cs_zscore(const double* x, double* out, int64_t rows, int64_t cols,
                  int64_t min_count);
void fg_cs_corr(const double* a, const double* b, double* out, int64_t rows,
                int64_t cols, int64_t min_stocks);
}

namespace py = pybind11;

namespace {

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

template <typename Kernel>
py::array_t<double> run2(Kernel kernel, Arr x, int64_t min_count) {
  if (x.ndim() != 2) throw std::invalid_argument("需要二维数组 (dates, symbols)");
  const int64_t rows = static_cast<int64_t>(x.shape(0));
  const int64_t cols = static_cast<int64_t>(x.shape(1));
  auto out = py::array_t<double>({rows, cols});
  kernel(x.data(), out.mutable_data(), rows, cols, min_count);
  return out;
}

}  // namespace

PYBIND11_MODULE(fg_native, m) {
  m.doc() = "FactorGPT 原生热核（横截面排名 / 标准化 / 相关系数）";

  m.def("cs_rank_pct",
        [](Arr x, int64_t min_count) {
          return run2(fg_cs_rank_pct, x, min_count);
        },
        py::arg("x"), py::arg("min_count") = 1,
        "逐行百分位排名（并列取平均秩，NaN 保持 NaN）");

  m.def("cs_zscore",
        [](Arr x, int64_t min_count) {
          return run2(fg_cs_zscore, x, min_count);
        },
        py::arg("x"), py::arg("min_count") = 2,
        "逐行去均值再除以样本标准差（ddof=1）");

  m.def("cs_corr",
        [](Arr a, Arr b, int64_t min_stocks) {
          if (a.ndim() != 2 || b.ndim() != 2)
            throw std::invalid_argument("需要二维数组 (dates, symbols)");
          if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1))
            throw std::invalid_argument("两侧形状必须一致");
          const int64_t rows = static_cast<int64_t>(a.shape(0));
          const int64_t cols = static_cast<int64_t>(a.shape(1));
          auto out = py::array_t<double>({rows});
          fg_cs_corr(a.data(), b.data(), out.mutable_data(), rows, cols,
                     min_stocks);
          return out;
        },
        py::arg("a"), py::arg("b"), py::arg("min_stocks") = 10,
        "逐行 Pearson 相关；有效样本不足或退化行为 NaN");

  m.attr("__version__") = "0.1.0";
}
