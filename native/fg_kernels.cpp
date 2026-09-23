// FactorGPT 原生热核（纯 C 接口，**不依赖 Python 头文件**）
//
// 为什么先是"纯 C 接口"而不是直接写 pybind11：
// 1. 可以用任何编译器编成动态库，用 ctypes 直接加载并做**数值对拍**——
//    在没有 MSVC 的机器上（比如开发机）也能验证 C++ 侧的正确性，不必等到 CI；
// 2. 同一份代码给 pybind11 绑定用（``fg_bindings.cpp``），两边不会漂移。
//
// 语义基准是 Python 侧的 pandas 实现（``ops.cs_rank`` / ``ops.cs_zscore``），
// 本文件必须与它逐位对齐，包括：NaN 的处理、并列取平均秩、pct 的分母是
// 非空计数、ddof=1、以及标准差恰为 0 时输出 NaN。任何"差不多"都会让因子值
// 悄悄改变，而这恰恰是回测里最难查的那类 bug。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#if defined(_WIN32)
#  define FG_API __declspec(dllexport)
#else
#  define FG_API __attribute__((visibility("default")))
#endif

extern "C" {

// 逐行横截面百分位排名（并列取平均秩；NaN 不参与且输出保持 NaN）
FG_API void fg_cs_rank_pct(const double* x, double* out,
                           int64_t rows, int64_t cols, int64_t min_count) {
  if (!x || !out || rows <= 0 || cols <= 0) return;
  std::vector<std::pair<double, int64_t>> buf;
  buf.reserve(static_cast<size_t>(cols));
  for (int64_t r = 0; r < rows; ++r) {
    const double* row = x + r * cols;
    double* orow = out + r * cols;
    buf.clear();
    for (int64_t c = 0; c < cols; ++c) {
      if (!std::isnan(row[c])) buf.emplace_back(row[c], c);
    }
    const int64_t n = static_cast<int64_t>(buf.size());
    if (n < min_count) {
      for (int64_t c = 0; c < cols; ++c) orow[c] = NAN;
      continue;
    }
    // 稳定排序足够：并列元素的相对顺序不影响平均秩
    std::stable_sort(buf.begin(), buf.end(),
                     [](const std::pair<double, int64_t>& a,
                        const std::pair<double, int64_t>& b) {
                       return a.first < b.first;
                     });
    const double denom = static_cast<double>(n);
    int64_t i = 0;
    while (i < n) {
      int64_t j = i;
      while (j + 1 < n && buf[j + 1].first == buf[i].first) ++j;
      // 1-based 平均秩：(i+1 + j+1) / 2，再除以非空计数得到 pct
      const double avg = 0.5 * (static_cast<double>(i) + static_cast<double>(j)) + 1.0;
      for (int64_t k = i; k <= j; ++k) orow[buf[k].second] = avg / denom;
      i = j + 1;
    }
    for (int64_t c = 0; c < cols; ++c) {
      if (std::isnan(row[c])) orow[c] = NAN;
    }
  }
}

// 逐行横截面标准化（去均值 / 样本标准差 ddof=1；sd==0 或计数不足 → NaN）
FG_API void fg_cs_zscore(const double* x, double* out,
                         int64_t rows, int64_t cols, int64_t min_count) {
  if (!x || !out || rows <= 0 || cols <= 0) return;
  const int64_t need = std::max<int64_t>(min_count, 2);
  for (int64_t r = 0; r < rows; ++r) {
    const double* row = x + r * cols;
    double* orow = out + r * cols;
    int64_t n = 0;
    double sum = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
      const double v = row[c];
      if (!std::isnan(v)) { sum += v; ++n; }
    }
    if (n < need) {
      for (int64_t c = 0; c < cols; ++c) orow[c] = NAN;
      continue;
    }
    const double mu = sum / static_cast<double>(n);
    double sq = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
      const double v = row[c];
      if (!std::isnan(v)) { const double d = v - mu; sq += d * d; }
    }
    const double sd = std::sqrt(sq / static_cast<double>(n - 1));
    if (!(sd > 0.0) || !std::isfinite(sd)) {   // sd==0 → NaN（与 pandas 一致）
      for (int64_t c = 0; c < cols; ++c) orow[c] = NAN;
      continue;
    }
    for (int64_t c = 0; c < cols; ++c) {
      const double v = row[c];
      orow[c] = std::isnan(v) ? NAN : (v - mu) / sd;
    }
  }
}

// 逐行（横截面）Pearson 相关
//
// 为什么要融合：``ops.cs_corr`` 的 numpy 版本是正确的，但一次调用要分配近十个
// 临时矩阵（掩码、两侧去均值、偏差、平方和……），而它在一次搜索里被调用上千次。
// 融合版每个交易日只扫两遍内存、零临时分配，省掉的是内存带宽与 GC 压力。
//
// 语义对齐 ``ops.cs_corr``：用 **isfinite**（inf 视为无效，与 -ffast-math 无关）、
// 有效样本 ``n < max(min_stocks, 3)`` 的行给 NaN、结果非有限也置 NaN。
FG_API void fg_cs_corr(const double* a, const double* b, double* out,
                       int64_t rows, int64_t cols, int64_t min_stocks) {
  if (!a || !b || !out || rows <= 0 || cols <= 0) return;
  const int64_t need = std::max<int64_t>(min_stocks, 3);
  for (int64_t r = 0; r < rows; ++r) {
    const double* ra = a + r * cols;
    const double* rb = b + r * cols;
    int64_t n = 0;
    double sa = 0.0, sb = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
      const double va = ra[c], vb = rb[c];
      if (std::isfinite(va) && std::isfinite(vb)) { sa += va; sb += vb; ++n; }
    }
    if (n < need) { out[r] = NAN; continue; }
    const double ma = sa / static_cast<double>(n);
    const double mb = sb / static_cast<double>(n);
    double ab = 0.0, aa = 0.0, bb = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
      const double va = ra[c], vb = rb[c];
      if (std::isfinite(va) && std::isfinite(vb)) {
        const double da = va - ma, db = vb - mb;
        ab += da * db; aa += da * da; bb += db * db;
      }
    }
    const double denom = std::sqrt(aa * bb);
    const double rho = denom > 0.0 ? (ab / denom) : NAN;
    out[r] = std::isfinite(rho) ? rho : NAN;
  }
}

}  // extern "C"
