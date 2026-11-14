target_sources(
  sirius_objects
  PRIVATE date_cast.cu
          decimal_integer_cast.cu
          dense_count_join_impl.cu
          sorted_keys.cu
          low_cardinality_groupby.cu
          device_copy_batch.cpp
          device_transfer.cpp
          dynamic_filter_mask_ops.cu
          dynamic_filter_replica_transfer.cu
          sirius_dynamic_bloom_filter.cu
          sirius_dynamic_in_list_filter.cu
          sirius_dynamic_small_in_list_filter.cu
          sirius_like_multiliteral.cu
          sirius_round_floating_point.cu
          scan/gpu_decode_alp.cu
          scan/gpu_decode_bitpacking.cu
          scan/gpu_decode_rle.cu
          scan/gpu_decode_strings.cu
          scan/gpu_native_decode.cu
          scan/strings/dict_fsst.cu
          scan/strings/dictionary.cu
          scan/strings/fsst.cu
          scan/strings/uncompressed.cu
          vss/cagra_index.cu
          vss/cudf_raft_interop.cu
          vss/brute_force_search.cu
          vss/brute_force_threshold.cu
          vss/cluster_fold.cu
          vss/device_rates.cu
          vss/gemm_topk.cu
          vss/knn_merge.cu
          vss/routed_edges.cu
          vss/vector_clustering.cu
          vss/ivf_flat_index.cu)

# The bounded tile kernel sits at 128 registers for two blocks per SM, and
# separable compilation makes it spill (a standalone copy on an A5000: 83 TOPS
# whole-program, 74 with -rdc). The file neither defines nor calls device code
# across files, so it is its own object library compiled whole-program, with
# sirius_objects' includes, definitions and options; the targets that take
# sirius_objects' objects take these too (cmake/sirius-targets.cmake).
add_library(sirius_vss_bound_gemm_objects OBJECT vss/bound_gemm.cu vss/bound_gemm_cutlass.cu)
# The bounded FP16 filter's tile kernel is built on CUTLASS's 2.x GEMM (header-only).
# SIRIUS_CUTLASS_INCLUDE_DIR, as a cache or environment variable, names a local copy; without one
# the pinned release is fetched at configure time.
set(SIRIUS_CUTLASS_INCLUDE_DIR
    ""
    CACHE PATH "CUTLASS include directory (holds cutlass/cutlass.h)")
if(NOT SIRIUS_CUTLASS_INCLUDE_DIR AND DEFINED ENV{SIRIUS_CUTLASS_INCLUDE_DIR})
  set(SIRIUS_CUTLASS_INCLUDE_DIR "$ENV{SIRIUS_CUTLASS_INCLUDE_DIR}")
endif()
if(SIRIUS_CUTLASS_INCLUDE_DIR)
  if(NOT EXISTS "${SIRIUS_CUTLASS_INCLUDE_DIR}/cutlass/cutlass.h")
    message(FATAL_ERROR "SIRIUS_CUTLASS_INCLUDE_DIR=${SIRIUS_CUTLASS_INCLUDE_DIR} holds no "
                        "cutlass/cutlass.h")
  endif()
else()
  include(FetchContent)
  FetchContent_Declare(
    sirius_cutlass
    URL https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.2.1.tar.gz
    URL_HASH
      SHA256=a4513ba33ae82fd754843c6d8437bee1ac71a6ef1c74df886de2338e3917d4df
    SOURCE_SUBDIR do-not-build)
  FetchContent_MakeAvailable(sirius_cutlass)
  set(SIRIUS_CUTLASS_INCLUDE_DIR "${sirius_cutlass_SOURCE_DIR}/include")
endif()
set_source_files_properties(
  vss/bound_gemm_cutlass.cu
  PROPERTIES INCLUDE_DIRECTORIES "${SIRIUS_CUTLASS_INCLUDE_DIR}"
             COMPILE_OPTIONS "--expt-relaxed-constexpr")
set_target_properties(
  sirius_vss_bound_gemm_objects
  PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
             POSITION_INDEPENDENT_CODE ON
             CXX_VISIBILITY_PRESET hidden
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON)
target_include_directories(
  sirius_vss_bound_gemm_objects
  PRIVATE $<TARGET_PROPERTY:sirius_objects,INCLUDE_DIRECTORIES>)
target_compile_definitions(
  sirius_vss_bound_gemm_objects
  PRIVATE $<TARGET_PROPERTY:sirius_objects,COMPILE_DEFINITIONS>)
target_compile_options(
  sirius_vss_bound_gemm_objects
  PRIVATE $<TARGET_PROPERTY:sirius_objects,COMPILE_OPTIONS>)
