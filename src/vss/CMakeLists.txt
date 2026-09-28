target_sources(
  sirius_objects
  PRIVATE cuvs_index_cache.cpp
          distance_metric.cpp
          enn_top_k.cpp
          join_result_shaping.cpp
          kmeans_functions.cpp
          vector_join_binding.cpp
          pinned_column.cpp
          vector_search.cpp
          vector_search_ann.cpp
          vector_search_enn.cpp)
