/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_transaction.hpp>

namespace duckdb {

// Register two table functions used by the preloaded-scan-region experiment:
//   CALL gpu_preload_region('<table>(<col1>,<col2>,...)[; ...]', '<encoded|decoded>')
//   CALL gpu_release_region()
void RegisterGpuPreloadRegionFunctions(CatalogTransaction& transaction, Catalog& catalog);

}  // namespace duckdb
