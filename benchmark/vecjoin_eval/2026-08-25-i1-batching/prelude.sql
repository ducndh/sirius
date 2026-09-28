SET memory_limit='32GB';
SET temp_directory='/var/tmp/ddb_spill';
SELECT * FROM pin_table(name => 'base',     tier => 'gpu', format => 'duckdb');
SELECT * FROM pin_table(name => 'corpus64', tier => 'gpu', format => 'duckdb');
SELECT * FROM pin_table(name => 'corpus256',tier => 'gpu', format => 'duckdb');
SELECT * FROM sirius_kmeans_fit('base','vec', name => 'c64',  n_clusters => 64);
SELECT * FROM sirius_kmeans_fit('base','vec', name => 'c256', n_clusters => 256);
