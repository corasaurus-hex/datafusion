// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::{
    array::{ArrayRef, StringArray, UInt64Array},
    record_batch::RecordBatch,
};
use arrow_schema::{SchemaRef, SortOptions};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr, expressions::col};
use datafusion_physical_plan::test::TestMemoryExec;
use datafusion_physical_plan::{
    collect, sorts::sort_preserving_merge::SortPreservingMergeExec,
};

use rand::SeedableRng;
use rand::distr::Distribution;
use rand::rngs::StdRng;
use rand_distr::Geometric;
use std::sync::Arc;

const BENCH_ROWS: usize = 1_000_000;
const BENCH_ROWS_U64: usize = 10_000_000;

fn get_large_string(idx: usize) -> String {
    let base_content = [
        concat!(
            "# Advanced Topics in Computer Science\n\n",
            "## Summary\nThis article explores complex system design patterns and...\n\n",
            "```rust\nfn process_data(data: &mut [i32]) {\n    // Parallel processing example\n    data.par_iter_mut().for_each(|x| *x *= 2);\n}\n```\n\n",
            "## Performance Considerations\nWhen implementing concurrent systems...\n"
        ),
        concat!(
            "## API Documentation\n\n",
            "```json\n{\n  \"endpoint\": \"/api/v2/users\",\n  \"methods\": [\"GET\", \"POST\"],\n  \"parameters\": {\n    \"page\": \"number\"\n  }\n}\n```\n\n",
            "# Authentication Guide\nSecure your API access using OAuth 2.0...\n"
        ),
        concat!(
            "# Data Processing Pipeline\n\n",
            "```python\nfrom multiprocessing import Pool\n\ndef main():\n    with Pool(8) as p:\n        results = p.map(process_item, data)\n```\n\n",
            "## Summary of Optimizations\n1. Batch processing\n2. Memory pooling\n3. Concurrent I/O operations\n"
        ),
        concat!(
            "# System Architecture Overview\n\n",
            "## Components\n- Load Balancer\n- Database Cluster\n- Cache Service\n\n",
            "```go\nfunc main() {\n    router := gin.Default()\n    router.GET(\"/api/health\", healthCheck)\n    router.Run(\":8080\")\n}\n```\n"
        ),
        concat!(
            "## Configuration Reference\n\n",
            "```yaml\nserver:\n  port: 8080\n  max_threads: 32\n\ndatabase:\n  url: postgres://user@prod-db:5432/main\n```\n\n",
            "# Deployment Strategies\nBlue-green deployment patterns with...\n"
        ),
    ];
    base_content[idx % base_content.len()].to_string()
}

fn generate_sorted_string_column(rows: usize) -> ArrayRef {
    let mut values = Vec::with_capacity(rows);
    for i in 0..rows {
        values.push(get_large_string(i));
    }
    values.sort();
    Arc::new(StringArray::from(values))
}

fn generate_sorted_u64_column(rows: usize) -> ArrayRef {
    Arc::new(UInt64Array::from((0_u64..rows as u64).collect::<Vec<_>>()))
}

/// Generate partitions where each owns exclusive, non-overlapping ranges
/// that alternate round-robin across partitions.
///
/// Example with 3 partitions, block_size=3:
///   P0: [0,1,2,  9,10,11, 18,19,20, ...]
///   P1: [3,4,5, 12,13,14, 21,22,23, ...]
///   P2: [6,7,8, 15,16,17, 24,25,26, ...]
///
/// The same stream wins for exactly `block_size` rows before switching.
fn generate_nonoverlapping_blocks_u64(
    num_partitions: usize,
    rows_per_partition: usize,
    block_size: usize,
) -> Vec<Vec<RecordBatch>> {
    let total_values = (rows_per_partition * num_partitions) as u64;
    let all_values: Vec<u64> = (0..total_values).collect();
    // one vec of u64 values per partition, each becomes a single-column RecordBatch
    let mut partition_values: Vec<Vec<u64>> = (0..num_partitions)
        .map(|_| Vec::with_capacity(rows_per_partition))
        .collect();
    for (i, block) in all_values.chunks(block_size).enumerate() {
        partition_values[i % num_partitions].extend_from_slice(block);
    }
    // already sorted by construction
    partition_values
        .into_iter()
        .map(|values| {
            let array: ArrayRef = Arc::new(UInt64Array::from(values));
            let batch = RecordBatch::try_from_iter(vec![("col-0", array)]).unwrap();
            vec![batch]
        })
        .collect()
}

/// Generate partitions with overlapping ranges but clustered values.
///
/// The value space is divided into "stripes" of `run_length` consecutive
/// values. Stripes are assigned round-robin to partitions. The assigned
/// partition gets all `run_length` values in a stripe; every other
/// partition gets just the first value, creating overlap.
///
/// After sorting, each partition has long runs of consecutive values
/// (from its assigned stripes) interrupted by short bursts of duplicates
/// (the first values of stripes assigned to other partitions).
///
/// Models real range-partitioned data where each partition covers the
/// full range but has natural clusters of consecutive values.
fn generate_overlapping_clusters_u64(
    num_partitions: usize,
    rows_per_partition: usize,
    run_length: usize,
) -> Vec<Vec<RecordBatch>> {
    // pick total_stripes so each partition ends up with ~rows_per_partition values.
    // each partition gets run_length values from each of its assigned stripes
    // (1/N of all stripes), and 1 value from every other stripe. so:
    //   rows_per_partition = (total_stripes / N) * run_length
    //                      + (total_stripes - total_stripes / N) * 1
    // solving for total_stripes:
    let total_stripes =
        (rows_per_partition * num_partitions) / (run_length + num_partitions - 1);

    (0..num_partitions)
        .map(|partition| {
            let mut values: Vec<u64> = Vec::with_capacity(rows_per_partition);
            for stripe_idx in 0..total_stripes {
                let stripe_start = (stripe_idx * run_length) as u64;
                if stripe_idx % num_partitions == partition {
                    // assigned partition gets all values in the stripe
                    for i in 0..run_length as u64 {
                        values.push(stripe_start + i);
                    }
                } else {
                    // other partitions get just the first value
                    values.push(stripe_start);
                }
            }
            values.sort();
            let array: ArrayRef = Arc::new(UInt64Array::from(values));
            let batch = RecordBatch::try_from_iter(vec![("col-0", array)]).unwrap();
            vec![batch]
        })
        .collect()
}

/// Generate partitions with geometrically distributed run lengths.
///
/// Run lengths are drawn from a geometric distribution: most runs are
/// short (median = `median_run`), but occasional long runs occur
/// naturally. This models real-world data where run lengths vary
/// across partitions — e.g., time-range-partitioned files with
/// uneven coverage.
fn generate_geometric_runs_u64(
    num_partitions: usize,
    rows_per_partition: usize,
    median_run: usize,
) -> Vec<Vec<RecordBatch>> {
    // Geometric::new takes a probability p, but we want to specify the
    // median run length instead. Invert the median formula from
    // https://en.wikipedia.org/wiki/Geometric_distribution to get p.
    let p = 1.0 - (-(1.0 / median_run as f64)).exp2();
    let geo = Geometric::new(p).unwrap();
    let mut rng = StdRng::seed_from_u64(42);

    // build the global sorted sequence, then split into variable-length
    // runs and deal them round-robin across partitions
    let total_values = rows_per_partition * num_partitions;
    let all_values: Vec<u64> = (0..total_values as u64).collect();
    let mut partition_values: Vec<Vec<u64>> = (0..num_partitions)
        .map(|_| Vec::with_capacity(rows_per_partition))
        .collect();

    let mut offset = 0;
    let mut partition = 0;
    while offset < total_values {
        let run_len = (geo.sample(&mut rng) as usize + 1).min(total_values - offset);
        partition_values[partition]
            .extend_from_slice(&all_values[offset..offset + run_len]);
        offset += run_len;
        partition = (partition + 1) % num_partitions;
    }

    partition_values
        .into_iter()
        .map(|values| {
            let array: ArrayRef = Arc::new(UInt64Array::from(values));
            let batch = RecordBatch::try_from_iter(vec![("col-0", array)]).unwrap();
            vec![batch]
        })
        .collect()
}

fn create_partitions<const IS_LARGE_COLUMN_TYPE: bool>(
    num_partitions: usize,
    num_columns: usize,
    num_rows: usize,
) -> Vec<Vec<RecordBatch>> {
    (0..num_partitions)
        .map(|_| {
            let rows = (0..num_columns)
                .map(|i| {
                    (
                        format!("col-{i}"),
                        if IS_LARGE_COLUMN_TYPE {
                            generate_sorted_string_column(num_rows)
                        } else {
                            generate_sorted_u64_column(num_rows)
                        },
                    )
                })
                .collect::<Vec<_>>();

            let batch = RecordBatch::try_from_iter(rows).unwrap();
            vec![batch]
        })
        .collect()
}

struct BenchData {
    bench_name: String,
    partitions: Vec<Vec<RecordBatch>>,
    schema: SchemaRef,
    sort_order: LexOrdering,
}

fn get_bench_data() -> Vec<BenchData> {
    let mut ret = Vec::new();
    let mut push_bench_data = |bench_name: &str, partitions: Vec<Vec<RecordBatch>>| {
        let schema = partitions[0][0].schema();
        // Define sort order (col1 ASC, col2 ASC, col3 ASC)
        let sort_order = LexOrdering::new(schema.fields().iter().map(|field| {
            PhysicalSortExpr::new(
                col(field.name(), &schema).unwrap(),
                SortOptions::default(),
            )
        }))
        .unwrap();
        ret.push(BenchData {
            bench_name: bench_name.to_string(),
            partitions,
            schema,
            sort_order,
        });
    };
    // 1. single large string column
    {
        let partitions = create_partitions::<true>(3, 1, BENCH_ROWS);
        push_bench_data("single_large_string_column_with_1m_rows", partitions);
    }
    // 2. single u64 column
    {
        let partitions = create_partitions::<false>(3, 1, BENCH_ROWS_U64);
        push_bench_data("single_u64_column_with_10m_rows", partitions);
    }
    // 3. multiple large string columns
    {
        let partitions = create_partitions::<true>(3, 3, BENCH_ROWS);
        push_bench_data("multiple_large_string_columns_with_1m_rows", partitions);
    }
    // 4. multiple u64 columns
    {
        let partitions = create_partitions::<false>(3, 3, BENCH_ROWS_U64);
        push_bench_data("multiple_u64_columns_with_10m_rows", partitions);
    }
    // 5. u64 non-overlapping blocks — same stream wins for block_size rows (10M rows)
    for block_size in [100, 1000, 10_000] {
        let partitions =
            generate_nonoverlapping_blocks_u64(3, BENCH_ROWS_U64, block_size);
        push_bench_data(
            &format!("u64_nonoverlapping_blocks_{block_size}"),
            partitions,
        );
    }
    // 6. u64 overlapping clusters — long runs with short interruptions (10M rows)
    for run_length in [100, 1000] {
        let partitions = generate_overlapping_clusters_u64(3, BENCH_ROWS_U64, run_length);
        push_bench_data(
            &format!("u64_overlapping_clusters_{run_length}"),
            partitions,
        );
    }
    // 7. u64 geometric runs — realistic variable-length runs (10M rows)
    // median_run=10: mostly short runs (1-30), occasional longer
    // median_run=100: medium runs with wider spread
    for median_run in [1, 10, 100] {
        let partitions = generate_geometric_runs_u64(3, BENCH_ROWS_U64, median_run);
        push_bench_data(
            &format!("u64_geometric_runs_median_{median_run}"),
            partitions,
        );
    }
    ret
}

/// Add a benchmark to test the optimization effect of reusing Rows.
/// Run this benchmark with:
/// ```sh
/// cargo bench --features="bench"  --bench sort_preserving_merge -- --sample-size=10
/// ```
fn bench_merge_sorted_preserving(c: &mut Criterion) {
    let task_ctx = Arc::new(TaskContext::default());
    let bench_data = get_bench_data();
    for data in bench_data.into_iter() {
        let BenchData {
            bench_name,
            partitions,
            schema,
            sort_order,
        } = data;
        c.bench_function(
            &format!("bench_merge_sorted_preserving/{bench_name}"),
            |b| {
                b.iter_batched(
                    || {
                        let exec = TestMemoryExec::try_new_exec(
                            &partitions,
                            schema.clone(),
                            None,
                        )
                        .unwrap();
                        Arc::new(SortPreservingMergeExec::new(sort_order.clone(), exec))
                    },
                    |merge_exec| {
                        let rt = tokio::runtime::Runtime::new().unwrap();
                        rt.block_on(async {
                            collect(merge_exec, task_ctx.clone()).await.unwrap();
                        });
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }
}

criterion_group!(benches, bench_merge_sorted_preserving);
criterion_main!(benches);
