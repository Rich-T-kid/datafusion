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

//! End-to-end `RepartitionExec` benchmark focused on hash partitioning
//! shape (fan-out, fan-in, n-to-n). Each scenario builds a fixed in-memory
//! input, pipes it through a single `RepartitionExec`, and drains every
//! output stream concurrently.
//!
//! Columns: 10 `Int64` columns per batch. Hash keys: `k1` only (1-key
//! scenarios) and `k1,k2` (2-key scenarios). Batch size and per-partition
//! batch count are fixed, so the only variable per scenario is the input ×
//! output partition fan-out.

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringViewArray,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::expressions::col;
use datafusion_physical_expr::{Partitioning, PhysicalExpr};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::test::TestMemoryExec;
use futures::StreamExt;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tokio::runtime::Runtime;

const BATCH_ROWS: usize = 8_192;
/// Total rows per scenario; sized to mimic a mid-size TPCH query. Split evenly
/// across input partitions, so every shape processes the same total payload.
const TOTAL_ROWS: usize = 1_000_000;
const KEY_CARDINALITY: i64 = 100_000;
const SEED: u64 = 0xABCD_EF01;

/// Schema with 10 columns, mixing primitives and `Utf8View` to approximate the
/// column mix in TPCH tables (ids, measures, text).
///
/// Columns: `k1` i64, `k2` i64, `v_i32` i32, `v_i64` i64, `v_f64_a` f64,
/// `v_f64_b` f64, `v_sv_short` utf8view, `v_sv_long` utf8view, `v_i64_pay` i64,
/// `v_i32_pay` i32.
fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k1", DataType::Int64, false),
        Field::new("k2", DataType::Int64, false),
        Field::new("v_i32", DataType::Int32, false),
        Field::new("v_i64", DataType::Int64, false),
        Field::new("v_f64_a", DataType::Float64, false),
        Field::new("v_f64_b", DataType::Float64, false),
        Field::new("v_sv_short", DataType::Utf8View, false),
        Field::new("v_sv_long", DataType::Utf8View, false),
        Field::new("v_i64_pay", DataType::Int64, false),
        Field::new("v_i32_pay", DataType::Int32, false),
    ]))
}

fn make_batch(schema: &SchemaRef, rows: usize, seed_offset: u64) -> RecordBatch {
    let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(seed_offset));

    let k1: Vec<i64> = (0..rows)
        .map(|_| rng.random_range(0..KEY_CARDINALITY))
        .collect();
    let k2: Vec<i64> = (0..rows)
        .map(|_| rng.random_range(0..KEY_CARDINALITY))
        .collect();
    let v_i32: Vec<i32> = (0..rows).map(|_| rng.random::<i32>()).collect();
    let v_i64: Vec<i64> = (0..rows).map(|_| rng.random::<i64>()).collect();
    let v_f64_a: Vec<f64> = (0..rows).map(|_| rng.random::<f64>()).collect();
    let v_f64_b: Vec<f64> = (0..rows).map(|_| rng.random::<f64>()).collect();

    // Short strings (inline in StringView), ~10 bytes each.
    let v_sv_short: Vec<String> = (0..rows)
        .map(|_| format!("s{:08}", rng.random_range(0..10_000_000u32)))
        .collect();
    // Longer strings (heap-allocated in StringView), ~24 bytes each.
    let v_sv_long: Vec<String> = (0..rows)
        .map(|_| format!("longer_text_payload_{:04}", rng.random_range(0..10_000u32)))
        .collect();

    let v_i64_pay: Vec<i64> = (0..rows).map(|_| rng.random::<i64>()).collect();
    let v_i32_pay: Vec<i32> = (0..rows).map(|_| rng.random::<i32>()).collect();

    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(k1)),
        Arc::new(Int64Array::from(k2)),
        Arc::new(Int32Array::from(v_i32)),
        Arc::new(Int64Array::from(v_i64)),
        Arc::new(Float64Array::from(v_f64_a)),
        Arc::new(Float64Array::from(v_f64_b)),
        Arc::new(StringViewArray::from_iter_values(
            v_sv_short.iter().map(String::as_str),
        )),
        Arc::new(StringViewArray::from_iter_values(
            v_sv_long.iter().map(String::as_str),
        )),
        Arc::new(Int64Array::from(v_i64_pay)),
        Arc::new(Int32Array::from(v_i32_pay)),
    ];
    RecordBatch::try_new(Arc::clone(schema), cols).unwrap()
}

/// Build `num_input_partitions` input streams such that the TOTAL row count
/// across all inputs is ~`TOTAL_ROWS`. Each input stream holds enough
/// `BATCH_ROWS`-sized batches to carry its share of the work.
fn build_input_partitions(
    schema: &SchemaRef,
    num_input_partitions: usize,
) -> Vec<Vec<RecordBatch>> {
    let rows_per_input = TOTAL_ROWS.div_ceil(num_input_partitions);
    let batches_per_input = rows_per_input.div_ceil(BATCH_ROWS);

    (0..num_input_partitions)
        .map(|p| {
            let mut remaining = rows_per_input;
            (0..batches_per_input)
                .map(|b| {
                    let rows = remaining.min(BATCH_ROWS);
                    remaining -= rows;
                    make_batch(schema, rows, (p * 10_000 + b) as u64)
                })
                .collect()
        })
        .collect()
}

fn hash_partitioning(
    schema: &SchemaRef,
    num_keys: usize,
    num_output_partitions: usize,
) -> Partitioning {
    let mut exprs: Vec<Arc<dyn PhysicalExpr>> = Vec::with_capacity(num_keys);
    exprs.push(col("k1", schema).unwrap());
    if num_keys >= 2 {
        exprs.push(col("k2", schema).unwrap());
    }
    Partitioning::Hash(exprs, num_output_partitions)
}

/// Build `RepartitionExec` + drain all output streams concurrently to
/// completion. Returns the total row count so the compiler can't elide the
/// work. All setup (building the input, constructing the exec) is included
/// in the measured region so the delta reflects the full pipeline, which is
/// what shows up in profiles.
async fn run_once(
    schema: SchemaRef,
    input_partitions: Vec<Vec<RecordBatch>>,
    partitioning: Partitioning,
    task_ctx: Arc<TaskContext>,
) -> usize {
    let source =
        TestMemoryExec::try_new_exec(&input_partitions, Arc::clone(&schema), None)
            .unwrap();
    let exec = Arc::new(RepartitionExec::try_new(source, partitioning).unwrap());

    let n_out = exec.partitioning().partition_count();
    let mut handles = Vec::with_capacity(n_out);
    for p in 0..n_out {
        let exec = Arc::clone(&exec);
        let ctx = Arc::clone(&task_ctx);
        handles.push(tokio::spawn(async move {
            let mut stream = exec.execute(p, ctx).unwrap();
            let mut rows = 0usize;
            while let Some(res) = stream.next().await {
                let batch = res.unwrap();
                rows += batch.num_rows();
            }
            rows
        }));
    }

    let mut total = 0usize;
    for h in handles {
        total += h.await.unwrap();
    }
    total
}

/// Shapes exercised: (input_partitions, output_partitions, num_keys).
/// Grouped into fan-out, fan-in, and n-to-n to make profiling comparisons
/// easy to read.
const SHAPES: &[(usize, usize, usize)] = &[
    // Fan-out: 1 input stream → N output streams.
    (1, 4, 1),
    (1, 16, 1),
    (1, 64, 1),
    (1, 64, 2),
    // Fan-in: N input streams → 1 output stream.
    (4, 1, 1),
    (16, 1, 1),
    (64, 1, 1),
    // N-to-N: equal fan-out/fan-in.
    (4, 4, 1),
    (16, 16, 1),
    (64, 64, 1),
    (16, 16, 2),
];

fn bench_repartition_shapes(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let schema = schema();
    let task_ctx = Arc::new(TaskContext::default());

    let mut group = c.benchmark_group("repartition_shapes");
    // Each sample reruns full pipeline; keep sample count modest to bound
    // bench wall-time.
    group.sample_size(20);

    for &(n_in, n_out, n_keys) in SHAPES {
        let input_partitions = build_input_partitions(&schema, n_in);
        let partitioning = hash_partitioning(&schema, n_keys, n_out);
        let id = format!("in{n_in}_out{n_out}_keys{n_keys}");

        group.bench_with_input(
            BenchmarkId::new("hash", id),
            &(n_in, n_out, n_keys),
            |b, _| {
                b.iter(|| {
                    let schema = Arc::clone(&schema);
                    let input = input_partitions.clone();
                    let part = partitioning.clone();
                    let ctx = Arc::clone(&task_ctx);
                    let rows =
                        rt.block_on(
                            async move { run_once(schema, input, part, ctx).await },
                        );
                    black_box(rows)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_repartition_shapes);
criterion_main!(benches);
