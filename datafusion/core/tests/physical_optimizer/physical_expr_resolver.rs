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

//! Integration tests for [`PhysicalExprResolver`] optimizer rule.

use std::{collections::HashMap, sync::Arc};

use arrow::array::record_batch;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::{assert_batches_eq, config::ConfigOptions, prelude::SessionContext};
use datafusion_common::{ParamValues, Result, ScalarValue};
use datafusion_execution::TaskContext;
use datafusion_expr::Operator;
use datafusion_physical_expr::{
    Partitioning,
    expressions::{BinaryExpr, col, lit, placeholder},
};
use datafusion_physical_optimizer::{
    PhysicalOptimizerRule, physical_expr_resolver::PhysicalExprResolver,
};
use datafusion_physical_plan::{
    ExecutionPlan, collect, filter::FilterExec, get_plan_string,
    plan_transformer::TransformPlanExec, repartition::RepartitionExec,
};

use crate::physical_optimizer::test_utils::{
    coalesce_partitions_exec, global_limit_exec, resolve_placeholders_exec, stream_exec,
};

fn create_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("c1", DataType::Int32, true),
        Field::new("c2", DataType::Int32, true),
        Field::new("c3", DataType::Int32, true),
    ]))
}

fn filter_exec(
    schema: SchemaRef,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    Ok(Arc::new(FilterExec::try_new(
        Arc::new(BinaryExpr::new(
            col("c3", schema.as_ref()).unwrap(),
            Operator::Gt,
            lit(0),
        )),
        input,
    )?))
}

fn filter_exec_with_placeholders(
    schema: SchemaRef,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    Ok(Arc::new(FilterExec::try_new(
        Arc::new(BinaryExpr::new(
            col("c3", schema.as_ref()).unwrap(),
            Operator::Gt,
            placeholder("$foo", DataType::Int32),
        )),
        input,
    )?))
}

fn repartition_exec(
    streaming_table: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    Ok(Arc::new(RepartitionExec::try_new(
        streaming_table,
        Partitioning::RoundRobinBatch(8),
    )?))
}

#[test]
fn test_noop_if_no_placeholders_found() -> Result<()> {
    let schema = create_schema();
    let streaming_table = stream_exec(&schema);
    let repartition = repartition_exec(streaming_table)?;
    let filter = filter_exec(schema, repartition)?;
    let coalesce_partitions = coalesce_partitions_exec(filter);
    let plan = global_limit_exec(coalesce_partitions, 0, Some(5));

    let initial = get_plan_string(&plan);
    let expected_initial = [
        "GlobalLimitExec: skip=0, fetch=5",
        "  CoalescePartitionsExec",
        "    FilterExec: c3@2 > 0",
        "      RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "        StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    assert_eq!(initial, expected_initial);

    let after_optimize = PhysicalExprResolver::new_post_optimization()
        .optimize(plan, &ConfigOptions::new())?;

    let optimized_plan_string = get_plan_string(&after_optimize);
    assert_eq!(initial, optimized_plan_string);

    Ok(())
}

#[test]
fn test_wrap_plan_with_transformer() -> Result<()> {
    let schema = create_schema();
    let streaming_table = stream_exec(&schema);
    let repartition = repartition_exec(streaming_table)?;
    let filter = filter_exec_with_placeholders(schema, repartition)?;
    let coalesce_partitions = coalesce_partitions_exec(filter);
    let plan = global_limit_exec(coalesce_partitions, 0, Some(5));

    let initial = get_plan_string(&plan);
    let expected_initial = [
        "GlobalLimitExec: skip=0, fetch=5",
        "  CoalescePartitionsExec",
        "    FilterExec: c3@2 > $foo",
        "      RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "        StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    assert_eq!(initial, expected_initial);

    let after_optimize = PhysicalExprResolver::new_post_optimization()
        .optimize(plan, &ConfigOptions::new())?;

    let expected_optimized = [
        "TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=1]",
        "  GlobalLimitExec: skip=0, fetch=5",
        "    CoalescePartitionsExec",
        "      FilterExec: c3@2 > $foo",
        "        RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "          StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    let optimized_plan_string = get_plan_string(&after_optimize);
    assert_eq!(optimized_plan_string, expected_optimized);

    let transformer = after_optimize
        .as_ref()
        .as_any()
        .downcast_ref::<TransformPlanExec>()
        .expect("should downcast");

    let param_values = ParamValues::Map(HashMap::from_iter([(
        "foo".to_string(),
        ScalarValue::Int32(Some(100)).into(),
    )]));

    let ctx = Arc::new(TaskContext::default().with_param_values(param_values));
    let resolved_plan = transformer.transform(&ctx)?;
    let resolved_plan_string = get_plan_string(&resolved_plan);
    let expected_resolved = [
        "GlobalLimitExec: skip=0, fetch=5",
        "  CoalescePartitionsExec",
        "    FilterExec: c3@2 > 100",
        "      RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "        StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    assert_eq!(resolved_plan_string, expected_resolved);

    Ok(())
}

#[test]
fn test_remove_useless_transformers() -> Result<()> {
    let schema = create_schema();
    let streaming_table = stream_exec(&schema);
    let repartition = repartition_exec(streaming_table)?;
    let filter = filter_exec(schema, repartition)?;
    let transformer = resolve_placeholders_exec(filter);
    let coalesce_partitions = coalesce_partitions_exec(transformer);
    let global_limit = global_limit_exec(coalesce_partitions, 0, Some(5));
    let plan = resolve_placeholders_exec(global_limit);

    let initial = get_plan_string(&plan);
    let expected_initial = [
        "TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=0]",
        "  GlobalLimitExec: skip=0, fetch=5",
        "    CoalescePartitionsExec",
        "      TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=0]",
        "        FilterExec: c3@2 > 0",
        "          RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "            StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    assert_eq!(initial, expected_initial);

    let after_optimize =
        PhysicalExprResolver::new().optimize(plan, &ConfigOptions::new())?;

    let expected_optimized = [
        "GlobalLimitExec: skip=0, fetch=5",
        "  CoalescePartitionsExec",
        "    FilterExec: c3@2 > 0",
        "      RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "        StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    let optimized_plan_string = get_plan_string(&after_optimize);
    assert_eq!(optimized_plan_string, expected_optimized);

    Ok(())
}

#[test]
fn test_combine_transformers() -> Result<()> {
    let schema = create_schema();
    let streaming_table = stream_exec(&schema);
    let repartition = repartition_exec(streaming_table)?;
    let transformer = resolve_placeholders_exec(repartition);
    let filter = filter_exec_with_placeholders(schema, transformer)?;
    let transformer = resolve_placeholders_exec(filter);
    let coalesce_partitions = coalesce_partitions_exec(transformer);
    let global_limit = global_limit_exec(coalesce_partitions, 0, Some(5));
    let plan = resolve_placeholders_exec(global_limit);

    let initial = get_plan_string(&plan);
    let expected_initial = [
        "TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=0]",
        "  GlobalLimitExec: skip=0, fetch=5",
        "    CoalescePartitionsExec",
        "      TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=1]",
        "        FilterExec: c3@2 > $foo",
        "          TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=0]",
        "            RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "              StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    assert_eq!(initial, expected_initial);

    let after_pre_optimization =
        PhysicalExprResolver::new().optimize(plan, &ConfigOptions::new())?;

    let expected_optimized = [
        "GlobalLimitExec: skip=0, fetch=5",
        "  CoalescePartitionsExec",
        "    FilterExec: c3@2 > $foo",
        "      RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "        StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    let optimized_plan_string = get_plan_string(&after_pre_optimization);
    assert_eq!(optimized_plan_string, expected_optimized);

    let after_post_optimization = PhysicalExprResolver::new_post_optimization()
        .optimize(after_pre_optimization, &ConfigOptions::new())?;

    let expected_optimized = [
        "TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=1]",
        "  GlobalLimitExec: skip=0, fetch=5",
        "    CoalescePartitionsExec",
        "      FilterExec: c3@2 > $foo",
        "        RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=1",
        "          StreamingTableExec: partition_sizes=1, projection=[c1, c2, c3], infinite_source=true",
    ];

    let optimized_plan_string = get_plan_string(&after_post_optimization);
    assert_eq!(optimized_plan_string, expected_optimized);

    Ok(())
}

#[tokio::test]
async fn test_resolve_window_function() -> Result<()> {
    let ctx = SessionContext::new();
    let batch = record_batch!(("id", Int32, [1, 2]), ("name", Utf8, ["Alex", "Bob"]))?;
    ctx.register_batch("t1", batch)?;

    let plan = ctx
        .sql("SELECT id, SUM(id + $1) OVER (PARTITION BY name ORDER BY id) FROM t1")
        .await?
        .create_physical_plan()
        .await?;

    let expected_plan = [
        "TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=1]",
        "  ProjectionExec: expr=[id@0 as id, sum(t1.id + $1) PARTITION BY [t1.name] ORDER BY [t1.id ASC NULLS LAST] RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW@2 as sum(t1.id + $1) PARTITION BY [t1.name] ORDER BY [t1.id ASC NULLS LAST] RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW]",
        "    BoundedWindowAggExec: wdw=[sum(t1.id + $1) PARTITION BY [t1.name] ORDER BY [t1.id ASC NULLS LAST] RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW: Field { \"sum(t1.id + $1) PARTITION BY [t1.name] ORDER BY [t1.id ASC NULLS LAST] RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW\": nullable Int64 }, frame: RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW], mode=[Sorted]",
        "      SortExec: expr=[name@1 ASC NULLS LAST, id@0 ASC NULLS LAST], preserve_partitioning=[false]",
        "        DataSourceExec: partitions=1, partition_sizes=[1]",
    ];

    let plan_string = get_plan_string(&plan);
    assert_eq!(plan_string, expected_plan);

    let param_values = ParamValues::List(vec![ScalarValue::Int32(Some(100)).into()]);
    let task_ctx = Arc::new(TaskContext::from(&ctx).with_param_values(param_values));
    let batch = collect(plan, task_ctx).await?;

    assert_batches_eq!(
        [
            "+----+--------------------------------------------------------------------------------------------------------------------------+",
            "| id | sum(t1.id + $1) PARTITION BY [t1.name] ORDER BY [t1.id ASC NULLS LAST] RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW |",
            "+----+--------------------------------------------------------------------------------------------------------------------------+",
            "| 1  | 101                                                                                                                      |",
            "| 2  | 102                                                                                                                      |",
            "+----+--------------------------------------------------------------------------------------------------------------------------+",
        ],
        &batch
    );

    Ok(())
}

#[tokio::test]
async fn test_resolve_join_function() -> Result<()> {
    let ctx = SessionContext::new();
    let batch_1 = record_batch!(
        ("id", Int32, [1, 2]),
        ("name", Utf8, ["Alex", "Bob"]),
        ("age", Int32, [30, 40])
    )?;

    let batch_2 = record_batch!(
        ("id", Int32, [10, 20]),
        ("name", Utf8, ["Carol", "David"]),
        ("age", Int32, [35, 45])
    )?;

    ctx.register_batch("t1", batch_1)?;
    ctx.register_batch("t2", batch_2)?;

    let plan = ctx
        .sql("SELECT t1.name, t2.age FROM t1 JOIN t2 ON t1.id + $1 = t2.id;")
        .await?
        .create_physical_plan()
        .await?;

    let expected_plan = [
        "TransformPlanExec: rules=[ResolvePlaceholders: plans_to_modify=1]",
        "  ProjectionExec: expr=[name@1 as name, age@0 as age]",
        "    HashJoinExec: mode=CollectLeft, join_type=Inner, on=[(id@0, t1.id + $1@2)], projection=[age@1, name@3]",
        "      DataSourceExec: partitions=1, partition_sizes=[1]",
        "      ProjectionExec: expr=[id@0 as id, name@1 as name, id@0 + $1 as t1.id + $1]",
        "        DataSourceExec: partitions=1, partition_sizes=[1]",
    ];

    let plan_string = get_plan_string(&plan);
    assert_eq!(plan_string, expected_plan);

    let param_values = ParamValues::List(vec![ScalarValue::Int32(Some(8)).into()]);
    let task_ctx = Arc::new(TaskContext::from(&ctx).with_param_values(param_values));
    let batch = collect(plan, task_ctx).await?;

    assert_batches_eq!(
        [
            "+------+-----+",
            "| name | age |",
            "+------+-----+",
            "| Bob  | 35  |",
            "+------+-----+",
        ],
        &batch
    );

    Ok(())
}

#[tokio::test]
async fn test_resolve_cast() -> Result<()> {
    let ctx = SessionContext::new();
    let plan = ctx
        .sql("SELECT CAST($1 as INT)")
        .await?
        .create_physical_plan()
        .await?;

    let param_values = ParamValues::List(vec![
        ScalarValue::Utf8(Some("not a number".to_string())).into(),
    ]);

    let task_ctx = Arc::new(TaskContext::from(&ctx).with_param_values(param_values));
    let result = collect(Arc::clone(&plan), task_ctx).await;
    assert!(result.is_err());

    let param_values =
        ParamValues::List(vec![ScalarValue::Utf8(Some("200".to_string())).into()]);
    let task_ctx = Arc::new(TaskContext::from(&ctx).with_param_values(param_values));
    let batch = collect(plan, task_ctx).await?;

    assert_batches_eq!(
        ["+-----+", "| $1  |", "+-----+", "| 200 |", "+-----+"],
        &batch
    );

    Ok(())
}

#[tokio::test]
async fn test_resolve_try_cast() -> Result<()> {
    let ctx = SessionContext::new();
    let plan = ctx
        .sql("SELECT TRY_CAST($1 as INT)")
        .await?
        .create_physical_plan()
        .await?;

    let param_values = ParamValues::List(vec![
        ScalarValue::Utf8(Some("not a number".to_string())).into(),
    ]);

    let task_ctx = Arc::new(TaskContext::from(&ctx).with_param_values(param_values));
    let batch = collect(Arc::clone(&plan), task_ctx).await?;
    assert_batches_eq!(["+----+", "| $1 |", "+----+", "|    |", "+----+"], &batch);

    let param_values =
        ParamValues::List(vec![ScalarValue::Utf8(Some("200".to_string())).into()]);
    let task_ctx = Arc::new(TaskContext::from(&ctx).with_param_values(param_values));
    let batch = collect(plan, task_ctx).await?;

    assert_batches_eq!(
        ["+-----+", "| $1  |", "+-----+", "| 200 |", "+-----+"],
        &batch
    );

    Ok(())
}

#[tokio::test]
async fn test_resolve_const_expr() -> Result<()> {
    let ctx = SessionContext::new();
    let plan = ctx
        .sql("SELECT 10 + 5 * $1")
        .await?
        .create_physical_plan()
        .await?;

    let transformer = plan
        .as_ref()
        .as_any()
        .downcast_ref::<TransformPlanExec>()
        .expect("should downcast");

    let param_values = ParamValues::List(vec![ScalarValue::Int32(Some(3)).into()]);

    let ctx = Arc::new(TaskContext::default().with_param_values(param_values));
    let resolved_plan = transformer.transform(&ctx)?;
    let resolved_plan_string = get_plan_string(&resolved_plan);

    let expected_resolved = [
        "ProjectionExec: expr=[25 as Int64(10) + Int64(5) * $1]",
        "  PlaceholderRowExec",
    ];

    assert_eq!(resolved_plan_string, expected_resolved);

    Ok(())
}
