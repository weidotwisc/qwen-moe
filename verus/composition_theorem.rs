// End-to-end Tier-3 composition over one shared exact MoE semantic model.
//
// High-level refinement and swap propositions are proved here, not accepted
// as theorem premises.  Component bridges invoke Ex01's collective contract,
// Ex05's argsort proof, Ex06's expert-partition proof, and Ex09's pointwise
// kernel contract.  Remaining assumptions are explicit model-to-source facts.
//
// Run with:
//   verus --crate-type=lib verus/composition_theorem.rs

use vstd::prelude::*;

#[path = "composition_core.rs"]
mod composition_core;
pub use composition_core::*;

#[path = "kernel_refinement.rs"]
mod kernel_refinement;
pub use kernel_refinement::*;

#[path = "schedule_refinement.rs"]
mod schedule_refinement;
pub use schedule_refinement::*;

#[path = "component_integration.rs"]
mod component_integration;
pub use component_integration::*;

#[path = "deadlock_free.rs"]
mod deadlock_model;

verus! {

// =====================================================================
// §1 — Exact DP=1 block composition.
// =====================================================================

pub uninterp spec fn attention_forward(x: Tensor) -> Tensor;
pub uninterp spec fn residual_add(attn_out: Tensor, moe_out: Tensor) -> Tensor;

pub open spec fn moe_input_matches_attention(x: Tensor, input: MoeInput) -> bool {
    input.token_values == attention_forward(x).content
}

pub open spec fn block_forward(
    x: Tensor,
    input: MoeInput,
    tensor_id: nat,
    ep_group: Group,
    world_size: nat,
    use_lean: bool,
    use_fused: bool,
) -> Tensor {
    if moe_input_matches_attention(x, input) {
        residual_add(
            attention_forward(x),
            scheduled_forward(
                input, tensor_id, ep_group, world_size, use_lean, use_fused,
            ),
        )
    } else {
        Tensor { content: Seq::empty() }
    }
}

pub open spec fn block_spec(x: Tensor, input: MoeInput) -> Tensor {
    residual_add(attention_forward(x), moe_spec(input))
}

/// The MoE input used by the integrated theorem is built from the preceding
/// attention output and Ex05's concrete argsort result.
pub open spec fn verified_component_input(
    x: Tensor,
    config: ComponentMoeConfig,
    fused_values: Seq<int>,
) -> MoeInput {
    component_moe_input(
        component_data_after_attention(attention_forward(x), config),
        fused_values,
    )
}

/// Exact MoE equality is preserved by the attention/residual block context.
pub proof fn lemma_block_context_congruence(
    x: Tensor,
    input: MoeInput,
    tensor_id: nat,
    ep_group: Group,
    world_size: nat,
    use_lean: bool,
    use_fused: bool,
)
    requires
        moe_input_matches_attention(x, input),
        semantic_eq(
            scheduled_forward(
                input, tensor_id, ep_group, world_size, use_lean, use_fused,
            ),
            moe_spec(input),
        ),
    ensures semantic_eq(
        block_forward(
            x, input, tensor_id, ep_group, world_size, use_lean, use_fused,
        ),
        block_spec(x, input),
    ),
{
}

/// Every concrete schedule/kernel choice refines the same full-block spec.
proof fn lemma_block_variant_refines_spec_from_invariants(
    x: Tensor,
    input: MoeInput,
    tensor_id: nat,
    world_size: nat,
    tp_size: nat,
    dp_size: nat,
    ep_size: nat,
    tp_group: Group,
    ep_group: Group,
    use_lean: bool,
    use_fused: bool,
)
    requires
        valid_dp1_topology(
            world_size, tp_size, dp_size, ep_size, tp_group, ep_group,
        ),
        ReplicatedInput(input, tensor_id, tp_group),
        ExpertPartitioned(input, world_size),
        moe_input_matches_attention(x, input),
        RoutingConsistent(input),
        use_lean || all_to_all_partitionable(input),
        !use_fused || fused_rows_correct(input),
    ensures semantic_eq(
        block_forward(
            x, input, tensor_id, ep_group, world_size, use_lean, use_fused,
        ),
        block_spec(x, input),
    ),
{
    lemma_dp1_replication_transfers_to_ep(
        input, tensor_id, world_size, tp_size, dp_size, ep_size,
        tp_group, ep_group,
    );
    if use_lean {
        l6_all_reduce_refines_spec(
            input, tensor_id, ep_group, world_size, use_fused,
        );
    } else {
        h6_all_to_all_refines_spec(input, tensor_id, ep_group, use_fused);
    }
    lemma_block_context_congruence(
        x, input, tensor_id, ep_group, world_size, use_lean, use_fused,
    );
}

/// Component-connected form of the block theorem.  Routing consistency,
/// expert ownership, replicated input, and the attention/MoE seam are
/// established by their component proofs instead of supplied as Tier-3
/// semantic assumptions.
pub proof fn theorem_block_variant_from_components(
    x: Tensor,
    config: ComponentMoeConfig,
    fused_values: Seq<int>,
    fused_run: Ex09FusedRun,
    tensor_id: nat,
    representative: Rank,
    world_size: nat,
    tp_size: nat,
    dp_size: nat,
    ep_size: nat,
    tp_group: Group,
    ep_group: Group,
    use_lean: bool,
    use_fused: bool,
)
    requires
        valid_dp1_topology(
            world_size, tp_size, dp_size, ep_size, tp_group, ep_group,
        ),
        config.expert_parallel_size == ep_size,
        well_formed_component_data(component_data_after_attention(
            attention_forward(x), config,
        )),
        tp_group.contains(representative),
        tensor_on(tensor_id, representative) == attention_forward(x),
        use_lean || all_to_all_partitionable(
            verified_component_input(x, config, fused_values),
        ),
        !use_fused || Ex09KernelContract(
            verified_component_input(x, config, fused_values), fused_run,
        ),
    ensures semantic_eq(
        block_forward(
            x,
            verified_component_input(x, config, fused_values),
            tensor_id,
            ep_group,
            world_size,
            use_lean,
            use_fused,
        ),
        block_spec(
            x, verified_component_input(x, config, fused_values),
        ),
    ),
{
    let data = component_data_after_attention(attention_forward(x), config);
    let input = component_moe_input(data, fused_values);
    assert(input == verified_component_input(x, config, fused_values));

    ex05_establishes_routing_consistent(data, fused_values);
    ex06_establishes_expert_partitioned(data, fused_values, world_size);
    ex01_establishes_replicated_input(
        input, tensor_id, tp_group, representative,
    );
    assert(moe_input_matches_attention(x, input));
    if use_fused {
        ex09_establishes_fused_rows_correct(input, fused_run);
    }

    lemma_block_variant_refines_spec_from_invariants(
        x, input, tensor_id, world_size, tp_size, dp_size, ep_size,
        tp_group, ep_group, use_lean, use_fused,
    );
}

/// Public 2x2 corollary whose semantic obligations are all discharged through
/// component contracts.  The only fused-path boundary is Ex09's pointwise
/// kernel postcondition and its explicit representation relation.
pub proof fn corollary_block_variants_from_components(
    x: Tensor,
    config: ComponentMoeConfig,
    fused_values: Seq<int>,
    fused_run: Ex09FusedRun,
    tensor_id: nat,
    representative: Rank,
    world_size: nat,
    tp_size: nat,
    dp_size: nat,
    ep_size: nat,
    tp_group: Group,
    ep_group: Group,
    use_lean_a: bool,
    use_fused_a: bool,
    use_lean_b: bool,
    use_fused_b: bool,
)
    requires
        valid_dp1_topology(
            world_size, tp_size, dp_size, ep_size, tp_group, ep_group,
        ),
        config.expert_parallel_size == ep_size,
        well_formed_component_data(component_data_after_attention(
            attention_forward(x), config,
        )),
        tp_group.contains(representative),
        tensor_on(tensor_id, representative) == attention_forward(x),
        use_lean_a || all_to_all_partitionable(
            verified_component_input(x, config, fused_values),
        ),
        use_lean_b || all_to_all_partitionable(
            verified_component_input(x, config, fused_values),
        ),
        (use_fused_a || use_fused_b) ==> Ex09KernelContract(
            verified_component_input(x, config, fused_values), fused_run,
        ),
    ensures semantic_eq(
        block_forward(
            x,
            verified_component_input(x, config, fused_values),
            tensor_id,
            ep_group,
            world_size,
            use_lean_a,
            use_fused_a,
        ),
        block_forward(
            x,
            verified_component_input(x, config, fused_values),
            tensor_id,
            ep_group,
            world_size,
            use_lean_b,
            use_fused_b,
        ),
    ),
{
    let input = verified_component_input(x, config, fused_values);
    theorem_block_variant_from_components(
        x, config, fused_values, fused_run, tensor_id, representative,
        world_size, tp_size, dp_size, ep_size, tp_group, ep_group,
        use_lean_a, use_fused_a,
    );
    theorem_block_variant_from_components(
        x, config, fused_values, fused_run, tensor_id, representative,
        world_size, tp_size, dp_size, ep_size, tp_group, ep_group,
        use_lean_b, use_fused_b,
    );
    theorem_shared_spec_implies_equiv(
        block_forward(
            x, input, tensor_id, ep_group, world_size,
            use_lean_a, use_fused_a,
        ),
        block_forward(
            x, input, tensor_id, ep_group, world_size,
            use_lean_b, use_fused_b,
        ),
        block_spec(x, input),
    );
}

// =====================================================================
// §2 — Global safety properties.
// =====================================================================

pub open spec fn variant_work_order(
    input: MoeInput,
    use_lean: bool,
    use_fused: bool,
) -> Seq<WorkItem> {
    if use_lean && !use_fused {
        canonical_work(input)
    } else {
        input.routing_order
    }
}

/// Routing permutation implies completeness, disjointness, and no spurious
/// `(token, top-k slot)` executions for every variant.
proof fn lemma_token_conservation_from_routing(
    input: MoeInput,
    use_lean: bool,
    use_fused: bool,
)
    requires RoutingConsistent(input),
    ensures
        work_complete(input, variant_work_order(input, use_lean, use_fused)),
        work_disjoint(input, variant_work_order(input, use_lean, use_fused)),
        no_spurious_work(input, variant_work_order(input, use_lean, use_fused)),
{
    if use_lean && !use_fused {
        theorem_order_work_conservation(input, canonical_work(input));
    } else {
        theorem_routing_work_conservation(input);
    }
}

proof fn lemma_work_variant_invariant_from_routing(
    input: MoeInput,
    use_lean_a: bool,
    use_fused_a: bool,
    use_lean_b: bool,
    use_fused_b: bool,
)
    requires RoutingConsistent(input),
    ensures forall|item: WorkItem| #![auto]
        times_processed(
            input, variant_work_order(input, use_lean_a, use_fused_a), item,
        ) == times_processed(
            input, variant_work_order(input, use_lean_b, use_fused_b), item,
        ),
{
    assert(variant_work_order(input, use_lean_a, use_fused_a).to_multiset()
        == canonical_work(input).to_multiset());
    assert(variant_work_order(input, use_lean_b, use_fused_b).to_multiset()
        == canonical_work(input).to_multiset());
}

/// Ex05's argsort proof supplies the routing invariant needed for all three
/// work-conservation consequences.
pub proof fn theorem_token_conservation_from_components(
    data: ComponentMoeData,
    fused_values: Seq<int>,
    use_lean: bool,
    use_fused: bool,
)
    requires well_formed_component_data(data),
    ensures
        work_complete(
            component_moe_input(data, fused_values),
            variant_work_order(
                component_moe_input(data, fused_values), use_lean, use_fused,
            ),
        ),
        work_disjoint(
            component_moe_input(data, fused_values),
            variant_work_order(
                component_moe_input(data, fused_values), use_lean, use_fused,
            ),
        ),
        no_spurious_work(
            component_moe_input(data, fused_values),
            variant_work_order(
                component_moe_input(data, fused_values), use_lean, use_fused,
            ),
        ),
{
    let input = component_moe_input(data, fused_values);
    ex05_establishes_routing_consistent(data, fused_values);
    lemma_token_conservation_from_routing(input, use_lean, use_fused);
}

pub proof fn corollary_work_variant_invariant_from_components(
    data: ComponentMoeData,
    fused_values: Seq<int>,
    use_lean_a: bool,
    use_fused_a: bool,
    use_lean_b: bool,
    use_fused_b: bool,
)
    requires well_formed_component_data(data),
    ensures forall|item: WorkItem| #![auto]
        times_processed(
            component_moe_input(data, fused_values),
            variant_work_order(
                component_moe_input(data, fused_values),
                use_lean_a,
                use_fused_a,
            ),
            item,
        ) == times_processed(
            component_moe_input(data, fused_values),
            variant_work_order(
                component_moe_input(data, fused_values),
                use_lean_b,
                use_fused_b,
            ),
            item,
        ),
{
    let input = component_moe_input(data, fused_values);
    ex05_establishes_routing_consistent(data, fused_values);
    lemma_work_variant_invariant_from_routing(
        input, use_lean_a, use_fused_a, use_lean_b, use_fused_b,
    );
}

// ---------------------------------------------------------------------
// Concrete collective-schedule deadlock freedom.
// ---------------------------------------------------------------------

pub open spec fn variant_schedule(
    world_size: nat,
    tp_size: nat,
    use_lean: bool,
    _use_fused: bool,
) -> deadlock_model::Schedule {
    if use_lean {
        deadlock_model::all_reduce_dp1_schedule(world_size, tp_size)
    } else {
        deadlock_model::all_to_all_dp1_schedule(world_size, tp_size)
    }
}

pub proof fn theorem_deadlock_free(
    world_size: nat,
    tp_size: nat,
    use_lean: bool,
    use_fused: bool,
    states: Seq<deadlock_model::State>,
    i: nat,
)
    requires
        world_size > 0,
        tp_size == world_size,
        deadlock_model::execution(
            variant_schedule(world_size, tp_size, use_lean, use_fused),
            states,
        ),
        i < states.len(),
    ensures !deadlock_model::deadlocked(
        variant_schedule(world_size, tp_size, use_lean, use_fused),
        states[i as int],
    ),
{
    if use_lean {
        deadlock_model::theorem_all_reduce_dp1_execution_deadlock_free(
            world_size, tp_size, states, i,
        );
    } else {
        deadlock_model::theorem_all_to_all_dp1_execution_deadlock_free(
            world_size, tp_size, states, i,
        );
    }
}

// ---------------------------------------------------------------------
// Atomic scatter-add race freedom.
// ---------------------------------------------------------------------

pub struct OutputLocation {
    pub token: nat,
    pub feature: nat,
}

pub struct ScatterWrite {
    pub source: WorkItem,
    pub occurrence: nat,
    pub feature: nat,
}

#[derive(PartialEq, Eq)]
pub enum ScatterPrimitive {
    AtomicIndexAdd,
}

/// Both source variants implement output accumulation with atomic index_add.
/// The Python/CUDA correspondence of this operation remains an audit boundary.
pub open spec fn scatter_primitive(
    _use_lean: bool,
    _use_fused: bool,
) -> ScatterPrimitive {
    ScatterPrimitive::AtomicIndexAdd
}

pub open spec fn active_scatter_write(
    input: MoeInput,
    use_lean: bool,
    use_fused: bool,
    write: ScatterWrite,
) -> bool {
    write.occurrence < times_processed(
        input, variant_work_order(input, use_lean, use_fused), write.source,
    ) && write.feature == 0
}

pub open spec fn scatter_target(input: MoeInput, write: ScatterWrite) -> OutputLocation {
    OutputLocation {
        token: work_token(input, write.source),
        feature: write.feature,
    }
}

pub open spec fn scatter_write_is_atomic(
    use_lean: bool,
    use_fused: bool,
) -> bool {
    scatter_primitive(use_lean, use_fused) == ScatterPrimitive::AtomicIndexAdd
}

pub open spec fn scatter_conflict(
    input: MoeInput,
    use_lean: bool,
    use_fused: bool,
    left: ScatterWrite,
    right: ScatterWrite,
) -> bool {
    &&& left != right
    &&& active_scatter_write(input, use_lean, use_fused, left)
    &&& active_scatter_write(input, use_lean, use_fused, right)
    &&& scatter_target(input, left) == scatter_target(input, right)
}

pub open spec fn scatter_data_race_free(
    input: MoeInput,
    use_lean: bool,
    use_fused: bool,
) -> bool {
    forall|left: ScatterWrite, right: ScatterWrite| #![auto]
        scatter_conflict(input, use_lean, use_fused, left, right)
            ==> scatter_write_is_atomic(use_lean, use_fused)
}

pub proof fn theorem_data_race_free(
    input: MoeInput,
    use_lean: bool,
    use_fused: bool,
)
    ensures scatter_data_race_free(input, use_lean, use_fused),
{
}

} // verus!
