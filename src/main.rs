mod adynkra_derivative_intertwiners;
mod adynkra_genome;
mod adynkrafield_operator;
mod baselines;
mod bbbm;
mod bbbm_closure;
mod bbbm_component;
mod bbbm_holoraumy;
mod bbbm_nonabelian;
#[cfg(test)]
mod bbbm_nonabelian_crosscheck;
#[cfg(test)]
mod bbbm_nonabelian_source_audit;
mod bbbm_sixteen_onshell;
#[cfg(test)]
mod bbbm_sixteen_onshell_crosscheck;
#[cfg(test)]
mod bbbm_sixteen_source_audit;
#[cfg(test)]
mod bbbm_source_audit;
mod bbbm_worldline;
mod canonical;
mod central_hypermultiplet_4d;
mod checkpointable_sha256;
mod chiral_tensor_4d;
mod chiral_vector_4d;
mod chromochar;
mod chromotopology;
mod code;
mod coset_primed_lanczos;
mod dashing;
mod decompose;
mod eleven_dimensional_abstract_clifford_join;
mod eleven_dimensional_b5_majorana_target_join;
mod eleven_dimensional_bridge;
mod eleven_dimensional_clifford;
mod eleven_dimensional_covariant_cohomology_gate;
mod eleven_dimensional_direct_local_lorentz;
mod eleven_dimensional_first_superspace_jet;
mod eleven_dimensional_free_complex;
mod eleven_dimensional_gauge;
mod eleven_dimensional_hook_bianchi;
mod eleven_dimensional_j1_lorentz_residual;
mod eleven_dimensional_k_fag_solver;
mod eleven_dimensional_level16_couplings;
mod eleven_dimensional_level18_embedded;
mod eleven_dimensional_level18_momentum;
mod eleven_dimensional_level18_target_quotient;
mod eleven_dimensional_linear_susy;
mod eleven_dimensional_lorentz_holonomy_compensator_audit;
mod eleven_dimensional_majorana;
mod eleven_dimensional_physical_adapter_audit;
mod eleven_dimensional_physical_curvature;
mod eleven_dimensional_prepotential;
mod eleven_dimensional_prepotential_gate;
mod eleven_dimensional_relaxed_spinorial_cohomology;
mod eleven_dimensional_second_momentum;
mod eleven_dimensional_second_momentum_10001_fx;
mod eleven_dimensional_second_momentum_10001_maps;
mod eleven_dimensional_second_momentum_20001_fx;
mod eleven_dimensional_second_momentum_20001_maps;
mod eleven_dimensional_second_momentum_30001_fx;
mod eleven_dimensional_second_momentum_30001_maps;
mod eleven_dimensional_second_momentum_full_fx;
mod eleven_dimensional_second_momentum_full_inventory;
mod eleven_dimensional_second_momentum_full_maps;
mod eleven_dimensional_second_momentum_fx;
mod eleven_dimensional_second_momentum_gpu;
mod eleven_dimensional_second_momentum_recoupling;
mod eleven_dimensional_second_momentum_remaining_recouplings;
mod eleven_dimensional_source_fixed_curvature;
mod eleven_dimensional_spinor_bridge;
mod eleven_dimensional_spinor_bridge_kernels;
mod eleven_dimensional_spinorial_differential;
mod eleven_dimensional_target_equation_complex;
mod eleven_dimensional_target_stream;
mod eleven_dimensional_top_down;
mod enhance;
mod eval;
mod exact_component_algebra;
mod filters;
mod four_color;
#[allow(dead_code)]
mod higher_dimensional_canonical;
mod higher_dimensional_fingerprint;
mod higher_dimensional_fixture_adapters;
mod higher_dimensional_parentage;
mod holoraumy;
mod lorentz;
mod lorentz_intertwiners;
mod lr_matrix;
mod maxwell_phantom;
mod maxwell_s4_atlas_scan;
mod maxwell_s8_subalgebra_scan;
mod maxwell_worldline_search;
mod minimal_supergravity_action;
mod minimal_supergravity_curvatures;
mod nauty_canonical;
mod orientation;
mod permutahedron;
mod permutahedron_atlas;
mod permutahedron_fixtures;
mod permutahedron_garden;
mod permutahedron_hypergraph;
mod permutahedron_hypergraph_controls;
mod permutahedron_hypergraph_higher_dimensional_gate;
mod permutahedron_hypergraph_recursion_maxwell_bridge;
mod permutahedron_hypergraph_resolution;
mod permutahedron_hypergraph_signed;
mod permutahedron_hypergraph_signed_equivalence;
mod permutahedron_s4_supersymmetry;
mod permutahedron_s8_conjugate_separation;
mod permutahedron_s8_orbit_leakage;
mod permutahedron_s8_orbits;
mod permutahedron_s8_separation;
mod permutahedron_s8_signed_recursion;
mod permutahedron_s8_source_fixture_audit;
mod permutahedron_s8_spectral_identifiability;
mod permutahedron_s8_supersymmetry;
mod permutahedron_s8_unrestricted_recursion;
mod permutahedron_spectral;
mod permutahedron_spectral_cli;
mod pipeline;
mod prepotential_curvature;
mod prepotential_gauge;
mod quotient_graph_analysis;
mod r8_block_invariants;
mod ranking;
mod s8_characters;
mod scalar_tensor_tangent;
mod search;
mod second_momentum_cpu_progress;
#[cfg_attr(not(test), allow(dead_code))]
mod second_momentum_full_gpu_jobs;
#[cfg_attr(not(test), allow(dead_code))]
mod second_momentum_gpu_checkpoint;
#[cfg_attr(not(test), allow(dead_code))]
mod second_momentum_gpu_group;
#[cfg_attr(not(test), allow(dead_code))]
mod second_momentum_gpu_jobs;
mod second_momentum_gpu_multi_prime_checkpoint;
#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
mod second_momentum_gpu_progress;
#[cfg_attr(not(test), allow(dead_code))]
mod second_momentum_gpu_word_hash;
mod signed_perm;
mod spectral_lanczos;
mod sr_hole;
mod streamed_gadget;
mod supercovariant_derivative;
mod tendim_data;
mod tendim_generate;
mod vector_spinor_intertwiners;
mod vector_tensor_4d;
mod vector_tensor_central_atlas;
mod vector_tensor_central_charge;
mod vector_tensor_central_equivalence;
mod viz_export;

use std::time::Instant;

use canonical::{compute_invariants, deduplicate, is_decomposable};
use code::{DoublyEvenCode, enumerate_codes};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    match args[1].as_str() {
        "enumerate" => cmd_enumerate(&args),
        "count" => cmd_count(&args),
        "eval" => cmd_eval(&args),
        "eval-all" => cmd_eval_all(&args),
        "invariants" => cmd_invariants(&args),
        "validate" => cmd_validate(),
        "search" => cmd_search(&args),
        "saturate" => cmd_saturate(&args),
        "validate-miller" => cmd_validate_miller(&args),
        "pipeline" => cmd_pipeline(&args),
        "pipeline-k" => cmd_pipeline_k(&args),
        "decompose-k" => cmd_decompose_k(&args, false),
        "decompose-k-disk" => cmd_decompose_k(&args, true),
        "decompose-structure" => cmd_decompose_structure(&args),
        "q-scan" => cmd_q_scan(&args),
        "lift-scan" => cmd_lift_scan(&args),
        "lift-construct" => cmd_lift_construct(&args),
        "lift-search" | "lift-attack" => cmd_lift_attack(&args),
        "worldsheet-verify" => cmd_worldsheet_verify(&args),
        "central-charge" => cmd_central_charge(&args),
        "enhance-scan" => cmd_enhance_scan(&args),
        "sr-investigation" | "sr-hole" => cmd_sr_hole(&args),
        "bbbm" => cmd_bbbm(&args),
        "bbbm-closure" => cmd_bbbm_closure(&args),
        "bbbm-holoraumy" => cmd_bbbm_holoraumy(&args),
        "bbbm-nonabelian" => cmd_bbbm_nonabelian(&args),
        "bbbm-sixteen-onshell" => cmd_bbbm_sixteen_onshell(&args),
        "tendim-reproduce" => cmd_tendim_reproduce(&args),
        "tendim-generate" => cmd_tendim_generate(&args),
        "tendim-convention-scan" => cmd_tendim_convention_scan(),
        "perm-atlas-build" => cmd_perm_atlas_build(&args),
        "perm-atlas-verify" => cmd_perm_atlas_verify(),
        "perm-garden-scan" => cmd_perm_garden_scan(&args),
        "perm-hypergraph-build" => cmd_perm_hypergraph_build(&args),
        "perm-hypergraph-verify" => cmd_perm_hypergraph_verify(),
        "perm-hypergraph-controls-build" => cmd_perm_hypergraph_controls_build(&args),
        "perm-hypergraph-controls-verify" => cmd_perm_hypergraph_controls_verify(),
        "perm-hypergraph-higher-dimensional-build" => {
            cmd_perm_hypergraph_higher_dimensional_build(&args)
        }
        "perm-hypergraph-higher-dimensional-verify" => {
            cmd_perm_hypergraph_higher_dimensional_verify()
        }
        "perm-hypergraph-resolution-build" => cmd_perm_hypergraph_resolution_build(&args),
        "perm-hypergraph-resolution-verify" => cmd_perm_hypergraph_resolution_verify(),
        "perm-hypergraph-signed-build" => cmd_perm_hypergraph_signed_build(&args),
        "perm-hypergraph-signed-verify" => cmd_perm_hypergraph_signed_verify(),
        "perm-hypergraph-signed-equivalence-build" => {
            cmd_perm_hypergraph_signed_equivalence_build(&args)
        }
        "perm-hypergraph-signed-equivalence-verify" => {
            cmd_perm_hypergraph_signed_equivalence_verify()
        }
        "perm-s4-susy-build" => cmd_perm_s4_susy_build(&args),
        "perm-s4-susy-verify" => cmd_perm_s4_susy_verify(),
        "perm-s8-conjugates-build" => cmd_perm_s8_conjugates_build(&args),
        "perm-s8-conjugates-verify" => cmd_perm_s8_conjugates_verify(),
        "perm-s8-orbits-build" => cmd_perm_s8_orbits_build(&args),
        "perm-s8-orbits-verify" => cmd_perm_s8_orbits_verify(),
        "perm-s8-separation-build" => cmd_perm_s8_separation_build(&args),
        "perm-s8-separation-verify" => cmd_perm_s8_separation_verify(),
        "perm-s8-susy-build" => cmd_perm_s8_susy_build(&args),
        "perm-s8-susy-verify" => cmd_perm_s8_susy_verify(),
        "vector-tensor-central-charge-build" => cmd_vector_tensor_central_charge_build(&args),
        "vector-tensor-central-charge-verify" => cmd_vector_tensor_central_charge_verify(),
        "vector-tensor-central-equivalence-build" => {
            cmd_vector_tensor_central_equivalence_build(&args)
        }
        "vector-tensor-central-equivalence-verify" => {
            cmd_vector_tensor_central_equivalence_verify()
        }
        "vector-tensor-central-atlas-build" => cmd_vector_tensor_central_atlas_build(&args),
        "vector-tensor-central-atlas-verify" => cmd_vector_tensor_central_atlas_verify(),
        "vector-tensor-4d-build" => cmd_vector_tensor_4d_build(&args),
        "vector-tensor-4d-verify" => cmd_vector_tensor_4d_verify(),
        "central-hypermultiplet-4d-build" => cmd_central_hypermultiplet_4d_build(&args),
        "central-hypermultiplet-4d-verify" => cmd_central_hypermultiplet_4d_verify(),
        "scalar-tensor-tangent-build" => cmd_scalar_tensor_tangent_build(&args),
        "scalar-tensor-tangent-verify" => cmd_scalar_tensor_tangent_verify(),
        "chiral-vector-4d-build" => cmd_chiral_vector_4d_build(&args),
        "chiral-vector-4d-verify" => cmd_chiral_vector_4d_verify(),
        "chiral-tensor-4d-build" => cmd_chiral_tensor_4d_build(&args),
        "chiral-tensor-4d-verify" => cmd_chiral_tensor_4d_verify(),
        "higher-dimensional-fingerprint-build" => cmd_higher_dimensional_fingerprint_build(&args),
        "higher-dimensional-fingerprint-verify" => cmd_higher_dimensional_fingerprint_verify(),
        "higher-dimensional-parentage-build" => cmd_higher_dimensional_parentage_build(&args),
        "higher-dimensional-parentage-verify" => cmd_higher_dimensional_parentage_verify(),
        "higher-dimensional-parentage-query" => cmd_higher_dimensional_parentage_query(&args),
        "maxwell-phantom-build" => cmd_maxwell_phantom_build(&args),
        "maxwell-phantom-verify" => cmd_maxwell_phantom_verify(),
        "maxwell-worldline-search-build" => cmd_maxwell_worldline_search_build(&args),
        "maxwell-worldline-search-verify" => cmd_maxwell_worldline_search_verify(),
        "maxwell-s4-atlas-build" => cmd_maxwell_s4_atlas_build(&args),
        "maxwell-s4-atlas-verify" => cmd_maxwell_s4_atlas_verify(),
        "maxwell-s8-subalgebra-build" => cmd_maxwell_s8_subalgebra_build(&args),
        "maxwell-s8-subalgebra-verify" => cmd_maxwell_s8_subalgebra_verify(),
        "perm-hypergraph-recursion-maxwell-build" => {
            cmd_perm_hypergraph_recursion_maxwell_build(&args)
        }
        "perm-hypergraph-recursion-maxwell-verify" => {
            cmd_perm_hypergraph_recursion_maxwell_verify()
        }
        "perm-s8-unrestricted-recursion-build" => cmd_perm_s8_unrestricted_recursion_build(&args),
        "perm-s8-unrestricted-recursion-verify" => cmd_perm_s8_unrestricted_recursion_verify(),
        "perm-s8-orbit-leakage-build" => cmd_perm_s8_orbit_leakage_build(&args),
        "perm-s8-orbit-leakage-verify" => cmd_perm_s8_orbit_leakage_verify(),
        "perm-s8-source-fixture-audit-build" => cmd_perm_s8_source_fixture_audit_build(&args),
        "perm-s8-source-fixture-audit-verify" => cmd_perm_s8_source_fixture_audit_verify(),
        "perm-s8-spectral-identifiability-build" => {
            cmd_perm_s8_spectral_identifiability_build(&args)
        }
        "perm-s8-spectral-identifiability-verify" => cmd_perm_s8_spectral_identifiability_verify(),
        "perm-spectral-probe" => permutahedron_spectral_cli::cmd_perm_spectral_probe(&args),
        "adynkra-genome-build" => cmd_adynkra_genome_build(&args),
        "adynkra-genome-verify" => cmd_adynkra_genome_verify(),
        "adynkra-derivative-verify" => cmd_adynkra_derivative_verify(),
        "adynkra-intertwiner-verify" => cmd_adynkra_intertwiner_verify(),
        "adynkra-vector-spinor-verify" => cmd_adynkra_vector_spinor_verify(),
        "adynkra-derivative-intertwiner-verify" => cmd_adynkra_derivative_intertwiner_verify(),
        "adynkra-prepotential-gauge-verify" => cmd_adynkra_prepotential_gauge_verify(),
        "adynkra-prepotential-curvature-verify" => cmd_adynkra_prepotential_curvature_verify(),
        "adynkra-minimal-curvature-verify" => cmd_adynkra_minimal_curvature_verify(),
        "adynkra-minimal-action-verify" => cmd_adynkra_minimal_action_verify(),
        "adynkrafield-operator-verify" => cmd_adynkrafield_operator_verify(),
        "adynkra-11d-prepotential-verify" => cmd_adynkra_11d_prepotential_verify(),
        "adynkra-11d-free-complex-build" => cmd_adynkra_11d_free_complex_build(&args),
        "adynkra-11d-hook-bianchi-build" => cmd_adynkra_11d_hook_bianchi_build(&args),
        "adynkra-11d-level18-momentum-build" => cmd_adynkra_11d_level18_momentum_build(&args),
        "adynkra-11d-prepotential-gate-build" => cmd_adynkra_11d_prepotential_gate_build(&args),
        "adynkra-11d-target-stream-build" => cmd_adynkra_11d_target_stream_build(&args),
        "adynkra-11d-second-momentum-build" => cmd_adynkra_11d_second_momentum_build(&args),
        "adynkra-11d-second-momentum-recoupling-build" => {
            cmd_adynkra_11d_second_momentum_recoupling_build(&args)
        }
        "adynkra-11d-second-momentum-component-build" => {
            cmd_adynkra_11d_second_momentum_component_build(&args)
        }
        "adynkra-11d-second-momentum-10001-fx" => cmd_adynkra_11d_second_momentum_10001_fx(&args),
        "adynkra-11d-second-momentum-full-map-plan" => {
            cmd_adynkra_11d_second_momentum_full_map_plan(&args)
        }
        "adynkra-11d-second-momentum-full-map-worker" => {
            cmd_adynkra_11d_second_momentum_full_map_worker(&args)
        }
        "adynkra-11d-second-momentum-full-map-status" => {
            cmd_adynkra_11d_second_momentum_full_map_status(&args)
        }
        "adynkra-11d-second-momentum-full-gpu-plan" => {
            cmd_adynkra_11d_second_momentum_full_gpu_plan(&args)
        }
        "adynkra-11d-second-momentum-full-gpu-status" => {
            cmd_adynkra_11d_second_momentum_full_gpu_status(&args)
        }
        "adynkra-11d-second-momentum-full-gpu-worker" => {
            cmd_adynkra_11d_second_momentum_full_gpu_worker(&args)
        }
        "adynkra-11d-second-momentum-full-gpu-rank" => {
            cmd_adynkra_11d_second_momentum_full_gpu_rank(&args)
        }
        "adynkra-11d-second-momentum-gpu-rank-28" => {
            cmd_adynkra_11d_second_momentum_gpu_rank_28(&args)
        }
        "adynkra-11d-second-momentum-cpu-fx" => cmd_adynkra_11d_second_momentum_cpu_fx(&args),
        "adynkra-11d-second-momentum-gpu-fx" => cmd_adynkra_11d_second_momentum_gpu_fx(&args),
        "adynkra-11d-second-momentum-gpu-fx-plan" => {
            cmd_adynkra_11d_second_momentum_gpu_fx_plan(&args)
        }
        "adynkra-11d-second-momentum-gpu-fx-worker" => {
            cmd_adynkra_11d_second_momentum_gpu_fx_worker(&args)
        }
        "adynkra-11d-second-momentum-gpu-fx-multi-prime-worker" => {
            cmd_adynkra_11d_second_momentum_gpu_fx_multi_prime_worker(&args)
        }
        "adynkra-11d-second-momentum-gpu-fx-status" => {
            cmd_adynkra_11d_second_momentum_gpu_fx_status(&args)
        }
        "adynkra-11d-second-momentum-gpu-fx-import" => {
            cmd_adynkra_11d_second_momentum_gpu_fx_import(&args)
        }
        "adynkra-11d-second-momentum-gpu-status-reconcile" => {
            cmd_adynkra_11d_second_momentum_gpu_status_reconcile(&args)
        }
        "adynkra-11d-top-down-build" => cmd_adynkra_11d_top_down_build(&args),
        "adynkra-11d-first-momentum-fx-aggregate" => {
            cmd_adynkra_11d_first_momentum_fx_aggregate(&args)
        }
        "adynkra-11d-clifford-verify" => cmd_adynkra_11d_clifford_verify(),
        "adynkra-11d-gauge-intertwiner-verify" => cmd_adynkra_11d_gauge_intertwiner_verify(),
        "adynkra-11d-gauge-composition-manifest" => cmd_adynkra_11d_gauge_composition_manifest(),
        "adynkra-11d-gauge-zero-column" => cmd_adynkra_11d_gauge_zero_column(&args),
        "adynkra-11d-gauge-zero-merge" => cmd_adynkra_11d_gauge_zero_merge(&args),
        "adynkra-11d-gauge-zero-classify" => cmd_adynkra_11d_gauge_zero_classify(&args),
        "adynkra-11d-gauge-first-functional" => cmd_adynkra_11d_gauge_first_functional(&args),
        "adynkra-11d-gauge-first-functional-stream" => {
            cmd_adynkra_11d_gauge_first_functional_stream(&args)
        }
        "adynkra-11d-gauge-first-functional-stream-prefix" => {
            cmd_adynkra_11d_gauge_first_functional_stream_prefix(&args)
        }
        "adynkra-11d-gauge-first-functional-merge" => {
            cmd_adynkra_11d_gauge_first_functional_merge(&args)
        }
        "adynkra-11d-bridge-verify" => cmd_adynkra_11d_bridge_verify(),
        "adynkra-11d-level16-coupling-precheck" => cmd_adynkra_11d_level16_coupling_precheck(),
        "adynkra-11d-level16-coupling-build" => cmd_adynkra_11d_level16_coupling_build(&args),
        "adynkra-11d-level16-coupling-verify" => cmd_adynkra_11d_level16_coupling_verify(&args),
        "adynkra-11d-level17-hook-precheck" => cmd_adynkra_11d_level17_hook_precheck(),
        "adynkra-11d-level17-hook-build" => cmd_adynkra_11d_level17_hook_build(&args),
        "adynkra-11d-level17-hook-verify" => cmd_adynkra_11d_level17_hook_verify(&args),
        "adynkra-11d-level17-derivative-matrix" => cmd_adynkra_11d_level17_derivative_matrix(),
        "adynkra-11d-first-momentum-precheck" => cmd_adynkra_11d_first_momentum_precheck(),
        "adynkra-11d-first-momentum-kernel-verify" => {
            cmd_adynkra_11d_first_momentum_kernel_verify()
        }
        "adynkra-11d-first-momentum-coupling-build" => {
            cmd_adynkra_11d_first_momentum_coupling_build(&args)
        }
        "adynkra-11d-first-momentum-coupling-verify" => {
            cmd_adynkra_11d_first_momentum_coupling_verify(&args)
        }
        "adynkra-11d-first-momentum-target-verify" => {
            cmd_adynkra_11d_first_momentum_target_verify()
        }
        "adynkra-11d-joint-compatibility" => cmd_adynkra_11d_joint_compatibility(),
        "adynkra-11d-joint-column" => cmd_adynkra_11d_joint_column(&args),
        "adynkra-11d-joint-merge" => cmd_adynkra_11d_joint_merge(&args),
        "adynkra-11d-joint-manifest" => cmd_adynkra_11d_joint_manifest(),
        "adynkra-11d-spinor-bridge-verify" => cmd_adynkra_11d_spinor_bridge_verify(),
        "adynkra-11d-spinor-kernel-verify" => cmd_adynkra_11d_spinor_kernel_verify(),
        "export-3d-assets" => cmd_export_3d_assets(&args),
        "decompose-audit" => cmd_decompose_audit(&args),
        "decompose-probe" => cmd_decompose_probe(&args),
        "cls-g-full-build" => cmd_cls_g_full_build(&args),
        "cls-g-full-verify" => cmd_cls_g_full_verify(&args),
        "cls-g-csp-build" => cmd_cls_g_csp_build(&args),
        "cls-g-csp-shard" => cmd_cls_g_csp_shard(&args),
        "cls-g-csp-status" => cmd_cls_g_csp_status(&args),
        "cls-g-csp-merge" => cmd_cls_g_csp_merge(&args),
        "help" | "--help" | "-h" => print_usage(&args[0]),
        other => {
            eprintln!("Unknown command: {}", other);
            print_usage(&args[0]);
            std::process::exit(1);
        }
    }
}

fn print_usage(prog: &str) {
    eprintln!("Usage: {} <command> [args]", prog);
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  enumerate <n>           Enumerate and print all doubly-even codes of length n");
    eprintln!("  count <max_n>           Count equivalence classes for n=1 to max_n");
    eprintln!("  eval <held_out_n>       Run leave-one-N-out evaluation for a specific N");
    eprintln!("  eval-all [max_cand]     Run evaluation for all N from 4 to 10");
    eprintln!("  invariants <n>          Print invariants for all codes of length n");
    eprintln!("  validate                Self-test: enumerate N=4..8, verify known results");
    eprintln!("  search [n] [pop] [gen]  Search for doubly-even codes at N (default 16)");
    eprintln!("  saturate [n] [batch_size] [max_batches]");
    eprintln!("                          Saturation test at N (defaults: 16, 5000, 500)");
    eprintln!("  validate-miller [n]     Compare counts against Miller/Doran-Faux-Gates");
    eprintln!("                          reference (available: N=4, N=8, N=12, N=16)");
    eprintln!("  pipeline <json>         Run the full dimensional lifting pipeline");
    eprintln!("  pipeline-k <k> [json]   Run pipeline for a single k-stratum only");
    eprintln!("  lift-scan <k> [json]    Worldsheet (p,q) lift-scan over a single k-stratum;");
    eprintln!("                          emits a re-checkable JSON witness per code class.");
    eprintln!("                          Budget env vars (hangings are SAMPLED):");
    eprintln!("                            ADINKRA_LIFT_CHAINS   source-raise chains (default 32)");
    eprintln!(
        "                            ADINKRA_LIFT_MAXRANK  max hangings/code   (default 512)"
    );
    eprintln!(
        "                          Low-k strata (k<=4) need a larger budget to reach 145/145:"
    );
    eprintln!("                            ADINKRA_LIFT_CHAINS=128 ADINKRA_LIFT_MAXRANK=8000");
    eprintln!("  worldsheet-verify [catalog] [certificate]");
    eprintln!("                          Verify the retained 145-class spin-sum witnesses");
    eprintln!("  decompose-k <k> [json]  Irreducible-decompose a single k-stratum (F8 route b)");
    eprintln!("                          and compute the gadget on irreducible pieces");
    eprintln!("  decompose-k-disk <k> [json] [--f64]");
    eprintln!("                          like decompose-k but spills W to a disk scratch file");
    eprintln!(
        "                          (--f64 = exact f64 store + Gram; trustworthy value count)"
    );
    eprintln!("  decompose-structure <k> [json]");
    eprintln!("                          basis-invariant Schur structure (commutant only, no eig;");
    eprintln!("                          scales to k<=3 where the dense path cannot reach)");
    eprintln!("  decompose-audit <k> <sample_reps> [json]");
    eprintln!("                          f32 error audit: dense f64 vs GEMM f64 vs GEMM f32");
    eprintln!("  cls-g-full-build [side] [blocks] [threads] [cap] [json]");
    eprintln!("                          Full {{-1,0,1}} CLS G-matrix enumeration via the");
    eprintln!(
        "                          K=Q(sqrt(-3)) commutant reduction (side L|R, blocks 1..3)"
    );
    eprintln!("  cls-g-full-verify [side] [blocks] [json]");
    eprintln!(
        "                          Re-verify the ladder and every stored sample of an artifact"
    );
    eprintln!("  cls-g-csp-build [side] [blocks] [threads] [cap] [stride] [json]");
    eprintln!("                          v2 slot-level CSP engine: correlated product boxes,");
    eprintln!(
        "                          residual arc-consistency, MRV order (stride>1 = stratified sample)"
    );
    eprintln!("  cls-g-csp-shard [side] [blocks] [start] [count] [threads] [dir] [stride]");
    eprintln!("                          Durable sharded enumeration: one immutable shard per");
    eprintln!(
        "                          slot-0 prefix, atomic writes, skips existing valid shards"
    );
    eprintln!("  cls-g-csp-status [dir]");
    eprintln!("                          Live dashboard over a shard dir: coverage, checksum,");
    eprintln!("                          per-pod heartbeats, per-worker progress, ETA (read-only)");
    eprintln!("  cls-g-csp-merge [side] [blocks] [dir] [json]");
    eprintln!("                          Verify full shard coverage and merge commutatively;");
    eprintln!("                          refuses an incomplete or inconsistent census");
    eprintln!("  sr-investigation [json] Analyze the minimal unpaired Siegel-Rocek case");
    eprintln!("                          (default: adinkra_codes_n16.json)");
    eprintln!("  bbbm                    Verify the generic minimal N=9 valise scaffold");
    eprintln!("  bbbm-holoraumy          Compute the generic N=9 gadget invariants");
    eprintln!("  bbbm-closure            Verify BBBM component closure and worldline reduction");
    eprintln!("  bbbm-nonabelian         Verify the full nonabelian BBBM component algebra");
    eprintln!("  bbbm-sixteen-onshell    Verify full 16-charge closure modulo the Dirac equation");
    eprintln!("  tendim-reproduce [json] Audit the pinned 10D supergravity L/R reproduction");
    eprintln!("  tendim-generate [json]  Generate the 10D supergravity L/R artifact in Rust");
    eprintln!("  tendim-convention-scan  Compare the 1/16 and 1/8 formula branches");
    eprintln!("  perm-atlas-build [dir] [report]");
    eprintln!("                          Build complete S4 and S8 permutahedron atlases");
    eprintln!("  perm-atlas-verify       Verify graphs, paper correlators, cosets, and embeddings");
    eprintln!("  perm-garden-scan [json]");
    eprintln!("                          Solve Garden signs for all 5,040 R8 cosets");
    eprintln!("  perm-hypergraph-build [data-json] [validation-json]");
    eprintln!("                          Discover exact S4/S8 unsigned constraint hypergraphs");
    eprintln!("  perm-hypergraph-verify Verify the unlabeled clique and incidence calculation");
    eprintln!("  perm-hypergraph-controls-build [data-json] [validation-json]");
    eprintln!("                          Project published controls onto hypergraph families");
    eprintln!("  perm-hypergraph-controls-verify Verify control membership and closure");
    eprintln!("  perm-hypergraph-higher-dimensional-build [data-json] [validation-json]");
    eprintln!("                          Run the sourced CV/CT control gate and stop audit");
    eprintln!("  perm-hypergraph-higher-dimensional-verify Verify the bounded control gate");
    eprintln!("  perm-hypergraph-resolution-build [data-json] [validation-json]");
    eprintln!("                          Find and certify a minimum mixed-cover trade");
    eprintln!("  perm-hypergraph-resolution-verify Verify the exact mixed-cover certificate");
    eprintln!("  perm-hypergraph-signed-build [data-json] [validation-json]");
    eprintln!("                          Transport Garden signs across all 151,200 octets");
    eprintln!("  perm-hypergraph-signed-verify Verify all signed transports and affine ranks");
    eprintln!("  perm-hypergraph-signed-equivalence-build [data-json] [validation-json]");
    eprintln!("                          Classify the 30 signed identity representatives");
    eprintln!("  perm-hypergraph-signed-equivalence-verify Verify every ledger witness");
    eprintln!("  perm-s4-susy-build [data-json] [validation-json]");
    eprintln!("                          Build the six signed S4 sectors and their Adinkras");
    eprintln!("  perm-s4-susy-verify     Verify all 96 published fiducial signings");
    eprintln!("  perm-s8-conjugates-build [data-json] [validation-json]");
    eprintln!("                          Scan all 30 conjugate R8 coset families");
    eprintln!("  perm-s8-conjugates-verify");
    eprintln!("                          Verify all 151,200 unsigned GR(8,8) supports");
    eprintln!("  perm-s8-orbits-build [data-json] [validation-json]");
    eprintln!("                          Classify one R8 coset atlas under its normalizer");
    eprintln!("  perm-s8-orbits-verify   Verify the 20 exact normalizer orbits");
    eprintln!("  perm-s8-separation-build [data-json] [validation-json]");
    eprintln!("                          Classify R8 cosets that split into paired S4 sectors");
    eprintln!("  perm-s8-separation-verify");
    eprintln!("                          Verify all invariant 4+4 splits and pair classes");
    eprintln!("  perm-s8-susy-build [data-json] [validation-json]");
    eprintln!("                          Build the six published signed S8 representations");
    eprintln!("  perm-s8-susy-verify     Verify closure, nonclosure, HYMN, and all m/n branches");
    eprintln!("  vector-tensor-central-charge-build [data-json] [validation-json]");
    eprintln!("                          Factor S8 residuals and certify central extensions");
    eprintln!("  vector-tensor-central-charge-verify Verify the one-Z vector-tensor completion");
    eprintln!("  vector-tensor-central-equivalence-build [json]");
    eprintln!("                          Classify all 25 printed one-Z branches");
    eprintln!("  vector-tensor-central-equivalence-verify Verify every enriched witness");
    eprintln!("  vector-tensor-central-atlas-build [json]");
    eprintln!("                          Transport one-Z closure to all 151,200 supports");
    eprintln!("  vector-tensor-central-atlas-verify Verify complete one-Z support coverage");
    eprintln!("  vector-tensor-4d-build [data-json] [validation-json]");
    eprintln!("                          Build the corrected Eq. 78 component fixture");
    eprintln!("  vector-tensor-4d-verify Verify the corrected Eq. 78 component closure");
    eprintln!("  central-hypermultiplet-4d-build [data-json] [validation-json]");
    eprintln!("                          Build the exact Wess-Fayet one-Z holdout");
    eprintln!("  central-hypermultiplet-4d-verify Verify its 4D closure and CC bridge");
    eprintln!("  scalar-tensor-tangent-build [json]");
    eprintln!("                          Derive the regular rigid tangent preflight");
    eprintln!("  scalar-tensor-tangent-verify Verify composites, gauges, and 8+8 count");
    eprintln!("  chiral-vector-4d-build [data-json] [validation-json]");
    eprintln!("                          Reproduce 4D chiral-vector closure and reduction");
    eprintln!("  chiral-vector-4d-verify Verify Eqs. 32-41 of arXiv:1405.0048 exactly");
    eprintln!("  chiral-tensor-4d-build [data-json] [validation-json]");
    eprintln!("                          Reproduce 4D chiral-tensor closure and reduction");
    eprintln!("  chiral-tensor-4d-verify Verify Eqs. 44-53 of arXiv:1405.0048 exactly");
    eprintln!("  higher-dimensional-fingerprint-build [json]");
    eprintln!("                          Compare CV and CT spatial and gauge data");
    eprintln!("  higher-dimensional-fingerprint-verify Verify the CV/CT comparison gates");
    eprintln!("  higher-dimensional-parentage-build [json]");
    eprintln!("                          Build the invariant catalog and inference audit");
    eprintln!("  higher-dimensional-parentage-verify Verify classification and mutations");
    eprintln!("  higher-dimensional-parentage-query <query-json>");
    eprintln!("                          Classify supplied physical decorations");
    eprintln!("  maxwell-phantom-build [json]");
    eprintln!("                          Verify Maxwell phantom and Bianchi linkage data");
    eprintln!("  maxwell-phantom-verify Verify the complete Maxwell Eq. 5.11 gate");
    eprintln!("  maxwell-worldline-search-build [json]");
    eprintln!("                          Recover Maxwell from four-color worldline data");
    eprintln!("  maxwell-worldline-search-verify Verify recovery and negative controls");
    eprintln!("  maxwell-s4-atlas-build [json]");
    eprintln!("                          Scan all 96 published four-color signings");
    eprintln!("  maxwell-s4-atlas-verify Verify the complete four-color scan");
    eprintln!("  maxwell-s8-subalgebra-build [json]");
    eprintln!("                          Scan both embedded four-color blocks of S8 closers");
    eprintln!("  maxwell-s8-subalgebra-verify Verify the embedded-block classification");
    eprintln!("  perm-hypergraph-recursion-maxwell-build [json]");
    eprintln!("                          Map S8 recursion closers and Maxwell classes");
    eprintln!("  perm-hypergraph-recursion-maxwell-verify Verify the exact bridge");
    eprintln!("  perm-s8-unrestricted-recursion-build [json]");
    eprintln!("                          Exhaust all 256 masks and same-source controls");
    eprintln!("  perm-s8-unrestricted-recursion-verify Verify the unrestricted census");
    eprintln!("  perm-s8-orbit-leakage-build [json]");
    eprintln!("                          Audit normalizer-orbit basis dependence");
    eprintln!("  perm-s8-orbit-leakage-verify Verify node-relabeling reachability");
    eprintln!("  perm-s8-source-fixture-audit-build [json]");
    eprintln!("                          Classify physical controls by source provenance");
    eprintln!("  perm-s8-source-fixture-audit-verify Verify source eligibility and stop gate");
    eprintln!("  perm-s8-spectral-identifiability-build [json]");
    eprintln!("                          Audit all 30 equitable R8 partitions");
    eprintln!("  perm-s8-spectral-identifiability-verify Verify the spectral no-go result");
    eprintln!("  adynkra-genome-build [data-json] [validation-json]");
    eprintln!("                          Build the six published 4D N=1 Adynkra genomes");
    eprintln!("  adynkra-genome-verify Verify Eqs. 3.6-3.11 term by term");
    eprintln!("  adynkra-derivative-verify Verify the 4D N=1 derivative algebra in Eq. 2.22");
    eprintln!("  adynkra-intertwiner-verify Verify the rank-two projectors in Eqs. 2.5 and 2.18");
    eprintln!(
        "  adynkra-vector-spinor-verify Verify the vector-spinor projectors in Eqs. 2.13-2.19"
    );
    eprintln!(
        "  adynkra-derivative-intertwiner-verify Verify fundamental CG and repeated-irrep maps"
    );
    eprintln!("  adynkra-prepotential-gauge-verify Verify the supergravity prepotential gauge map");
    eprintln!("  adynkra-prepotential-curvature-verify Verify the chiral super-Weyl curvature");
    eprintln!("  adynkra-minimal-curvature-verify Verify the old-minimal curvature complex");
    eprintln!("  adynkra-minimal-action-verify Verify the quadratic old-minimal action");
    eprintln!("  adynkrafield-operator-verify Verify the old-minimal Adynkrafield operator");
    eprintln!(
        "  adynkra-11d-prepotential-verify Verify the 11D prepotential-candidate inventories"
    );
    eprintln!("  adynkra-11d-top-down-build [json] Build the bounded top-down gate report");
    eprintln!("  adynkra-11d-first-momentum-fx-aggregate [json] [checkpoint-root]");
    eprintln!("                          Merge 336 complete F_X checkpoints without recomputing");
    eprintln!("  adynkra-11d-free-complex-build [data-json] [validation-json]");
    eprintln!(
        "                          Build the exact complexified 44+84|128 free target complex"
    );
    eprintln!("  adynkra-11d-hook-bianchi-build [data-json] [validation-json]");
    eprintln!("                          Build the bounded level-17 hook continuation gate");
    eprintln!("  adynkra-11d-level18-momentum-build [data-json] [validation-json]");
    eprintln!("                          Build exact level-18 lifts and momentum source audit");
    eprintln!("  adynkra-11d-prepotential-gate-build [json]");
    eprintln!("                          Build the exact 336-job source-side kill-gate work list");
    eprintln!("  adynkra-11d-target-stream-build [data-json] [validation-json]");
    eprintln!("                          Build the target-resolved exact 11 by 32 stream");
    eprintln!("  adynkra-11d-second-momentum-build [json]");
    eprintln!("                          Build the bounded p^2 D^12 inventory and stream report");
    eprintln!("  adynkra-11d-second-momentum-recoupling-build [json]");
    eprintln!("                          Certify the trace/STT rank-two momentum recoupling");
    eprintln!("  adynkra-11d-second-momentum-component-build [results-dir]");
    eprintln!("                          Build bounded 10001/30001 component-map certificates");
    eprintln!("  adynkra-11d-second-momentum-10001-fx [json]");
    eprintln!("                          Publish the exact four-column original 10001 slice");
    eprintln!("  adynkra-11d-second-momentum-full-map-plan");
    eprintln!("                          Print the canonical portable 47-job map inventory");
    eprintln!(
        "  adynkra-11d-second-momentum-full-map-worker <job-list> [checkpoint-dir] [status-file]"
    );
    eprintln!(
        "                          Build a portable map list such as 0-3,8 with JSONL progress"
    );
    eprintln!("  adynkra-11d-second-momentum-full-map-status [checkpoint-dir]");
    eprintln!("                          Validate durable missing-map coverage and proof gates");
    eprintln!("  adynkra-11d-second-momentum-full-gpu-plan [output-dir]");
    eprintln!(
        "                          Print the 96-job manifest for all 53 non-large-tranche columns"
    );
    eprintln!("  adynkra-11d-second-momentum-full-gpu-status <job-list> [output-dir]");
    eprintln!("                          Summarize portable full-inventory GPU assignments");
    eprintln!(
        "  adynkra-11d-second-momentum-full-gpu-worker <job-list> <map-dir> [output-dir] [device] [cpu-parity-terms]"
    );
    eprintln!("                          Run resumable missing-column groups with 5-second status");
    eprintln!(
        "  adynkra-11d-second-momentum-full-gpu-rank <prime> <output-json> <artifact-dir> [artifact-dir ...]"
    );
    eprintln!("                          Verify and rank all 77 same-prime modular columns");
    eprintln!("  adynkra-11d-second-momentum-cpu-fx <20001|30001> [output-file] [status-file]");
    eprintln!(
        "  adynkra-11d-second-momentum-gpu-fx <20001|30001> <local-column> [prime] [output-dir] [cpu-parity-terms] [device] [status-file]"
    );
    eprintln!("                          Run one exact GPU F_X column with JSONL/status telemetry");
    eprintln!("  adynkra-11d-second-momentum-gpu-fx-plan [output-dir]");
    eprintln!(
        "                          Print or publish the canonical 36-job, three-prime inventory"
    );
    eprintln!(
        "  adynkra-11d-second-momentum-gpu-fx-worker <job-list> [output-dir] [device] [cpu-parity-terms]"
    );
    eprintln!("                          Run a portable list such as 20001@0 or 30001-g7-p0");
    eprintln!(
        "  adynkra-11d-second-momentum-gpu-fx-multi-prime-worker <same-group-job-list> [output-dir] [device] [cpu-parity-terms]"
    );
    eprintln!("                          Traverse exact sources once for 2-3 prime jobs");
    eprintln!("  adynkra-11d-second-momentum-gpu-fx-status <job-list> [output-dir]");
    eprintln!("                          Validate coverage and live/stale worker status");
    eprintln!(
        "  adynkra-11d-second-momentum-gpu-fx-import <job-list> <source-dir> [destination-dir]"
    );
    eprintln!("                          Verify and import portable completed jobs");
    eprintln!(
        "  adynkra-11d-second-momentum-gpu-status-reconcile <status-file> <child-pid> <exit:N|signal:N|unknown>"
    );
    eprintln!("                          Reconcile a status snapshot after waiting for its child");
    eprintln!("  adynkra-11d-clifford-verify Verify the 11D Clifford and vector-spinor projectors");
    eprintln!(
        "  adynkra-11d-gauge-intertwiner-verify Construct the six candidate 11D spinor gauge maps"
    );
    eprintln!(
        "  adynkra-11d-gauge-composition-manifest Print the deterministic 336-job gauge work list"
    );
    eprintln!("  adynkra-11d-bridge-verify Verify the 11D bridge and first lower symbol");
    eprintln!(
        "  adynkra-11d-level16-coupling-precheck Verify the fixed level-16 work list and multiplicities"
    );
    eprintln!(
        "  adynkra-11d-level16-coupling-build --label LABEL Build one exact abstract coupling"
    );
    eprintln!(
        "  adynkra-11d-level16-coupling-verify --label LABEL --copy N Verify one embedded coupling"
    );
    eprintln!(
        "  adynkra-11d-level16-coupling-verify --all [--resume] Verify all 12 embedded couplings"
    );
    eprintln!("  adynkra-11d-level17-hook-precheck Verify the seven-copy hook manifest");
    eprintln!("  adynkra-11d-level17-hook-build --label LABEL Build one hook coupling");
    eprintln!("  adynkra-11d-level17-hook-verify --all [--resume] Verify all seven hook couplings");
    eprintln!("  adynkra-11d-level17-derivative-matrix Build the exact 7-by-12 derivative matrix");
    eprintln!("  adynkra-11d-first-momentum-precheck Emit the 44-map level-14 work list");
    eprintln!("  adynkra-11d-first-momentum-kernel-verify Verify all 28 level-14 source kernels");
    eprintln!("  adynkra-11d-first-momentum-coupling-build --source LABEL --target LABEL");
    eprintln!("  adynkra-11d-first-momentum-coupling-verify --all [--resume] Verify all 44 maps");
    eprintln!(
        "  adynkra-11d-first-momentum-target-verify Verify the four momentum target couplings"
    );
    eprintln!(
        "  adynkra-11d-joint-compatibility Build the exact leading plus first-momentum matrix"
    );
    eprintln!(
        "  adynkra-11d-joint-column <ordinal> <root> Build one durable raw joint column artifact"
    );
    eprintln!("  adynkra-11d-joint-merge <root> [--deep] Verify and merge all 56 column artifacts");
    eprintln!("  adynkra-11d-joint-manifest Print the deterministic 56-column work manifest");
    eprintln!(
        "  adynkra-11d-gauge-zero-column <form-degree> <leading-ordinal> <root> Build one durable D17 source-variation artifact"
    );
    eprintln!(
        "  adynkra-11d-gauge-zero-merge <form-degree> <root> [--deep] Compute the exact 12-column source-invariant kernel"
    );
    eprintln!(
        "  adynkra-11d-gauge-zero-classify <root> Classify all 64 exact zero-momentum source-channel intersections"
    );
    eprintln!(
        "  adynkra-11d-gauge-first-functional <form-degree> <operator-ordinal> <root> Build one exact first-momentum functional column"
    );
    eprintln!(
        "  adynkra-11d-gauge-first-functional-stream <form-degree> <operator-ordinal> <root> Build the same exact functional without materializing residual coordinates"
    );
    eprintln!(
        "  adynkra-11d-gauge-first-functional-stream-prefix <form-degree> <operator-ordinal> <parameter-component-count> <root> Build an exact exclusion screen on a parameter-component prefix"
    );
    eprintln!(
        "  adynkra-11d-gauge-first-functional-merge <form-degree> <root> <zero-root> Screen the zero-momentum kernel against 44 corrections"
    );
    eprintln!("  adynkra-11d-spinor-bridge-verify Audit the direct 11D spinor bridge");
    eprintln!("  adynkra-11d-spinor-kernel-verify Verify its 19 source kernels exactly");
    eprintln!("  export-3d-assets [json] [output-dir]");
    eprintln!("                          Export catalog-wide 3D dashing assets");
    eprintln!("  help                    Print this help message");
}

fn cmd_sr_hole(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("adinkra_codes_n16.json");
    let report = sr_hole::run(path);
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn cmd_bbbm(_args: &[String]) {
    let report = bbbm::run();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn cmd_bbbm_holoraumy(_args: &[String]) {
    let report = bbbm_holoraumy::compute();
    println!("{:#?}", report);
}

fn cmd_bbbm_closure(_args: &[String]) {
    let report = bbbm_closure::run();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn cmd_bbbm_nonabelian(_args: &[String]) {
    let report = bbbm_nonabelian::run();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn cmd_bbbm_sixteen_onshell(_args: &[String]) {
    let report = bbbm_sixteen_onshell::run();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn cmd_tendim_reproduce(args: &[String]) {
    let default_path = format!("{}/data/tendim_10d_lr.json", env!("CARGO_MANIFEST_DIR"));
    let path = args.get(2).map(String::as_str).unwrap_or(&default_path);
    let report = tendim_data::reproduction_audit(path);
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_tendim_generate(args: &[String]) {
    let default_path = format!("{}/data/tendim_10d_lr.json", env!("CARGO_MANIFEST_DIR"));
    let path = args.get(2).map(String::as_str).unwrap_or(&default_path);
    let generated = tendim_generate::generate();
    let exact_pairs = tendim_generate::verify_exact_bosonic(&generated);
    let artifact = tendim_generate::artifact_json(&generated);
    std::fs::write(path, artifact).unwrap_or_else(|e| panic!("failed to write {path}: {e}"));
    println!(
        "{{\"output\":{},\"language\":\"Rust\",\"exact_bosonic_pairs\":{},\"python_role\":\"independent cross-check only\"}}",
        serde_json::to_string(path).unwrap(),
        exact_pairs
    );
}

fn cmd_tendim_convention_scan() {
    let report = tendim_generate::convention_scan();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn cmd_perm_atlas_build(args: &[String]) {
    let output = args.get(2).map(String::as_str).unwrap_or("data");
    let report_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_validation.json");
    let report = permutahedron_atlas::build_artifacts(
        std::path::Path::new(output),
        std::path::Path::new(report_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn cmd_perm_atlas_verify() {
    let report = permutahedron_atlas::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_garden_scan(args: &[String]) {
    let output = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/permutahedron_s8_garden.json");
    let report = permutahedron_garden::write_complete_garden_scan(std::path::Path::new(output));
    println!(
        "{{\"output\":{},\"cosets_scanned\":{},\"signable_cosets\":{},\"abnormal_cosets\":{},\"normalizer_order\":{},\"passed\":{}}}",
        serde_json::to_string(output).unwrap(),
        report.cosets_scanned,
        report.signable_cosets,
        report.contingency.abnormal_and_signable,
        report.normalizer.normalizer_order,
        report.passed
    );
}

fn cmd_perm_hypergraph_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/permutahedron_constraint_hypergraphs.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_constraint_hypergraphs_validation.json");
    let report = permutahedron_hypergraph::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_hypergraph_verify() {
    let artifact = permutahedron_hypergraph::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.findings).unwrap()
    );
    if !artifact.validation.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_hypergraph_controls_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/permutahedron_hypergraph_physical_controls.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_hypergraph_physical_controls_validation.json");
    let report = permutahedron_hypergraph_controls::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_hypergraph_controls_verify() {
    let artifact = permutahedron_hypergraph_controls::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.findings).unwrap()
    );
    if !artifact.validation.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_hypergraph_higher_dimensional_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/permutahedron_hypergraph_higher_dimensional_gate.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_hypergraph_higher_dimensional_gate_validation.json");
    let report = permutahedron_hypergraph_higher_dimensional_gate::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.audit_passed {
        std::process::exit(2);
    }
}

fn cmd_perm_hypergraph_higher_dimensional_verify() {
    let artifact = permutahedron_hypergraph_higher_dimensional_gate::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.findings).unwrap()
    );
    if !artifact.validation.audit_passed {
        std::process::exit(2);
    }
}

fn cmd_chiral_vector_4d_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/chiral_vector_4d.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/chiral_vector_4d_validation.json");
    let report = chiral_vector_4d::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_chiral_vector_4d_verify() {
    let report = chiral_vector_4d::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_chiral_tensor_4d_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/chiral_tensor_4d.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/chiral_tensor_4d_validation.json");
    let report = chiral_tensor_4d::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_chiral_tensor_4d_verify() {
    let report = chiral_tensor_4d::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_higher_dimensional_fingerprint_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/higher_dimensional_fingerprint.json");
    let artifact = higher_dimensional_fingerprint::write_artifact(std::path::Path::new(path));
    println!("{}", serde_json::to_string_pretty(&artifact).unwrap());
    if !artifact.passed {
        std::process::exit(2);
    }
}

fn cmd_higher_dimensional_fingerprint_verify() {
    let artifact = higher_dimensional_fingerprint::build();
    println!("{}", serde_json::to_string_pretty(&artifact).unwrap());
    if !artifact.passed {
        std::process::exit(2);
    }
}

fn cmd_higher_dimensional_parentage_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/higher_dimensional_parentage.json");
    let artifact = higher_dimensional_parentage::write_artifact(std::path::Path::new(path));
    println!("{}", serde_json::to_string_pretty(&artifact).unwrap());
    if !artifact.passed {
        std::process::exit(2);
    }
}

fn cmd_higher_dimensional_parentage_verify() {
    let artifact = higher_dimensional_parentage::build();
    println!("{}", serde_json::to_string_pretty(&artifact).unwrap());
    if !artifact.passed {
        std::process::exit(2);
    }
}

fn cmd_higher_dimensional_parentage_query(args: &[String]) {
    let Some(path) = args.get(2) else {
        eprintln!("higher-dimensional-parentage-query requires a query JSON path");
        std::process::exit(1);
    };
    let reader = std::io::BufReader::new(
        std::fs::File::open(path).expect("open higher-dimensional parentage query"),
    );
    let query: higher_dimensional_parentage::ParentageQuery =
        serde_json::from_reader(reader).expect("parse higher-dimensional parentage query");
    let result =
        higher_dimensional_parentage::infer(&query, &higher_dimensional_parentage::known_catalog());
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}

fn cmd_maxwell_phantom_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/maxwell_phantom.json");
    let artifact = maxwell_phantom::write_artifact(std::path::Path::new(path));
    println!("{}", serde_json::to_string_pretty(&artifact).unwrap());
    if !artifact.passed {
        std::process::exit(2);
    }
}

fn cmd_maxwell_phantom_verify() {
    let artifact = maxwell_phantom::build();
    println!("{}", serde_json::to_string_pretty(&artifact).unwrap());
    if !artifact.passed {
        std::process::exit(2);
    }
}

fn cmd_maxwell_worldline_search_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/maxwell_worldline_search.json");
    let artifact = maxwell_worldline_search::write_artifact(std::path::Path::new(path));
    println!("{}", serde_json::to_string_pretty(&artifact).unwrap());
    if !artifact.passed {
        std::process::exit(2);
    }
}

fn cmd_maxwell_worldline_search_verify() {
    let artifact = maxwell_worldline_search::build();
    println!("{}", serde_json::to_string_pretty(&artifact).unwrap());
    if !artifact.passed {
        std::process::exit(2);
    }
}

fn cmd_maxwell_s4_atlas_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/maxwell_s4_atlas_scan.json");
    let artifact = maxwell_s4_atlas_scan::write_artifact(std::path::Path::new(path));
    println!("{}", serde_json::to_string_pretty(&artifact).unwrap());
    if !artifact.passed {
        std::process::exit(2);
    }
}

fn cmd_maxwell_s4_atlas_verify() {
    let artifact = maxwell_s4_atlas_scan::build();
    println!("{}", serde_json::to_string_pretty(&artifact).unwrap());
    if !artifact.passed {
        std::process::exit(2);
    }
}

fn cmd_maxwell_s8_subalgebra_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/maxwell_s8_subalgebra_scan.json");
    let artifact = maxwell_s8_subalgebra_scan::write_artifact(std::path::Path::new(path));
    println!("{}", serde_json::to_string_pretty(&artifact).unwrap());
    if !artifact.passed {
        std::process::exit(2);
    }
}

fn cmd_maxwell_s8_subalgebra_verify() {
    let artifact = maxwell_s8_subalgebra_scan::build();
    println!("{}", serde_json::to_string_pretty(&artifact).unwrap());
    if !artifact.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_hypergraph_recursion_maxwell_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_hypergraph_recursion_maxwell_bridge.json");
    let validation = permutahedron_hypergraph_recursion_maxwell_bridge::write_artifact(
        std::path::Path::new(path),
    );
    println!("{}", serde_json::to_string_pretty(&validation).unwrap());
    if !validation.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_hypergraph_recursion_maxwell_verify() {
    let artifact = permutahedron_hypergraph_recursion_maxwell_bridge::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    if !artifact.validation.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_unrestricted_recursion_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_s8_unrestricted_recursion.json");
    let validation =
        permutahedron_s8_unrestricted_recursion::write_artifact(std::path::Path::new(path));
    println!("{}", serde_json::to_string_pretty(&validation).unwrap());
    if !validation.audit_passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_unrestricted_recursion_verify() {
    let artifact = permutahedron_s8_unrestricted_recursion::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    if !artifact.validation.audit_passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_orbit_leakage_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_s8_orbit_leakage.json");
    let validation = permutahedron_s8_orbit_leakage::write_artifact(std::path::Path::new(path));
    println!("{}", serde_json::to_string_pretty(&validation).unwrap());
    if !validation.audit_passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_orbit_leakage_verify() {
    let artifact = permutahedron_s8_orbit_leakage::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    if !artifact.validation.audit_passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_source_fixture_audit_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_s8_source_fixture_audit.json");
    let validation =
        permutahedron_s8_source_fixture_audit::write_artifact(std::path::Path::new(path));
    println!("{}", serde_json::to_string_pretty(&validation).unwrap());
    if !validation.audit_passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_source_fixture_audit_verify() {
    let artifact = permutahedron_s8_source_fixture_audit::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    if !artifact.validation.audit_passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_spectral_identifiability_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_s8_spectral_identifiability.json");
    let validation =
        permutahedron_s8_spectral_identifiability::write_artifact(std::path::Path::new(path));
    println!("{}", serde_json::to_string_pretty(&validation).unwrap());
    if !validation.audit_passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_spectral_identifiability_verify() {
    let artifact = permutahedron_s8_spectral_identifiability::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    if !artifact.validation.audit_passed {
        std::process::exit(2);
    }
}

fn cmd_perm_hypergraph_resolution_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/permutahedron_hypergraph_resolution.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_hypergraph_resolution_validation.json");
    let report = permutahedron_hypergraph_resolution::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_hypergraph_resolution_verify() {
    let artifact = permutahedron_hypergraph_resolution::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.findings).unwrap()
    );
    if !artifact.validation.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_hypergraph_signed_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/permutahedron_hypergraph_signed_transport.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_hypergraph_signed_transport_validation.json");
    let report = permutahedron_hypergraph_signed::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_hypergraph_signed_verify() {
    let artifact = permutahedron_hypergraph_signed::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.findings).unwrap()
    );
    if !artifact.validation.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_hypergraph_signed_equivalence_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/permutahedron_hypergraph_signed_equivalence.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_hypergraph_signed_equivalence_validation.json");
    let report = permutahedron_hypergraph_signed_equivalence::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_hypergraph_signed_equivalence_verify() {
    let artifact = permutahedron_hypergraph_signed_equivalence::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.findings).unwrap()
    );
    if !artifact.validation.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s4_susy_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/permutahedron_s4_supersymmetry.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_s4_supersymmetry_validation.json");
    let report = permutahedron_s4_supersymmetry::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s4_susy_verify() {
    let artifact = permutahedron_s4_supersymmetry::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    if !artifact.validation.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_conjugates_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/permutahedron_s8_conjugate_separation.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_s8_conjugate_separation_validation.json");
    let report = permutahedron_s8_conjugate_separation::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_conjugates_verify() {
    let artifact = permutahedron_s8_conjugate_separation::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.ordered_pair_correspondence).unwrap()
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.recursive_construction_audit).unwrap()
    );
    if !artifact.validation.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_orbits_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/permutahedron_s8_normalizer_orbits.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_s8_normalizer_orbits_validation.json");
    let report = permutahedron_s8_orbits::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_orbits_verify() {
    let artifact = permutahedron_s8_orbits::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    if !artifact.validation.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_separation_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/permutahedron_s8_separation_probe.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_s8_separation_probe_validation.json");
    let report = permutahedron_s8_separation::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_separation_verify() {
    let artifact = permutahedron_s8_separation::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.findings).unwrap()
    );
    if !artifact.validation.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_susy_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/permutahedron_s8_supersymmetry.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/permutahedron_s8_supersymmetry_validation.json");
    let report = permutahedron_s8_supersymmetry::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_perm_s8_susy_verify() {
    let artifact = permutahedron_s8_supersymmetry::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.separation).unwrap()
    );
    if !artifact.validation.passed {
        std::process::exit(2);
    }
}

fn cmd_vector_tensor_central_charge_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/vector_tensor_central_charge.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/vector_tensor_central_charge_validation.json");
    let report = vector_tensor_central_charge::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_vector_tensor_central_charge_verify() {
    let artifact = vector_tensor_central_charge::build();
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.validation).unwrap()
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.vector_tensor).unwrap()
    );
    if !artifact.validation.passed {
        std::process::exit(2);
    }
}

fn cmd_vector_tensor_central_equivalence_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/vector_tensor_central_equivalence.json");
    let report = vector_tensor_central_equivalence::write_artifact(std::path::Path::new(path));
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_vector_tensor_central_equivalence_verify() {
    let report = vector_tensor_central_equivalence::build();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_vector_tensor_central_atlas_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/vector_tensor_central_atlas.json");
    let report = vector_tensor_central_atlas::write_artifact(std::path::Path::new(path));
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_vector_tensor_central_atlas_verify() {
    let report = vector_tensor_central_atlas::build();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_vector_tensor_4d_verify() {
    let report = vector_tensor_4d::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_vector_tensor_4d_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/vector_tensor_4d.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/vector_tensor_4d_validation.json");
    let report = vector_tensor_4d::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_central_hypermultiplet_4d_verify() {
    let report = central_hypermultiplet_4d::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_central_hypermultiplet_4d_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/central_hypermultiplet_4d.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/central_hypermultiplet_4d_validation.json");
    let report = central_hypermultiplet_4d::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_scalar_tensor_tangent_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/scalar_tensor_tangent.json");
    let artifact = scalar_tensor_tangent::write_artifact(std::path::Path::new(path));
    println!("{}", serde_json::to_string_pretty(&artifact).unwrap());
    if !artifact.validation.passed {
        std::process::exit(2);
    }
}

fn cmd_scalar_tensor_tangent_verify() {
    let artifact = scalar_tensor_tangent::build();
    println!("{}", serde_json::to_string_pretty(&artifact).unwrap());
    if !artifact.validation.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_genome_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/adynkra_4d_n1_genomes.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/adynkra_4d_n1_genome_validation.json");
    let report = adynkra_genome::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_genome_verify() {
    let report = adynkra_genome::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_derivative_verify() {
    let report = supercovariant_derivative::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_intertwiner_verify() {
    let report = lorentz_intertwiners::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_vector_spinor_verify() {
    let report = vector_spinor_intertwiners::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_derivative_intertwiner_verify() {
    let report = adynkra_derivative_intertwiners::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_prepotential_gauge_verify() {
    let report = prepotential_gauge::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_prepotential_curvature_verify() {
    let report = prepotential_curvature::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_minimal_curvature_verify() {
    let report = minimal_supergravity_curvatures::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_minimal_action_verify() {
    let report = minimal_supergravity_action::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkrafield_operator_verify() {
    let report = adynkrafield_operator::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_prepotential_verify() {
    let report = eleven_dimensional_prepotential::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_top_down_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/adynkra_11d_top_down.json");
    let report = eleven_dimensional_top_down::write_artifact(std::path::Path::new(path));
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.bounded_gates_passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_second_momentum_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/adynkra_11d_second_momentum_validation.json");
    match eleven_dimensional_second_momentum::write_artifact(std::path::Path::new(path)) {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            if !report.passed {
                std::process::exit(2);
            }
        }
        Err(error) => {
            eprintln!("second-momentum artifact build failed: {error}");
            std::process::exit(2);
        }
    }
}

fn cmd_adynkra_11d_second_momentum_recoupling_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/adynkra_11d_second_momentum_recoupling.json");
    match eleven_dimensional_second_momentum_recoupling::write_artifact(std::path::Path::new(path))
    {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            if !report.passed {
                std::process::exit(2);
            }
        }
        Err(error) => {
            eprintln!("second-momentum recoupling build failed: {error}");
            std::process::exit(2);
        }
    }
}

fn cmd_adynkra_11d_second_momentum_component_build(args: &[String]) {
    let directory = std::path::Path::new(args.get(2).map(String::as_str).unwrap_or("results"));
    let checkpoint_directory = directory.join("adynkra_11d_second_momentum_30001_checkpoints");
    let map_10001 = directory.join("adynkra_11d_second_momentum_10001_maps.json");
    let map_30001 = directory.join("adynkra_11d_second_momentum_30001_maps.json");
    let remaining = directory.join("adynkra_11d_second_momentum_remaining_recouplings.json");
    let result = (|| -> std::io::Result<()> {
        eleven_dimensional_second_momentum_10001_maps::write_second_momentum_10001_map_artifact(
            &map_10001,
        )?;
        eleven_dimensional_second_momentum_30001_maps::write_artifact(
            &checkpoint_directory,
            &map_30001,
        )?;
        eleven_dimensional_second_momentum_remaining_recouplings::write_artifact(&remaining)?;
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("second-momentum component build failed: {error}");
        std::process::exit(2);
    }
    println!(
        "wrote {}, {}, and {}",
        map_10001.display(),
        map_30001.display(),
        remaining.display()
    );
}

fn cmd_adynkra_11d_second_momentum_10001_fx(args: &[String]) {
    if args.len() > 3 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-10001-fx [json]",
            args[0]
        );
        std::process::exit(2);
    }
    let path = std::path::Path::new(
        args.get(2)
            .map(String::as_str)
            .unwrap_or("results/adynkra_11d_second_momentum_10001_fx.json"),
    );
    match eleven_dimensional_second_momentum_10001_fx::write_second_momentum_10001_fx_artifact(path)
    {
        Ok(report) => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
        Err(error) => {
            eprintln!("second-momentum 10001 F_X build failed: {error}");
            std::process::exit(2);
        }
    }
}

fn cmd_adynkra_11d_second_momentum_full_map_plan(args: &[String]) {
    if args.len() != 2 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-full-map-plan",
            args[0]
        );
        std::process::exit(2);
    }
    let jobs = eleven_dimensional_second_momentum_full_maps::worklist();
    let unique_gpu_groups =
        eleven_dimensional_second_momentum_full_inventory::missing_unique_gpu_groups();
    let gpu_groups = eleven_dimensional_second_momentum_full_inventory::missing_gpu_groups();
    let payload = serde_json::json!({
        "schema_version": "adynkra-11d-second-momentum-missing-map-plan-v1",
        "role": "portable exact source-to-intermediate map jobs enabling the 49 columns outside the original 28-column slice",
        "full_column_layout_sha256": eleven_dimensional_second_momentum_full_inventory::layout_sha256(),
        "jobs_total": jobs.len(),
        "columns_enabled": eleven_dimensional_second_momentum_full_inventory::missing_49_column_specs().len(),
        "unique_path_gpu_columns": unique_gpu_groups.iter().map(Vec::len).sum::<usize>(),
        "gpu_columns_total": gpu_groups.iter().map(Vec::len).sum::<usize>(),
        "gpu_groups": gpu_groups.iter().enumerate().map(|(group_index, ordinals)| serde_json::json!({
            "group_index": group_index,
            "global_ordinals": ordinals,
            "width": ordinals.len()
        })).collect::<Vec<_>>(),
        "two_path_10001_columns": [15, 16, 17, 18],
        "jobs": jobs.iter().enumerate().map(|(ordinal, job)| serde_json::json!({
            "job_ordinal": ordinal,
            "job_key": job.key(),
            "target_dynkin_label": job.target_dynkin_label,
            "source_dynkin_label": job.source_dynkin_label,
            "source_copy": job.source_copy
        })).collect::<Vec<_>>()
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&payload).expect("serialize full map plan")
    );
}

fn cmd_adynkra_11d_second_momentum_full_map_worker(args: &[String]) {
    if args.len() < 3 || args.len() > 5 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-full-map-worker <job-list> [checkpoint-dir] [status-file]",
            args[0]
        );
        std::process::exit(2);
    }
    let jobs = eleven_dimensional_second_momentum_full_maps::parse_job_list(&args[2])
        .unwrap_or_else(|error| {
            eprintln!("invalid full-map job list: {error}");
            std::process::exit(2);
        });
    let directory = std::path::Path::new(
        args.get(3)
            .map(String::as_str)
            .unwrap_or("results/adynkra_11d_second_momentum_full_maps"),
    );
    let default_status = directory.join(format!("worker-status-{}.json", std::process::id()));
    let status_path = args
        .get(4)
        .map(std::path::PathBuf::from)
        .unwrap_or(default_status);
    let reporter = eleven_dimensional_second_momentum_full_maps::MissingMapProgressReporter::start(
        status_path,
        jobs.len(),
    )
    .unwrap_or_else(|error| {
        eprintln!("full-map worker status initialization failed: {error}");
        std::process::exit(2);
    });
    match eleven_dimensional_second_momentum_full_maps::run_jobs(directory, &jobs, |event| {
        reporter.observe(event)
    }) {
        Ok(summary) => {
            reporter.finish(&summary).unwrap_or_else(|error| {
                eprintln!("full-map worker terminal status failed: {error}");
                std::process::exit(2);
            });
        }
        Err(error) => {
            let _ = reporter.fail(&error);
            eprintln!("full-map worker failed: {error}");
            std::process::exit(2);
        }
    }
}

fn cmd_adynkra_11d_second_momentum_full_map_status(args: &[String]) {
    if args.len() > 3 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-full-map-status [checkpoint-dir]",
            args[0]
        );
        std::process::exit(2);
    }
    let directory = std::path::Path::new(
        args.get(2)
            .map(String::as_str)
            .unwrap_or("results/adynkra_11d_second_momentum_full_maps"),
    );
    let summary = eleven_dimensional_second_momentum_full_maps::summarize(directory);
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).expect("serialize full-map status")
    );
}

fn cmd_adynkra_11d_second_momentum_full_gpu_plan(args: &[String]) {
    if args.len() > 3 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-full-gpu-plan [output-dir]",
            args[0]
        );
        std::process::exit(2);
    }
    let manifest = second_momentum_full_gpu_jobs::build_manifest().unwrap_or_else(|error| {
        eprintln!("cannot build full GPU work manifest: {error}");
        std::process::exit(2);
    });
    if let Some(directory) = args.get(2) {
        let path = second_momentum_full_gpu_jobs::write_or_validate_manifest(std::path::Path::new(
            directory,
        ))
        .unwrap_or_else(|error| {
            eprintln!("cannot publish full GPU work manifest: {error}");
            std::process::exit(2);
        });
        eprintln!("published {}", path.display());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&manifest).expect("serialize full GPU work manifest")
    );
}

fn cmd_adynkra_11d_second_momentum_full_gpu_status(args: &[String]) {
    if args.len() < 3 || args.len() > 4 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-full-gpu-status <job-list> [output-dir]",
            args[0]
        );
        std::process::exit(2);
    }
    let jobs = second_momentum_full_gpu_jobs::parse_job_list(&args[2]).unwrap_or_else(|error| {
        eprintln!("invalid full GPU job list: {error}");
        std::process::exit(2);
    });
    let output_directory = std::path::Path::new(
        args.get(3)
            .map(String::as_str)
            .unwrap_or("results/second_momentum_full_gpu_fx"),
    );
    let summary = second_momentum_full_gpu_jobs::summarize(output_directory, &jobs);
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).expect("serialize full GPU status")
    );
    if summary
        .get("failed_count")
        .and_then(serde_json::Value::as_u64)
        != Some(0)
    {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_second_momentum_full_gpu_rank(args: &[String]) {
    if args.len() < 5 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-full-gpu-rank <prime> <output-json> <artifact-dir> [artifact-dir ...]",
            args[0]
        );
        std::process::exit(2);
    }
    let prime = args[2].parse::<u32>().unwrap_or_else(|_| {
        eprintln!("prime must be a pinned unsigned 32-bit prime");
        std::process::exit(2);
    });
    let output_path = std::path::Path::new(&args[3]);
    let input_directories = args[4..]
        .iter()
        .map(std::path::PathBuf::from)
        .collect::<Vec<_>>();
    let report =
        second_momentum_full_gpu_jobs::publish_full_rank(prime, &input_directories, output_path)
            .unwrap_or_else(|error| {
                eprintln!("full 77-column rank aggregation failed: {error}");
                std::process::exit(2);
            });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize full rank report")
    );
}

fn cmd_adynkra_11d_second_momentum_gpu_rank_28(args: &[String]) {
    if args.len() < 5 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-gpu-rank-28 <prime> <output-json> <artifact-dir> [artifact-dir ...]",
            args[0]
        );
        std::process::exit(2);
    }
    let prime = args[2].parse::<u32>().unwrap_or_else(|_| {
        eprintln!("prime must be a pinned unsigned 32-bit prime");
        std::process::exit(2);
    });
    let input_directories = args[4..].iter().map(std::path::PathBuf::from).collect::<Vec<_>>();
    let report = second_momentum_full_gpu_jobs::publish_declared_28_rank(
        prime,
        &input_directories,
        std::path::Path::new(&args[3]),
    )
    .unwrap_or_else(|error| {
        eprintln!("declared-28 rank aggregation failed: {error}");
        std::process::exit(2);
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

#[cfg(feature = "cuda")]
fn cmd_adynkra_11d_second_momentum_full_gpu_worker(args: &[String]) {
    use second_momentum_gpu_progress::{GroupProgressConfig, ProgressConfig, ProgressReporter};

    if args.len() < 4 || args.len() > 7 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-full-gpu-worker <job-list> <map-dir> [output-dir] [device] [cpu-parity-terms]",
            args[0]
        );
        std::process::exit(2);
    }
    let jobs = second_momentum_full_gpu_jobs::parse_job_list(&args[2]).unwrap_or_else(|error| {
        eprintln!("invalid full GPU job list: {error}");
        std::process::exit(2);
    });
    let map_directory = std::path::PathBuf::from(&args[3]);
    let selected_ordinals = jobs
        .iter()
        .flat_map(second_momentum_full_gpu_jobs::FullGpuJobKey::global_ordinals)
        .collect::<Vec<_>>();
    if selected_ordinals
        .iter()
        .any(|ordinal| !(19..=22).contains(ordinal))
    {
        let map_summary = eleven_dimensional_second_momentum_full_maps::summarize(&map_directory);
        if !map_summary.passed {
            eprintln!(
                "selected full GPU jobs require all 47 verified maps; found {}/47",
                map_summary.completed_jobs
            );
            std::process::exit(2);
        }
    }
    let output_directory = std::path::PathBuf::from(
        args.get(4)
            .map(String::as_str)
            .unwrap_or("results/second_momentum_full_gpu_fx"),
    );
    let device = args
        .get(5)
        .map(|value| value.parse::<i32>())
        .transpose()
        .unwrap_or_else(|_| {
            eprintln!("device must be a nonnegative integer");
            std::process::exit(2);
        })
        .unwrap_or(0);
    let cpu_parity_terms = args
        .get(6)
        .map(|value| value.parse::<usize>())
        .transpose()
        .unwrap_or_else(|_| {
            eprintln!("CPU parity terms must be a positive integer");
            std::process::exit(2);
        })
        .unwrap_or(128);
    if device < 0 || cpu_parity_terms == 0 {
        eprintln!("device must be nonnegative and CPU parity terms must be nonzero");
        std::process::exit(2);
    }
    second_momentum_full_gpu_jobs::write_or_validate_manifest(&output_directory).unwrap_or_else(
        |error| {
            eprintln!("cannot establish full GPU work manifest: {error}");
            std::process::exit(2);
        },
    );
    let columns = eleven_dimensional_second_momentum_full_inventory::full_column_specs();
    for (job_ordinal, job) in jobs.iter().enumerate() {
        match second_momentum_full_gpu_jobs::validate_completed_job(&output_directory, job) {
            Ok(true) => {
                println!(
                    "{}",
                    serde_json::json!({
                        "schema_version": second_momentum_full_gpu_jobs::FULL_GPU_RUN_SCHEMA,
                        "event": "job_adopted",
                        "job_id": job.id(),
                        "job_ordinal": job_ordinal,
                        "jobs_total": jobs.len()
                    })
                );
                continue;
            }
            Ok(false) => {}
            Err(error) => {
                eprintln!("cannot adopt {}: {error}", job.id());
                std::process::exit(2);
            }
        }
        let global_ordinals = job.global_ordinals();
        let tranche = job.tranche();
        let source_copies = global_ordinals
            .iter()
            .map(|ordinal| columns[*ordinal].source_copy)
            .collect::<Vec<_>>();
        let tranche_columns_total = columns
            .iter()
            .filter(|column| column.intermediate_dynkin_label == tranche)
            .count();
        let job_directory = output_directory.join("jobs").join(job.id());
        let report_path = second_momentum_full_gpu_jobs::report_path(&output_directory, job);
        let checkpoint_path = job_directory.join("checkpoint.json");
        let event_log_path = job_directory.join("events.jsonl");
        let status_path = job_directory.join("status.json");
        let reporter = ProgressReporter::start(ProgressConfig {
            command: "adynkra-11d-second-momentum-full-gpu-worker".to_string(),
            tranche: tranche.clone(),
            local_ordinal: global_ordinals[0],
            global_ordinal: global_ordinals[0],
            tranche_columns_total,
            prime: job.prime(),
            device,
            cpu_parity_terms,
            output_directory: output_directory.clone(),
            binary_output_path: report_path.clone(),
            report_output_path: report_path,
            status_snapshot_path: status_path,
            group: Some(GroupProgressConfig {
                job_id: job.id(),
                group_id: format!("pending-preflight:{}", job.id()),
                active_columns: global_ordinals.len(),
                ordered_local_ordinals: global_ordinals.clone(),
                ordered_global_ordinals: global_ordinals,
                ordered_source_copies: source_copies,
                checkpoint_path,
                event_log_path,
                resumable: true,
            }),
        })
        .unwrap_or_else(|error| {
            eprintln!("cannot start {} progress reporter: {error}", job.id());
            std::process::exit(2);
        });
        reporter
            .phase_start("group_execution")
            .unwrap_or_else(|error| {
                eprintln!("cannot mark {} started: {error}", job.id());
                std::process::exit(2);
            });
        let live = reporter.live_progress();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            second_momentum_gpu_jobs::run_full_group_job(
                job,
                &map_directory,
                &output_directory,
                device,
                cpu_parity_terms,
                &live,
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string()))
        }));
        match outcome {
            Ok(Ok(result)) => {
                if let Err(error) = reporter.finish_success(result) {
                    eprintln!("{} terminal status failed: {error}", job.id());
                    std::process::exit(2);
                }
            }
            Ok(Err(error)) => {
                let _ = reporter.finish_failure(&error);
                eprintln!("{} failed: {error}", job.id());
                std::process::exit(2);
            }
            Err(payload) => {
                let message = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("unknown panic");
                let _ = reporter.finish_failure(format!("panic: {message}"));
                eprintln!("{} panicked: {message}", job.id());
                std::process::exit(101);
            }
        }
    }
    let summary = second_momentum_full_gpu_jobs::summarize(&output_directory, &jobs);
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).expect("serialize full GPU worker summary")
    );
}

#[cfg(not(feature = "cuda"))]
fn cmd_adynkra_11d_second_momentum_full_gpu_worker(_args: &[String]) {
    eprintln!("full second-momentum GPU worker requires a --features cuda build");
    std::process::exit(2);
}

fn cmd_adynkra_11d_second_momentum_gpu_status_reconcile(args: &[String]) {
    if args.len() != 5 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-gpu-status-reconcile <status-file> <child-pid> <exit:N|signal:N|unknown>",
            args.first().map(String::as_str).unwrap_or("adinkra")
        );
        std::process::exit(2);
    }
    let status_path = std::path::Path::new(&args[2]);
    let child_pid = args[3].parse::<u32>().unwrap_or_else(|_| {
        eprintln!("invalid supervised child PID {}", args[3]);
        std::process::exit(2);
    });
    let (exit_code, signal) = if args[4] == "unknown" {
        (None, None)
    } else if let Some(value) = args[4].strip_prefix("exit:") {
        let code = value.parse::<i32>().unwrap_or_else(|_| {
            eprintln!("invalid supervised exit observation {}", args[4]);
            std::process::exit(2);
        });
        (Some(code), None)
    } else if let Some(value) = args[4].strip_prefix("signal:") {
        let number = value.parse::<i32>().unwrap_or_else(|_| {
            eprintln!("invalid supervised signal observation {}", args[4]);
            std::process::exit(2);
        });
        if number <= 0 {
            eprintln!("supervised signal number must be positive");
            std::process::exit(2);
        }
        (None, Some(number))
    } else {
        eprintln!("observation must be exit:N, signal:N, or unknown");
        std::process::exit(2);
    };

    match second_momentum_gpu_progress::reconcile_status_snapshot(
        status_path,
        child_pid,
        exit_code,
        signal,
    ) {
        Ok(result) => println!(
            "{}",
            serde_json::json!({
                "status_snapshot_path": status_path.display().to_string(),
                "resumable": false,
                "reconciled": result.reconciled,
                "state": result.state
            })
        ),
        Err(error) => {
            eprintln!("status snapshot reconciliation failed: {error}");
            std::process::exit(2);
        }
    }
}

fn cmd_adynkra_11d_second_momentum_cpu_fx(args: &[String]) {
    use second_momentum_cpu_progress::{CpuProgressConfig, CpuProgressReporter};

    if args.len() > 5 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-cpu-fx <20001|30001> [output-file] [status-file]",
            args[0]
        );
        std::process::exit(2);
    }
    let tranche = args.get(2).map(String::as_str).unwrap_or_else(|| {
        eprintln!("missing tranche 20001 or 30001");
        std::process::exit(2);
    });
    let columns_total = match tranche {
        "20001" => 9,
        "30001" => 15,
        _ => {
            eprintln!("tranche must be 20001 or 30001");
            std::process::exit(2);
        }
    };
    let output_path = args
        .get(3)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(format!(
                "results/adynkra_11d_second_momentum_{tranche}_fx.json"
            ))
        });
    let status_path = args
        .get(4)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| output_path.with_extension("status.json"));
    let reporter = CpuProgressReporter::start(CpuProgressConfig {
        tranche: tranche.to_owned(),
        columns_total,
        output_path: output_path.clone(),
        status_path,
    })
    .unwrap_or_else(|error| {
        eprintln!("CPU progress initialization failed: {error}");
        std::process::exit(2);
    });

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match tranche {
        "20001" => eleven_dimensional_second_momentum_20001_fx::write_artifact_with_progress(
            &output_path,
            |event| reporter.observe(event),
        )
        .and_then(|report| serde_json::to_value(report).map_err(std::io::Error::other)),
        "30001" => eleven_dimensional_second_momentum_30001_fx::write_artifact_with_progress(
            &output_path,
            |event| reporter.observe(event),
        )
        .and_then(|report| serde_json::to_value(report).map_err(std::io::Error::other)),
        _ => unreachable!(),
    }));
    match outcome {
        Ok(Ok(result)) => {
            if let Err(error) = reporter.finish_success(&result) {
                eprintln!("CPU terminal status publication failed: {error}");
                std::process::exit(2);
            }
        }
        Ok(Err(error)) => {
            let _ = reporter.finish_failure(error.to_string());
            eprintln!("CPU tranche failed: {error}");
            std::process::exit(2);
        }
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown panic");
            let _ = reporter.finish_failure(format!("panic: {message}"));
            eprintln!("CPU tranche panicked: {message}");
            std::process::exit(101);
        }
    }
}

fn cmd_adynkra_11d_second_momentum_gpu_fx_plan(args: &[String]) {
    if args.len() > 3 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-gpu-fx-plan [output-dir]",
            args[0]
        );
        std::process::exit(2);
    }
    let manifest = second_momentum_gpu_jobs::build_job_manifest().unwrap_or_else(|error| {
        eprintln!("cannot build GPU work manifest: {error}");
        std::process::exit(2);
    });
    if let Some(directory) = args.get(2) {
        let path =
            second_momentum_gpu_jobs::write_or_validate_manifest(std::path::Path::new(directory))
                .unwrap_or_else(|error| {
                    eprintln!("cannot publish GPU work manifest: {error}");
                    std::process::exit(2);
                });
        eprintln!("published {}", path.display());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&manifest).expect("serialize GPU work manifest")
    );
}

fn cmd_adynkra_11d_second_momentum_gpu_fx_status(args: &[String]) {
    if args.len() < 3 || args.len() > 4 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-gpu-fx-status <job-list> [output-dir]",
            args[0]
        );
        std::process::exit(2);
    }
    let jobs = second_momentum_gpu_jobs::parse_job_list(&args[2]).unwrap_or_else(|error| {
        eprintln!("invalid GPU job list: {error}");
        std::process::exit(2);
    });
    let output_directory = std::path::Path::new(
        args.get(3)
            .map(String::as_str)
            .unwrap_or("results/second_momentum_gpu_fx"),
    );
    let summary = second_momentum_gpu_jobs::summarize_jobs(output_directory, &jobs);
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).expect("serialize GPU work status")
    );
    if summary
        .get("failed_count")
        .and_then(serde_json::Value::as_u64)
        != Some(0)
    {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_second_momentum_gpu_fx_import(args: &[String]) {
    if args.len() < 4 || args.len() > 5 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-gpu-fx-import <job-list> <source-dir> [destination-dir]",
            args[0]
        );
        std::process::exit(2);
    }
    let jobs = second_momentum_gpu_jobs::parse_job_list(&args[2]).unwrap_or_else(|error| {
        eprintln!("invalid GPU job list: {error}");
        std::process::exit(2);
    });
    let source = std::path::Path::new(&args[3]);
    let destination = std::path::Path::new(
        args.get(4)
            .map(String::as_str)
            .unwrap_or("results/second_momentum_gpu_fx"),
    );
    let result = second_momentum_gpu_jobs::import_completed_jobs(source, destination, &jobs)
        .unwrap_or_else(|error| {
            eprintln!("GPU job import failed: {error}");
            std::process::exit(2);
        });
    println!(
        "{}",
        serde_json::to_string_pretty(&result).expect("serialize GPU import report")
    );
}

#[cfg(feature = "cuda")]
fn cmd_adynkra_11d_second_momentum_gpu_fx_worker(args: &[String]) {
    use second_momentum_gpu_progress::{GroupProgressConfig, ProgressConfig, ProgressReporter};

    if args.len() < 3 || args.len() > 6 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-gpu-fx-worker <job-list> [output-dir] [device] [cpu-parity-terms]",
            args[0]
        );
        std::process::exit(2);
    }
    let jobs = second_momentum_gpu_jobs::parse_job_list(&args[2]).unwrap_or_else(|error| {
        eprintln!("invalid GPU job list: {error}");
        std::process::exit(2);
    });
    let output_directory = std::path::PathBuf::from(
        args.get(3)
            .map(String::as_str)
            .unwrap_or("results/second_momentum_gpu_fx"),
    );
    let device = args
        .get(4)
        .map(|value| value.parse::<i32>())
        .transpose()
        .unwrap_or_else(|_| {
            eprintln!("device must be a nonnegative integer");
            std::process::exit(2);
        })
        .unwrap_or(0);
    let cpu_parity_terms = args
        .get(5)
        .map(|value| value.parse::<usize>())
        .transpose()
        .unwrap_or_else(|_| {
            eprintln!("CPU parity terms must be a positive integer");
            std::process::exit(2);
        })
        .unwrap_or(128);
    if device < 0 || cpu_parity_terms == 0 {
        eprintln!("device must be nonnegative and CPU parity terms must be nonzero");
        std::process::exit(2);
    }
    second_momentum_gpu_jobs::write_or_validate_manifest(&output_directory).unwrap_or_else(
        |error| {
            eprintln!("cannot establish GPU work manifest: {error}");
            std::process::exit(2);
        },
    );

    for (job_ordinal, job) in jobs.iter().enumerate() {
        match second_momentum_gpu_jobs::validate_completed_job(&output_directory, job) {
            Ok(true) => {
                println!(
                    "{}",
                    serde_json::json!({
                        "schema_version": second_momentum_gpu_jobs::GPU_GROUP_RUN_SCHEMA,
                        "event": "job_adopted",
                        "job_id": job.id(),
                        "job_ordinal": job_ordinal,
                        "jobs_total": jobs.len()
                    })
                );
                continue;
            }
            Ok(false) => {}
            Err(error) => {
                eprintln!("cannot adopt {}: {error}", job.id());
                std::process::exit(2);
            }
        }
        let tranche = job.tranche().unwrap_or_else(|error| {
            eprintln!("invalid job tranche: {error}");
            std::process::exit(2);
        });
        let local_ordinals = job.local_ordinals().unwrap_or_else(|error| {
            eprintln!("invalid job group: {error}");
            std::process::exit(2);
        });
        let first_global = if tranche.as_str() == "20001" { 53 } else { 62 };
        let global_ordinals = local_ordinals
            .iter()
            .map(|ordinal| first_global + ordinal)
            .collect::<Vec<_>>();
        let job_directory = output_directory.join("jobs").join(job.id());
        let report_path =
            second_momentum_gpu_jobs::completed_job_report_path(&output_directory, job);
        let checkpoint_path = job_directory.join("checkpoint.json");
        let event_log_path = job_directory.join("events.jsonl");
        let status_path = job_directory.join("status.json");
        let reporter = ProgressReporter::start(ProgressConfig {
            command: "adynkra-11d-second-momentum-gpu-fx-worker".to_string(),
            tranche: tranche.as_str().to_string(),
            local_ordinal: local_ordinals[0],
            global_ordinal: global_ordinals[0],
            tranche_columns_total: if tranche.as_str() == "20001" { 9 } else { 15 },
            prime: job.prime().unwrap_or_else(|error| {
                eprintln!("invalid job prime: {error}");
                std::process::exit(2);
            }),
            device,
            cpu_parity_terms,
            output_directory: output_directory.clone(),
            binary_output_path: report_path.clone(),
            report_output_path: report_path,
            status_snapshot_path: status_path,
            group: Some(GroupProgressConfig {
                job_id: job.id(),
                group_id: format!("pending-preflight:{}", job.id()),
                active_columns: local_ordinals.len(),
                ordered_local_ordinals: local_ordinals.clone(),
                ordered_global_ordinals: global_ordinals,
                ordered_source_copies: (1..=local_ordinals.len()).collect(),
                checkpoint_path,
                event_log_path,
                resumable: local_ordinals.len() > 1,
            }),
        })
        .unwrap_or_else(|error| {
            eprintln!("cannot start {} progress reporter: {error}", job.id());
            std::process::exit(2);
        });
        reporter
            .phase_start("group_execution")
            .unwrap_or_else(|error| {
                eprintln!("cannot mark {} started: {error}", job.id());
                std::process::exit(2);
            });
        let live = reporter.live_progress();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if local_ordinals.len() == 1 {
                second_momentum_gpu_jobs::run_singleton_job(
                    &output_directory,
                    job,
                    device,
                    cpu_parity_terms,
                    &live,
                )
            } else {
                second_momentum_gpu_jobs::run_group_job(
                    job,
                    &output_directory,
                    device,
                    cpu_parity_terms,
                    &live,
                )
                .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string()))
            }
        }));
        match outcome {
            Ok(Ok(result)) => {
                if let Err(error) = reporter.finish_success(result) {
                    eprintln!("{} terminal status failed: {error}", job.id());
                    std::process::exit(2);
                }
            }
            Ok(Err(error)) => {
                let _ = reporter.finish_failure(&error);
                eprintln!("{} failed: {error}", job.id());
                std::process::exit(2);
            }
            Err(payload) => {
                let message = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("unknown panic");
                let _ = reporter.finish_failure(format!("panic: {message}"));
                eprintln!("{} panicked: {message}", job.id());
                std::process::exit(101);
            }
        }
    }
    println!(
        "{}",
        second_momentum_gpu_jobs::summarize_jobs(&output_directory, &jobs)
    );
}

#[cfg(not(feature = "cuda"))]
fn cmd_adynkra_11d_second_momentum_gpu_fx_worker(_args: &[String]) {
    eprintln!("GPU group worker requires a Linux build with --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn cmd_adynkra_11d_second_momentum_gpu_fx_multi_prime_worker(args: &[String]) {
    use second_momentum_gpu_progress::{GroupProgressConfig, ProgressConfig, ProgressReporter};

    if args.len() < 3 || args.len() > 6 {
        eprintln!(
            "usage: {} adynkra-11d-second-momentum-gpu-fx-multi-prime-worker <same-group-job-list> [output-dir] [device] [cpu-parity-terms]",
            args[0]
        );
        std::process::exit(2);
    }
    let jobs = second_momentum_gpu_jobs::parse_job_list(&args[2]).unwrap_or_else(|error| {
        eprintln!("invalid multi-prime GPU job list: {error}");
        std::process::exit(2);
    });
    if !(2..=3).contains(&jobs.len()) {
        eprintln!("multi-prime worker requires exactly 2 or 3 jobs");
        std::process::exit(2);
    }
    let first = &jobs[0];
    if jobs
        .iter()
        .any(|job| job.tranche != first.tranche || job.group_index != first.group_index)
    {
        eprintln!("multi-prime worker jobs must belong to one tranche/group");
        std::process::exit(2);
    }
    let output_directory = std::path::PathBuf::from(
        args.get(3)
            .map(String::as_str)
            .unwrap_or("results/second_momentum_gpu_fx"),
    );
    let device = args
        .get(4)
        .map(|value| value.parse::<i32>())
        .transpose()
        .unwrap_or_else(|_| {
            eprintln!("device must be a nonnegative integer");
            std::process::exit(2);
        })
        .unwrap_or(0);
    let cpu_parity_terms = args
        .get(5)
        .map(|value| value.parse::<usize>())
        .transpose()
        .unwrap_or_else(|_| {
            eprintln!("CPU parity terms must be a positive integer");
            std::process::exit(2);
        })
        .unwrap_or(128);
    let tranche = first.tranche().unwrap_or_else(|error| {
        eprintln!("invalid multi-prime tranche: {error}");
        std::process::exit(2);
    });
    let local_ordinals = first.local_ordinals().unwrap_or_else(|error| {
        eprintln!("invalid multi-prime group: {error}");
        std::process::exit(2);
    });
    if local_ordinals.len() < 2 {
        eprintln!("multi-prime worker currently requires a width-2/3 group");
        std::process::exit(2);
    }
    let first_global = if tranche.as_str() == "20001" { 53 } else { 62 };
    let global_ordinals = local_ordinals
        .iter()
        .map(|ordinal| first_global + ordinal)
        .collect::<Vec<_>>();
    let bundle_id = format!(
        "{}-g{}-mp{}",
        first.tranche,
        first.group_index,
        jobs.iter()
            .map(|job| job.prime_index.to_string())
            .collect::<String>()
    );
    let bundle_directory = output_directory.join("jobs").join(&bundle_id);
    let report_path = bundle_directory.join("bundle-result.json");
    let reporter = ProgressReporter::start(ProgressConfig {
        command: "adynkra-11d-second-momentum-gpu-fx-multi-prime-worker".to_string(),
        tranche: tranche.as_str().to_string(),
        local_ordinal: local_ordinals[0],
        global_ordinal: global_ordinals[0],
        tranche_columns_total: if tranche.as_str() == "20001" { 9 } else { 15 },
        prime: jobs[0].prime().unwrap_or_else(|error| {
            eprintln!("invalid first bundle prime: {error}");
            std::process::exit(2);
        }),
        device,
        cpu_parity_terms,
        output_directory: output_directory.clone(),
        binary_output_path: report_path.clone(),
        report_output_path: report_path,
        status_snapshot_path: bundle_directory.join("status.json"),
        group: Some(GroupProgressConfig {
            job_id: bundle_id.clone(),
            group_id: format!("pending-preflight:{bundle_id}"),
            active_columns: local_ordinals.len(),
            ordered_local_ordinals: local_ordinals.clone(),
            ordered_global_ordinals: global_ordinals,
            ordered_source_copies: (1..=local_ordinals.len()).collect(),
            checkpoint_path: bundle_directory.join("checkpoint.json"),
            event_log_path: bundle_directory.join("events.jsonl"),
            resumable: true,
        }),
    })
    .unwrap_or_else(|error| {
        eprintln!("cannot start {bundle_id} progress reporter: {error}");
        std::process::exit(2);
    });
    reporter
        .phase_start("multi_prime_group_execution")
        .unwrap_or_else(|error| {
            eprintln!("cannot mark {bundle_id} started: {error}");
            std::process::exit(2);
        });
    let live = reporter.live_progress();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        second_momentum_gpu_jobs::run_multi_prime_group_jobs(
            &jobs,
            &output_directory,
            device,
            cpu_parity_terms,
            &live,
        )
        .and_then(|reports| serde_json::to_value(reports).map_err(|error| error.to_string()))
    }));
    match outcome {
        Ok(Ok(result)) => {
            if let Err(error) = reporter.finish_success(result) {
                eprintln!("{bundle_id} terminal status failed: {error}");
                std::process::exit(2);
            }
        }
        Ok(Err(error)) => {
            let _ = reporter.finish_failure(&error);
            eprintln!("{bundle_id} failed: {error}");
            std::process::exit(2);
        }
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown panic");
            let _ = reporter.finish_failure(format!("panic: {message}"));
            eprintln!("{bundle_id} panicked: {message}");
            std::process::exit(101);
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn cmd_adynkra_11d_second_momentum_gpu_fx_multi_prime_worker(_args: &[String]) {
    eprintln!("GPU multi-prime worker requires a Linux build with --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn cmd_adynkra_11d_second_momentum_gpu_fx(args: &[String]) {
    use second_momentum_gpu_progress::{ProgressConfig, ProgressReporter, emit_fallback_error};

    if args.len() > 9 {
        second_momentum_gpu_argument_failure(
            "too many arguments for second-momentum GPU F_X".to_owned(),
        );
    }
    let tranche = args
        .get(2)
        .map(String::as_str)
        .ok_or_else(|| "missing tranche 20001 or 30001".to_owned())
        .unwrap_or_else(|error| second_momentum_gpu_argument_failure(error));
    let (first_global_ordinal, tranche_columns_total) = match tranche {
        "20001" => (53, 9),
        "30001" => (62, 15),
        _ => second_momentum_gpu_argument_failure("tranche must be 20001 or 30001".to_owned()),
    };
    let local_ordinal = args
        .get(3)
        .ok_or_else(|| "missing local column ordinal".to_owned())
        .and_then(|value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("invalid local column ordinal {value}"))
        })
        .unwrap_or_else(|error| second_momentum_gpu_argument_failure(error));
    if local_ordinal >= tranche_columns_total {
        second_momentum_gpu_argument_failure(format!(
            "{tranche} local column ordinal must lie in 0..{tranche_columns_total}"
        ));
    }
    let prime = args
        .get(4)
        .map(|value| {
            value
                .parse::<u32>()
                .map_err(|_| format!("invalid finite-field prime {value}"))
        })
        .transpose()
        .unwrap_or_else(|error| second_momentum_gpu_argument_failure(error))
        .unwrap_or(eleven_dimensional_second_momentum_gpu::GPU_FX_PRIMES[0]);
    let output_directory = std::path::PathBuf::from(
        args.get(5)
            .map(String::as_str)
            .unwrap_or("results/second_momentum_gpu_fx"),
    );
    let cpu_parity_terms = args
        .get(6)
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("invalid CPU parity term count {value}"))
        })
        .transpose()
        .unwrap_or_else(|error| second_momentum_gpu_argument_failure(error))
        .unwrap_or(128);
    let device = args
        .get(7)
        .map(|value| {
            value
                .parse::<i32>()
                .map_err(|_| format!("invalid CUDA device {value}"))
        })
        .transpose()
        .unwrap_or_else(|error| second_momentum_gpu_argument_failure(error))
        .unwrap_or(0);
    if device < 0 {
        second_momentum_gpu_argument_failure("CUDA device must be nonnegative".to_owned());
    }

    let global_ordinal = first_global_ordinal + local_ordinal;
    let stem = format!("second_momentum_{tranche}_column_{global_ordinal:02}_p{prime}");
    let binary_output_path = output_directory.join(format!("{stem}.bin"));
    let report_output_path = output_directory.join(format!("{stem}.json"));
    let status_path = args
        .get(8)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| output_directory.join(format!("{stem}.status.json")));
    let config = ProgressConfig {
        command: "adynkra-11d-second-momentum-gpu-fx".to_owned(),
        tranche: tranche.to_owned(),
        local_ordinal,
        global_ordinal,
        tranche_columns_total,
        prime,
        device,
        cpu_parity_terms,
        output_directory: output_directory.clone(),
        binary_output_path,
        report_output_path,
        status_snapshot_path: status_path,
        group: None,
    };
    let reporter = ProgressReporter::start(config).unwrap_or_else(|error| {
        emit_fallback_error("progress_initialization_error", &error.to_string());
        std::process::exit(2);
    });
    if let Err(error) = reporter.phase_start("column_execution") {
        let message = format!("failed to start progress phase: {error}");
        let _ = reporter.finish_failure(&message);
        emit_fallback_error("progress_error", &message);
        std::process::exit(2);
    }

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let live_progress = reporter.live_progress();
        eleven_dimensional_second_momentum_gpu::run_cuda_column(
            tranche,
            local_ordinal,
            prime,
            device,
            &output_directory,
            cpu_parity_terms,
            Some(&live_progress),
        )
    }));
    if let Some(signal) = reporter.observed_termination_signal() {
        let _ = reporter.finish_terminated(signal);
        std::process::exit(128 + signal);
    }
    match outcome {
        Ok(Ok(report)) => {
            if let Err(error) = reporter.phase_end(format!(
                "column execution completed: {} source terms and {} expanded contributions",
                report.source_terms, report.expanded_contributions
            )) {
                let message = format!("failed to finish progress phase: {error}");
                let _ = reporter.finish_failure(&message);
                emit_fallback_error("progress_error", &message);
                std::process::exit(2);
            }
            let result = match serde_json::to_value(&report) {
                Ok(result) => result,
                Err(error) => {
                    let message = format!("failed to serialize CUDA column report: {error}");
                    let _ = reporter.finish_failure(&message);
                    emit_fallback_error("serialization_error", &message);
                    std::process::exit(2);
                }
            };
            if let Err(error) = reporter.finish_success(result) {
                emit_fallback_error("terminal_status_snapshot_error", &error.to_string());
                std::process::exit(2);
            }
        }
        Ok(Err(error)) => {
            let _ = reporter.phase_end("column execution failed");
            if let Err(status_error) = reporter.finish_failure(&error) {
                emit_fallback_error("terminal_status_snapshot_error", &status_error.to_string());
            }
            std::process::exit(2);
        }
        Err(payload) => {
            let panic_message = if let Some(message) = payload.downcast_ref::<&str>() {
                (*message).to_owned()
            } else if let Some(message) = payload.downcast_ref::<String>() {
                message.clone()
            } else {
                "non-string panic payload".to_owned()
            };
            let error = format!("GPU column command panicked: {panic_message}");
            let _ = reporter.phase_end("column execution panicked");
            if let Err(status_error) = reporter.finish_failure(&error) {
                emit_fallback_error("terminal_status_snapshot_error", &status_error.to_string());
            }
            std::process::exit(2);
        }
    }
}

#[cfg(feature = "cuda")]
fn second_momentum_gpu_argument_failure(message: String) -> ! {
    second_momentum_gpu_progress::emit_fallback_error("argument_error", &message);
    std::process::exit(2);
}

#[cfg(not(feature = "cuda"))]
fn cmd_adynkra_11d_second_momentum_gpu_fx(_args: &[String]) {
    println!(
        "{}",
        serde_json::json!({
            "schema_version": second_momentum_gpu_progress::PROGRESS_SCHEMA,
            "event": "terminal",
            "state": "failed",
            "phase": "terminal",
            "pid": std::process::id(),
            "resources": {
                "gpu": {
                    "available": false,
                    "reason": "binary was compiled without the cuda feature"
                }
            },
            "error": "second-momentum CUDA F_X requires a Linux build with --features cuda"
        })
    );
    std::process::exit(2);
}

fn cmd_adynkra_11d_first_momentum_fx_aggregate(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/adynkra_11d_first_momentum_physical_fx_functional.json");
    let checkpoint_root = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/eleven_dimensional_first_momentum_fx_checkpoints");
    if let Err(error) = eleven_dimensional_physical_curvature::
        merge_first_momentum_fx_functional_artifact_from_complete_checkpoints(
            std::path::Path::new(path),
            std::path::Path::new(checkpoint_root),
        )
    {
        eprintln!("first-momentum F_X checkpoint merge failed: {error}");
        std::process::exit(2);
    }
    println!("wrote {path} from 336 validated checkpoints in {checkpoint_root}");
}

fn cmd_adynkra_11d_free_complex_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/eleven_dimensional_free_complex.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/adynkra_11d_free_complex_validation.json");
    let report = eleven_dimensional_free_complex::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_hook_bianchi_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/eleven_dimensional_hook_bianchi.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/adynkra_11d_hook_bianchi_validation.json");
    let report = eleven_dimensional_hook_bianchi::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_level18_momentum_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/eleven_dimensional_level18_momentum.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/adynkra_11d_level18_momentum_validation.json");
    let report = eleven_dimensional_level18_momentum::write_artifacts(
        std::path::Path::new("data/eleven_dimensional_spinor_bridge"),
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    );
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.bounded_program_passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_target_stream_build(args: &[String]) {
    let data_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/eleven_dimensional_target_stream.json");
    let validation_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/adynkra_11d_target_stream_validation.json");
    eleven_dimensional_target_stream::write_artifacts(
        std::path::Path::new(data_path),
        std::path::Path::new(validation_path),
    )
    .unwrap_or_else(|error| {
        eprintln!("Failed to write target stream artifacts: {error}");
        std::process::exit(2);
    });
    let report = eleven_dimensional_target_stream::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_prepotential_gate_build(args: &[String]) {
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/adynkra_11d_prepotential_gate.json");
    let report = eleven_dimensional_prepotential_gate::write_json(std::path::Path::new(path))
        .unwrap_or_else(|error| {
            eprintln!("Failed to write {path}: {error}");
            std::process::exit(2);
        });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.worklist_consistent_with_current_exact_engine {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_clifford_verify() {
    let report = eleven_dimensional_clifford::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_gauge_intertwiner_verify() {
    let report = eleven_dimensional_gauge::verify();
    let output = std::path::PathBuf::from("results/adynkra_11d_gauge_intertwiners.json");
    eleven_dimensional_level16_couplings::write_atomic_json(&output, &report, report.passed)
        .unwrap_or_else(|error| {
            eprintln!("Failed to checkpoint {}: {error}", output.display());
            std::process::exit(2);
        });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_gauge_composition_manifest() {
    let specs = eleven_dimensional_gauge::gauge_composition_specs();
    println!("{}", serde_json::to_string_pretty(&specs).unwrap());
}

fn cmd_adynkra_11d_gauge_zero_column(args: &[String]) {
    let usage = || {
        eprintln!(
            "Usage: {} adynkra-11d-gauge-zero-column <form-degree> <leading-ordinal> <root>",
            args[0]
        );
        std::process::exit(1);
    };
    let gauge_form_degree = args
        .get(2)
        .unwrap_or_else(|| usage())
        .parse::<usize>()
        .unwrap_or_else(|error| {
            eprintln!("Invalid gauge form degree: {error}");
            std::process::exit(1);
        });
    let leading_ordinal = args
        .get(3)
        .unwrap_or_else(|| usage())
        .parse::<usize>()
        .unwrap_or_else(|error| {
            eprintln!("Invalid leading ordinal: {error}");
            std::process::exit(1);
        });
    let root = std::path::PathBuf::from(args.get(4).unwrap_or_else(|| usage()));
    let report = eleven_dimensional_gauge::build_and_write_zero_momentum_gauge_composition_artifact(
        gauge_form_degree,
        leading_ordinal,
        &root,
    )
    .unwrap_or_else(|error| {
        eprintln!(
            "Failed to build zero-momentum gauge composition p={gauge_form_degree}, column={leading_ordinal}: {error}"
        );
        std::process::exit(2);
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn cmd_adynkra_11d_gauge_zero_merge(args: &[String]) {
    let usage = || {
        eprintln!(
            "Usage: {} adynkra-11d-gauge-zero-merge <form-degree> <root> [--deep]",
            args[0]
        );
        std::process::exit(1);
    };
    let gauge_form_degree = args
        .get(2)
        .unwrap_or_else(|| usage())
        .parse::<usize>()
        .unwrap_or_else(|error| {
            eprintln!("Invalid gauge form degree: {error}");
            std::process::exit(1);
        });
    let root = std::path::PathBuf::from(args.get(3).unwrap_or_else(|| usage()));
    let deep = args.iter().any(|argument| argument == "--deep");
    let report = eleven_dimensional_gauge::merge_zero_momentum_gauge_composition_artifacts(
        gauge_form_degree,
        &root,
        deep,
    )
    .unwrap_or_else(|error| {
        eprintln!("Failed to merge zero-momentum gauge form {gauge_form_degree}: {error}");
        std::process::exit(2);
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_gauge_zero_classify(args: &[String]) {
    let usage = || {
        eprintln!("Usage: {} adynkra-11d-gauge-zero-classify <root>", args[0]);
        std::process::exit(1);
    };
    let root = std::path::PathBuf::from(args.get(2).unwrap_or_else(|| usage()));
    let report = eleven_dimensional_gauge::classify_zero_momentum_gauge_channel_subsets(&root)
        .unwrap_or_else(|error| {
            eprintln!("Failed to classify zero-momentum gauge channel subsets: {error}");
            std::process::exit(2);
        });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_gauge_first_functional(args: &[String]) {
    let usage = || {
        eprintln!(
            "Usage: {} adynkra-11d-gauge-first-functional <form-degree> <operator-ordinal> <root>",
            args[0]
        );
        std::process::exit(1);
    };
    let gauge_form_degree = args
        .get(2)
        .unwrap_or_else(|| usage())
        .parse::<usize>()
        .unwrap_or_else(|error| {
            eprintln!("Invalid gauge form degree: {error}");
            std::process::exit(1);
        });
    let operator_ordinal = args
        .get(3)
        .unwrap_or_else(|| usage())
        .parse::<usize>()
        .unwrap_or_else(|error| {
            eprintln!("Invalid operator ordinal: {error}");
            std::process::exit(1);
        });
    let root = std::path::PathBuf::from(args.get(4).unwrap_or_else(|| usage()));
    let report =
        eleven_dimensional_gauge::build_and_write_first_momentum_gauge_functional_artifact(
            gauge_form_degree,
            operator_ordinal,
            &root,
        )
        .unwrap_or_else(|error| {
            eprintln!(
                "Failed to build first-momentum gauge functional p={gauge_form_degree}, column={operator_ordinal}: {error}"
            );
            std::process::exit(2);
        });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_gauge_first_functional_stream(args: &[String]) {
    let usage = || {
        eprintln!(
            "Usage: {} adynkra-11d-gauge-first-functional-stream <form-degree> <operator-ordinal> <root>",
            args[0]
        );
        std::process::exit(1);
    };
    let gauge_form_degree = args
        .get(2)
        .unwrap_or_else(|| usage())
        .parse::<usize>()
        .unwrap_or_else(|error| {
            eprintln!("Invalid gauge form degree: {error}");
            std::process::exit(1);
        });
    let operator_ordinal = args
        .get(3)
        .unwrap_or_else(|| usage())
        .parse::<usize>()
        .unwrap_or_else(|error| {
            eprintln!("Invalid operator ordinal: {error}");
            std::process::exit(1);
        });
    let root = std::path::PathBuf::from(args.get(4).unwrap_or_else(|| usage()));
    let report =
        eleven_dimensional_gauge::build_and_write_first_momentum_gauge_stream_functional_artifact(
            gauge_form_degree,
            operator_ordinal,
            &root,
        )
        .unwrap_or_else(|error| {
            eprintln!(
                "Failed to stream first-momentum gauge functional p={gauge_form_degree}, column={operator_ordinal}: {error}"
            );
            std::process::exit(2);
        });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_gauge_first_functional_stream_prefix(args: &[String]) {
    let usage = || {
        eprintln!(
            "Usage: {} adynkra-11d-gauge-first-functional-stream-prefix <form-degree> <operator-ordinal> <parameter-component-count> <root>",
            args[0]
        );
        std::process::exit(1);
    };
    let gauge_form_degree = args
        .get(2)
        .unwrap_or_else(|| usage())
        .parse::<usize>()
        .unwrap_or_else(|error| {
            eprintln!("Invalid gauge form degree: {error}");
            std::process::exit(1);
        });
    let operator_ordinal = args
        .get(3)
        .unwrap_or_else(|| usage())
        .parse::<usize>()
        .unwrap_or_else(|error| {
            eprintln!("Invalid operator ordinal: {error}");
            std::process::exit(1);
        });
    let parameter_component_count = args
        .get(4)
        .unwrap_or_else(|| usage())
        .parse::<usize>()
        .unwrap_or_else(|error| {
            eprintln!("Invalid parameter component count: {error}");
            std::process::exit(1);
        });
    let root = std::path::PathBuf::from(args.get(5).unwrap_or_else(|| usage()));
    let report = eleven_dimensional_gauge::
        build_and_write_first_momentum_gauge_stream_prefix_functional_artifact(
            gauge_form_degree,
            operator_ordinal,
            parameter_component_count,
            &root,
        )
        .unwrap_or_else(|error| {
            eprintln!(
                "Failed to stream first-momentum gauge functional prefix p={gauge_form_degree}, column={operator_ordinal}: {error}"
            );
            std::process::exit(2);
        });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_gauge_first_functional_merge(args: &[String]) {
    let usage = || {
        eprintln!(
            "Usage: {} adynkra-11d-gauge-first-functional-merge <form-degree> <root> <zero-root>",
            args[0]
        );
        std::process::exit(1);
    };
    let gauge_form_degree = args
        .get(2)
        .unwrap_or_else(|| usage())
        .parse::<usize>()
        .unwrap_or_else(|error| {
            eprintln!("Invalid gauge form degree: {error}");
            std::process::exit(1);
        });
    let root = std::path::PathBuf::from(args.get(3).unwrap_or_else(|| usage()));
    let zero_root = std::path::PathBuf::from(args.get(4).unwrap_or_else(|| usage()));
    let report = eleven_dimensional_gauge::merge_first_momentum_gauge_functional_artifacts(
        gauge_form_degree,
        &root,
        &zero_root,
    )
    .unwrap_or_else(|error| {
        eprintln!(
            "Failed to merge first-momentum gauge functionals p={gauge_form_degree}: {error}"
        );
        std::process::exit(2);
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_bridge_verify() {
    let report = eleven_dimensional_bridge::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn cmd_adynkra_11d_level16_coupling_precheck() {
    let report = eleven_dimensional_level16_couplings::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn option_value<'a>(args: &'a [String], option: &str) -> Option<&'a str> {
    args.iter()
        .position(|argument| argument == option)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}

fn read_passed_checkpoint<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Option<T> {
    let payload = std::fs::read(path).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    if value.get("passed").and_then(|passed| passed.as_bool()) != Some(true) {
        return None;
    }
    serde_json::from_value(value).ok()
}

fn cmd_adynkra_11d_level16_coupling_build(args: &[String]) {
    let label = option_value(args, "--label").unwrap_or_else(|| {
        eprintln!("Missing --label");
        std::process::exit(64);
    });
    let report = eleven_dimensional_level16_couplings::build_abstract(label);
    let output = std::path::PathBuf::from(format!(
        "results/adynkra_11d_level16_coupling_{label}_abstract.json"
    ));
    eleven_dimensional_level16_couplings::write_atomic_json(&output, &report, report.passed)
        .unwrap_or_else(|error| {
            eprintln!("Failed to checkpoint {}: {error}", output.display());
            std::process::exit(2);
        });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_level16_coupling_verify(args: &[String]) {
    if args.iter().any(|argument| argument == "--all") {
        let resume = args.iter().any(|argument| argument == "--resume");
        let mut copies_by_label = std::collections::BTreeMap::<&str, Vec<usize>>::new();
        for fixture in eleven_dimensional_spinor_bridge_kernels::level16_fixtures() {
            copies_by_label
                .entry(fixture.dynkin_label)
                .or_default()
                .push(fixture.copy);
        }
        let jobs = copies_by_label.into_iter().collect::<Vec<_>>();
        let memory_budget_gib = std::env::var("ADINKRA_LEVEL16_RAM_GIB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(48);
        let estimated_memory_gib_per_worker = std::env::var("ADINKRA_LEVEL16_WORKER_GIB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(10)
            .max(1);
        let requested_workers = std::env::var("ADINKRA_LEVEL16_WORKERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(4)
            .max(1);
        let execution_workers = requested_workers
            .min(memory_budget_gib / estimated_memory_gib_per_worker)
            .min(jobs.len())
            .max(1);
        eprintln!(
            "level-16 coupling workers={execution_workers}, memory budget={memory_budget_gib} GiB, estimate={estimated_memory_gib_per_worker} GiB/worker"
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(execution_workers)
            .build()
            .unwrap();
        use rayon::prelude::*;
        let completed = pool.install(|| {
            jobs.par_iter()
                .map(|(label, copies)| {
                    let abstract_output = std::path::PathBuf::from(format!(
                        "results/adynkra_11d_level16_coupling_{label}_abstract.json"
                    ));
                    let saved_abstract = resume
                        .then(|| read_passed_checkpoint(&abstract_output))
                        .flatten();
                    let abstract_was_reused = saved_abstract.is_some();
                    let abstract_report = saved_abstract.unwrap_or_else(|| {
                        eleven_dimensional_level16_couplings::build_abstract(label)
                    });
                    if !abstract_was_reused {
                        eleven_dimensional_level16_couplings::write_atomic_json(
                            &abstract_output,
                            &abstract_report,
                            abstract_report.passed,
                        )
                        .unwrap();
                    }
                    eprintln!("certified abstract coupling {label}");
                    let copy_reports = copies
                        .iter()
                        .map(|copy| {
                            let copy_output = std::path::PathBuf::from(format!(
                                "results/adynkra_11d_level16_coupling_{label}_copy{copy}.json"
                            ));
                            let saved_copy = resume
                                .then(|| read_passed_checkpoint(&copy_output))
                                .flatten();
                            let copy_was_reused = saved_copy.is_some();
                            let copy_report = saved_copy.unwrap_or_else(|| {
                                eleven_dimensional_level16_couplings::verify_copy_with_abstract(
                                    &abstract_report,
                                    *copy,
                                )
                            });
                            if !copy_was_reused {
                                eleven_dimensional_level16_couplings::write_atomic_json(
                                    &copy_output,
                                    &copy_report,
                                    copy_report.passed,
                                )
                                .unwrap();
                            }
                            eprintln!("certified embedded coupling {label} copy {copy}");
                            copy_report
                        })
                        .collect::<Vec<_>>();
                    (abstract_report, copy_reports)
                })
                .collect::<Vec<_>>()
        });
        let abstract_couplings = completed.iter().map(|(report, _)| report.clone()).collect();
        let embedded_copies = completed
            .into_iter()
            .flat_map(|(_, reports)| reports)
            .collect();
        let report = eleven_dimensional_level16_couplings::summarize_all(
            abstract_couplings,
            embedded_copies,
            execution_workers,
            memory_budget_gib,
            estimated_memory_gib_per_worker,
            resume,
        );
        let output = std::path::PathBuf::from("results/adynkra_11d_level16_couplings_all.json");
        eleven_dimensional_level16_couplings::write_atomic_json(&output, &report, report.passed)
            .unwrap_or_else(|error| {
                eprintln!("Failed to checkpoint {}: {error}", output.display());
                std::process::exit(2);
            });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        if !report.passed {
            std::process::exit(2);
        }
        return;
    }
    let label = option_value(args, "--label").unwrap_or_else(|| {
        eprintln!("Missing --label or --all");
        std::process::exit(64);
    });
    let copy = option_value(args, "--copy")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| {
            eprintln!("Missing or invalid --copy");
            std::process::exit(64);
        });
    let output = std::path::PathBuf::from(format!(
        "results/adynkra_11d_level16_coupling_{label}_copy{copy}.json"
    ));
    if args.iter().any(|argument| argument == "--resume") && output.exists() {
        let payload = std::fs::read_to_string(&output).unwrap();
        if serde_json::from_str::<serde_json::Value>(&payload)
            .ok()
            .and_then(|value| value.get("passed").and_then(|passed| passed.as_bool()))
            == Some(true)
        {
            print!("{payload}");
            return;
        }
    }
    let report = eleven_dimensional_level16_couplings::verify_copy(label, copy);
    eleven_dimensional_level16_couplings::write_atomic_json(&output, &report, report.passed)
        .unwrap_or_else(|error| {
            eprintln!("Failed to checkpoint {}: {error}", output.display());
            std::process::exit(2);
        });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_level17_hook_precheck() {
    let report = eleven_dimensional_level16_couplings::verify_hook_precheck();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_level17_hook_build(args: &[String]) {
    let label = option_value(args, "--label").unwrap_or_else(|| {
        eprintln!("Missing --label");
        std::process::exit(64);
    });
    let report = eleven_dimensional_level16_couplings::build_hook_abstract(label);
    let output = std::path::PathBuf::from(format!(
        "results/adynkra_11d_level17_hook_{label}_abstract.json"
    ));
    eleven_dimensional_level16_couplings::write_atomic_json(&output, &report, report.passed)
        .unwrap_or_else(|error| {
            eprintln!("Failed to checkpoint {}: {error}", output.display());
            std::process::exit(2);
        });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_level17_hook_verify(args: &[String]) {
    let resume = args.iter().any(|argument| argument == "--resume");
    if !args.iter().any(|argument| argument == "--all") {
        let label = option_value(args, "--label").unwrap_or_else(|| {
            eprintln!("Missing --label or --all");
            std::process::exit(64);
        });
        let copy = option_value(args, "--copy")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or_else(|| {
                eprintln!("Missing or invalid --copy");
                std::process::exit(64);
            });
        let abstract_report = eleven_dimensional_level16_couplings::build_hook_abstract(label);
        let report = eleven_dimensional_level16_couplings::verify_hook_copy_with_abstract(
            &abstract_report,
            copy,
        );
        let output = std::path::PathBuf::from(format!(
            "results/adynkra_11d_level17_hook_{label}_copy{copy}.json"
        ));
        eleven_dimensional_level16_couplings::write_atomic_json(&output, &report, report.passed)
            .unwrap();
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        if !report.passed {
            std::process::exit(2);
        }
        return;
    }
    let jobs = eleven_dimensional_level16_couplings::hook_copy_manifest()
        .into_iter()
        .collect::<Vec<_>>();
    let memory_budget_gib = std::env::var("ADINKRA_LEVEL17_RAM_GIB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(48);
    let estimated_memory_gib_per_worker = std::env::var("ADINKRA_LEVEL17_WORKER_GIB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10)
        .max(1);
    let requested_workers = std::env::var("ADINKRA_LEVEL17_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1);
    let execution_workers = requested_workers
        .min(memory_budget_gib / estimated_memory_gib_per_worker)
        .min(jobs.len())
        .max(1);
    eprintln!(
        "level-17 hook workers={execution_workers}, memory budget={memory_budget_gib} GiB, estimate={estimated_memory_gib_per_worker} GiB/worker"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(execution_workers)
        .build()
        .unwrap();
    use rayon::prelude::*;
    let completed = pool.install(|| {
        jobs.par_iter()
            .map(|(label, copies)| {
                let abstract_output = std::path::PathBuf::from(format!(
                    "results/adynkra_11d_level17_hook_{label}_abstract.json"
                ));
                let saved_abstract = resume
                    .then(|| read_passed_checkpoint(&abstract_output))
                    .flatten();
                let abstract_was_reused = saved_abstract.is_some();
                let abstract_report = saved_abstract.unwrap_or_else(|| {
                    eleven_dimensional_level16_couplings::build_hook_abstract(label)
                });
                if !abstract_was_reused {
                    eleven_dimensional_level16_couplings::write_atomic_json(
                        &abstract_output,
                        &abstract_report,
                        abstract_report.passed,
                    )
                    .unwrap();
                }
                eprintln!("certified abstract hook coupling {label}");
                let copy_reports = copies
                    .iter()
                    .map(|copy| {
                        let copy_output = std::path::PathBuf::from(format!(
                            "results/adynkra_11d_level17_hook_{label}_copy{copy}.json"
                        ));
                        let saved_copy = resume
                            .then(|| read_passed_checkpoint(&copy_output))
                            .flatten();
                        let copy_was_reused = saved_copy.is_some();
                        let copy_report = saved_copy.unwrap_or_else(|| {
                            eleven_dimensional_level16_couplings::verify_hook_copy_with_abstract(
                                &abstract_report,
                                *copy,
                            )
                        });
                        if !copy_was_reused {
                            eleven_dimensional_level16_couplings::write_atomic_json(
                                &copy_output,
                                &copy_report,
                                copy_report.passed,
                            )
                            .unwrap();
                        }
                        eprintln!("certified hook coupling {label} copy {copy}");
                        copy_report
                    })
                    .collect::<Vec<_>>();
                (abstract_report, copy_reports)
            })
            .collect::<Vec<_>>()
    });
    let abstract_couplings = completed.iter().map(|(report, _)| report.clone()).collect();
    let embedded_copies = completed
        .into_iter()
        .flat_map(|(_, reports)| reports)
        .collect();
    let report = eleven_dimensional_level16_couplings::summarize_hooks(
        abstract_couplings,
        embedded_copies,
        execution_workers,
        memory_budget_gib,
        estimated_memory_gib_per_worker,
        resume,
    );
    let output = std::path::PathBuf::from("results/adynkra_11d_level17_hook_couplings_all.json");
    eleven_dimensional_level16_couplings::write_atomic_json(&output, &report, report.passed)
        .unwrap();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_level17_derivative_matrix() {
    let report = eleven_dimensional_level16_couplings::build_level17_derivative_matrix();
    let output = std::path::PathBuf::from("results/adynkra_11d_level17_derivative_matrix.json");
    eleven_dimensional_level16_couplings::write_atomic_json(&output, &report, report.passed)
        .unwrap_or_else(|error| {
            eprintln!("Failed to checkpoint {}: {error}", output.display());
            std::process::exit(2);
        });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_first_momentum_precheck() {
    let report = eleven_dimensional_spinor_bridge::verify_first_momentum_source_precheck();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_first_momentum_kernel_verify() {
    let report = eleven_dimensional_spinor_bridge_kernels::verify_first_momentum_kernels();
    let output = std::path::PathBuf::from("results/adynkra_11d_first_momentum_kernels.json");
    eleven_dimensional_level16_couplings::write_atomic_json(&output, &report, report.passed)
        .unwrap_or_else(|error| {
            eprintln!("Failed to checkpoint {}: {error}", output.display());
            std::process::exit(2);
        });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_first_momentum_coupling_build(args: &[String]) {
    let source = option_value(args, "--source").unwrap_or_else(|| {
        eprintln!("Missing --source");
        std::process::exit(64);
    });
    let target = option_value(args, "--target").unwrap_or_else(|| {
        eprintln!("Missing --target");
        std::process::exit(64);
    });
    let report =
        eleven_dimensional_level16_couplings::build_first_momentum_abstract(source, target);
    let output = std::path::PathBuf::from(format!(
        "results/adynkra_11d_first_momentum_{target}_from_{source}_abstract.json"
    ));
    eleven_dimensional_level16_couplings::write_atomic_json(&output, &report, report.passed)
        .unwrap_or_else(|error| {
            eprintln!("Failed to checkpoint {}: {error}", output.display());
            std::process::exit(2);
        });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_first_momentum_coupling_verify(args: &[String]) {
    let resume = args.iter().any(|argument| argument == "--resume");
    if !args.iter().any(|argument| argument == "--all") {
        let source = option_value(args, "--source").unwrap_or_else(|| {
            eprintln!("Missing --source or --all");
            std::process::exit(64);
        });
        let target = option_value(args, "--target").unwrap_or_else(|| {
            eprintln!("Missing --target");
            std::process::exit(64);
        });
        let copy = option_value(args, "--copy")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or_else(|| {
                eprintln!("Missing or invalid --copy");
                std::process::exit(64);
            });
        let abstract_report =
            eleven_dimensional_level16_couplings::build_first_momentum_abstract(source, target);
        let report = eleven_dimensional_level16_couplings::verify_first_momentum_copy_with_abstract(
            &abstract_report,
            copy,
        );
        let output = std::path::PathBuf::from(format!(
            "results/adynkra_11d_first_momentum_{target}_from_{source}_copy{copy}.json"
        ));
        eleven_dimensional_level16_couplings::write_atomic_json(&output, &report, report.passed)
            .unwrap();
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        if !report.passed {
            std::process::exit(2);
        }
        return;
    }

    let jobs = eleven_dimensional_level16_couplings::first_momentum_copy_manifest()
        .into_iter()
        .collect::<Vec<_>>();
    let memory_budget_gib = std::env::var("ADINKRA_FIRST_MOMENTUM_RAM_GIB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(48);
    let estimated_memory_gib_per_worker = std::env::var("ADINKRA_FIRST_MOMENTUM_WORKER_GIB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10)
        .max(1);
    let requested_workers = std::env::var("ADINKRA_FIRST_MOMENTUM_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1);
    let execution_workers = requested_workers
        .min(memory_budget_gib / estimated_memory_gib_per_worker)
        .min(jobs.len())
        .max(1);
    eprintln!(
        "first-momentum workers={execution_workers}, memory budget={memory_budget_gib} GiB, estimate={estimated_memory_gib_per_worker} GiB/worker"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(execution_workers)
        .build()
        .unwrap();
    use rayon::prelude::*;
    let completed = pool.install(|| {
        jobs.par_iter()
            .map(|((source, target), copies)| {
                let abstract_output = std::path::PathBuf::from(format!(
                    "results/adynkra_11d_first_momentum_{target}_from_{source}_abstract.json"
                ));
                let saved_abstract = resume
                    .then(|| read_passed_checkpoint(&abstract_output))
                    .flatten();
                let abstract_was_reused = saved_abstract.is_some();
                let abstract_report = saved_abstract.unwrap_or_else(|| {
                    eleven_dimensional_level16_couplings::build_first_momentum_abstract(
                        source, target,
                    )
                });
                if !abstract_was_reused {
                    eleven_dimensional_level16_couplings::write_atomic_json(
                        &abstract_output,
                        &abstract_report,
                        abstract_report.passed,
                    )
                    .unwrap();
                }
                eprintln!("certified abstract first-momentum coupling {source} -> {target}");
                let copy_reports = copies
                    .iter()
                    .map(|copy| {
                        let copy_output = std::path::PathBuf::from(format!(
                            "results/adynkra_11d_first_momentum_{target}_from_{source}_copy{copy}.json"
                        ));
                        let saved_copy = resume
                            .then(|| read_passed_checkpoint(&copy_output))
                            .flatten();
                        let copy_was_reused = saved_copy.is_some();
                        let copy_report = saved_copy.unwrap_or_else(|| {
                            eleven_dimensional_level16_couplings::verify_first_momentum_copy_with_abstract(
                                &abstract_report,
                                *copy,
                            )
                        });
                        if !copy_was_reused {
                            eleven_dimensional_level16_couplings::write_atomic_json(
                                &copy_output,
                                &copy_report,
                                copy_report.passed,
                            )
                            .unwrap();
                        }
                        eprintln!(
                            "certified embedded first-momentum coupling {source} copy {copy} -> {target}"
                        );
                        copy_report
                    })
                    .collect::<Vec<_>>();
                (abstract_report, copy_reports)
            })
            .collect::<Vec<_>>()
    });
    let abstract_couplings = completed.iter().map(|(report, _)| report.clone()).collect();
    let embedded_maps = completed
        .into_iter()
        .flat_map(|(_, reports)| reports)
        .collect();
    let report = eleven_dimensional_level16_couplings::summarize_first_momentum(
        abstract_couplings,
        embedded_maps,
        execution_workers,
        memory_budget_gib,
        estimated_memory_gib_per_worker,
        resume,
    );
    let output = std::path::PathBuf::from("results/adynkra_11d_first_momentum_couplings_all.json");
    eleven_dimensional_level16_couplings::write_atomic_json(&output, &report, report.passed)
        .unwrap_or_else(|error| {
            eprintln!("Failed to checkpoint {}: {error}", output.display());
            std::process::exit(2);
        });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_first_momentum_target_verify() {
    let report = eleven_dimensional_bridge::verify_first_momentum_target_couplings();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    let output =
        std::path::PathBuf::from("results/adynkra_11d_first_momentum_target_couplings.json");
    eleven_dimensional_level16_couplings::write_atomic_json(&output, &report, report.passed)
        .unwrap_or_else(|error| {
            eprintln!("Failed to checkpoint {}: {error}", output.display());
            std::process::exit(2);
        });
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_joint_compatibility() {
    let report = eleven_dimensional_level16_couplings::build_joint_compatibility_matrix();
    let output = std::path::PathBuf::from("results/adynkra_11d_joint_compatibility_matrix.json");
    eleven_dimensional_level16_couplings::write_atomic_json(&output, &report, report.passed)
        .unwrap_or_else(|error| {
            eprintln!("Failed to checkpoint {}: {error}", output.display());
            std::process::exit(2);
        });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_joint_manifest() {
    let specs = eleven_dimensional_level16_couplings::joint_column_specs();
    println!("{}", serde_json::to_string_pretty(&specs).unwrap());
}

fn cmd_adynkra_11d_joint_column(args: &[String]) {
    let ordinal = args
        .get(2)
        .unwrap_or_else(|| {
            eprintln!(
                "Usage: {} adynkra-11d-joint-column <ordinal> <root>",
                args[0]
            );
            std::process::exit(1);
        })
        .parse::<usize>()
        .unwrap_or_else(|error| {
            eprintln!("Invalid joint column ordinal: {error}");
            std::process::exit(1);
        });
    let root = std::path::PathBuf::from(args.get(3).unwrap_or_else(|| {
        eprintln!(
            "Usage: {} adynkra-11d-joint-column <ordinal> <root>",
            args[0]
        );
        std::process::exit(1);
    }));
    let report =
        eleven_dimensional_level16_couplings::build_and_write_joint_column_artifact(ordinal, &root)
            .unwrap_or_else(|error| {
                eprintln!("Failed to build joint column {ordinal}: {error}");
                std::process::exit(2);
            });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn cmd_adynkra_11d_joint_merge(args: &[String]) {
    use std::io::Write;

    let root = std::path::PathBuf::from(args.get(2).unwrap_or_else(|| {
        eprintln!("Usage: {} adynkra-11d-joint-merge <root> [--deep]", args[0]);
        std::process::exit(1);
    }));
    let deep = args.iter().any(|argument| argument == "--deep");
    let report = eleven_dimensional_level16_couplings::merge_joint_column_artifacts(&root, deep)
        .unwrap_or_else(|error| {
            eprintln!("Failed to merge joint column artifacts: {error}");
            std::process::exit(2);
        });
    let merge_root = root.join("merge");
    std::fs::create_dir_all(&merge_root).unwrap();
    let final_path = merge_root.join("joint-compatibility.json");
    let temporary_path =
        merge_root.join(format!(".joint-compatibility.{}.tmp", std::process::id()));
    let payload = serde_json::to_vec_pretty(&report).unwrap();
    let mut output = std::fs::File::create(&temporary_path).unwrap();
    output.write_all(&payload).unwrap();
    output.write_all(b"\n").unwrap();
    output.sync_all().unwrap();
    std::fs::rename(&temporary_path, &final_path).unwrap();
    std::fs::File::open(&merge_root)
        .unwrap()
        .sync_all()
        .unwrap();
    println!("{}", String::from_utf8(payload).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_spinor_bridge_verify() {
    let report = eleven_dimensional_spinor_bridge::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_adynkra_11d_spinor_kernel_verify() {
    let report = eleven_dimensional_spinor_bridge_kernels::verify();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.passed {
        std::process::exit(2);
    }
}

fn cmd_export_3d_assets(args: &[String]) {
    let catalog = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("adinkra_codes_n16.json");
    let output = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("visualizer/adinkra_dashing");
    let manifest = viz_export::export(catalog, output);
    println!("{}", serde_json::to_string_pretty(&manifest).unwrap());
}

// ---------------------------------------------------------------------------
// enumerate
// ---------------------------------------------------------------------------

fn cmd_enumerate(args: &[String]) {
    let n = parse_usize_arg(args, 2, "enumerate <n>");

    eprintln!("Enumerating all doubly-even codes of length {}...", n);
    let start = Instant::now();
    let codes = enumerate_codes(n);
    let elapsed = start.elapsed();
    eprintln!(
        "Found {} codes (before dedup) in {:?}",
        codes.len(),
        elapsed
    );

    let start = Instant::now();
    let unique = deduplicate(codes);
    let elapsed = start.elapsed();
    eprintln!(
        "Deduplicated to {} equivalence classes in {:?}",
        unique.len(),
        elapsed
    );

    println!();
    println!(
        "Doubly-even codes of length {} ({} equivalence classes):",
        n,
        unique.len()
    );
    println!();

    for (i, code) in unique.iter().enumerate() {
        let d = code.min_distance();
        let decomp = if code.k() > 1 {
            if is_decomposable(code) {
                " [decomposable]"
            } else {
                " [indecomposable]"
            }
        } else {
            ""
        };
        println!("  [{}] [{},{},{}]{}", i, code.n, code.k(), d, decomp);
        for (j, &row) in code.generators.iter().enumerate() {
            let bits: String = (0..n)
                .map(|col| if row & (1 << col) != 0 { '1' } else { '0' })
                .collect();
            println!("       g{}: {}", j, bits);
        }
    }
}

// ---------------------------------------------------------------------------
// count
// ---------------------------------------------------------------------------

fn cmd_count(args: &[String]) {
    let max_n = parse_usize_arg(args, 2, "count <max_n>");

    println!(
        "{:>3} | {:>12} | {:>12} | {:>10} | {:>5} | {:>13}",
        "N", "Raw codes", "Equiv classes", "Nontrivial", "Max k", "Indecomposable"
    );
    println!("{}", "-".repeat(72));

    for n in 1..=max_n {
        let start = Instant::now();
        let codes = enumerate_codes(n);
        let raw_count = codes.len();
        let unique = deduplicate(codes);
        let elapsed = start.elapsed();

        let nontrivial: Vec<&DoublyEvenCode> = unique.iter().filter(|c| c.k() > 0).collect();
        let max_k = unique.iter().map(|c| c.k()).max().unwrap_or(0);
        let indecomposable = nontrivial.iter().filter(|c| !is_decomposable(c)).count();

        println!(
            "{:>3} | {:>12} | {:>12} | {:>10} | {:>5} | {:>13}",
            n,
            raw_count,
            unique.len(),
            nontrivial.len(),
            max_k,
            indecomposable
        );
        eprintln!("  N={} completed in {:?}", n, elapsed);
    }
}

// ---------------------------------------------------------------------------
// eval
// ---------------------------------------------------------------------------

fn cmd_eval(args: &[String]) {
    let held_out_n = parse_usize_arg(args, 2, "eval <held_out_n>");
    let max_cand = if args.len() > 3 {
        args[3].parse::<usize>().unwrap_or(500)
    } else {
        500
    };

    let results = eval::evaluate_held_out(held_out_n, max_cand);
    println!();
    eval::print_results(&results);
}

// ---------------------------------------------------------------------------
// eval-all
// ---------------------------------------------------------------------------

fn cmd_eval_all(args: &[String]) {
    let max_cand = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(500)
    } else {
        500
    };

    let mut all_results = Vec::new();
    for n in 4..=10 {
        eprintln!();
        let results = eval::evaluate_held_out(n, max_cand);
        all_results.extend(results);
    }

    println!();
    println!("========== FULL EVAL-ALL SUMMARY ==========");
    println!();
    eval::print_results(&all_results);
}

// ---------------------------------------------------------------------------
// invariants
// ---------------------------------------------------------------------------

fn cmd_invariants(args: &[String]) {
    let n = parse_usize_arg(args, 2, "invariants <n>");

    eprintln!("Computing invariants for all codes of length {}...", n);
    let start = Instant::now();
    let codes = enumerate_codes(n);
    let unique = deduplicate(codes);
    let elapsed = start.elapsed();
    eprintln!(
        "Found {} equivalence classes in {:?}",
        unique.len(),
        elapsed
    );

    println!();
    println!(
        "Invariants for doubly-even codes of length {} ({} classes):",
        n,
        unique.len()
    );
    println!();

    println!(
        "{:>4} | {:>3} | {:>3} | {:>6} | {:>13} | {}",
        "Idx", "n", "k", "d_min", "Decomposable", "Weight enumerator"
    );
    println!("{}", "-".repeat(80));

    for (i, code) in unique.iter().enumerate() {
        let inv = compute_invariants(code);
        let d = code.min_distance();
        let decomp = if code.k() <= 1 {
            "n/a".to_string()
        } else if is_decomposable(code) {
            "yes".to_string()
        } else {
            "no".to_string()
        };

        // Format weight enumerator compactly: only show nonzero entries
        let we: Vec<String> = inv
            .weight_enumerator
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(w, count)| format!("{}:{}", w, count))
            .collect();

        println!(
            "{:>4} | {:>3} | {:>3} | {:>6} | {:>13} | {}",
            i,
            inv.n,
            inv.k,
            d,
            decomp,
            we.join(" ")
        );
    }
}

// ---------------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------------

fn cmd_validate() {
    println!("=== Doubly-Even Code Enumeration Validation ===");
    println!();
    println!("Enumerating codes for N=4 through N=8 and comparing against known results.");
    println!("Reference: Doran et al., arXiv:0806.0050 (doubly-even codes and Adinkra graphs).");
    println!();

    println!(
        "{:>3} | {:>10} | {:>12} | {:>10} | {:>5} | {:>13}",
        "N", "Raw codes", "Total classes", "Nontrivial", "Max k", "Indecomposable"
    );
    println!("{}", "-".repeat(68));

    let mut all_ok = true;

    for n in 4..=8 {
        let start = Instant::now();
        let codes = enumerate_codes(n);
        let raw_count = codes.len();
        let unique = deduplicate(codes);
        let elapsed = start.elapsed();

        let nontrivial: Vec<&DoublyEvenCode> = unique.iter().filter(|c| c.k() > 0).collect();
        let max_k = unique.iter().map(|c| c.k()).max().unwrap_or(0);
        let indecomposable = nontrivial.iter().filter(|c| !is_decomposable(c)).count();

        println!(
            "{:>3} | {:>10} | {:>12} | {:>10} | {:>5} | {:>13}",
            n,
            raw_count,
            unique.len(),
            nontrivial.len(),
            max_k,
            indecomposable
        );
        eprintln!("  N={} completed in {:?}", n, elapsed);

        // Sanity checks
        // N=4: should have at least 1 nontrivial code (the [4,1,4] repetition-like code)
        if n == 4 && nontrivial.is_empty() {
            eprintln!("  FAIL: N=4 should have at least 1 nontrivial code");
            all_ok = false;
        }

        // N=8: the extended Hamming [8,4,4] code should be present
        if n == 8 {
            let has_8_4 = nontrivial
                .iter()
                .any(|c| c.k() == 4 && c.min_distance() == 4);
            if has_8_4 {
                eprintln!("  OK: found [8,4,4] extended Hamming code at N=8");
            } else {
                eprintln!("  FAIL: did not find [8,4,4] extended Hamming code at N=8");
                all_ok = false;
            }

            // Check: max k at N=8 should be 4 (the Hamming code)
            if max_k < 4 {
                eprintln!("  FAIL: max k at N=8 is {}, expected at least 4", max_k);
                all_ok = false;
            }
        }

        // Every code should actually be doubly-even
        for code in &unique {
            if !code.is_doubly_even() {
                eprintln!(
                    "  FAIL: found non-doubly-even code at N={}: {:?}",
                    n, code.generators
                );
                all_ok = false;
            }
        }
    }

    println!();

    // Additional validation: verify the Hamming code directly
    println!("--- Direct Hamming [8,4,4] Verification ---");
    let hamming = DoublyEvenCode::new(8, vec![0b11100001, 0b11010010, 0b10110100, 0b01111000]);
    let is_de = hamming.is_doubly_even();
    let d = hamming.min_distance();
    let we = hamming.weight_enumerator();
    println!(
        "  Hamming [8,4,4]: doubly_even={}, k={}, d_min={}, weight_enum={:?}",
        is_de,
        hamming.k(),
        d,
        we
    );
    if is_de && hamming.k() == 4 && d == 4 {
        eprintln!("  OK: Hamming code validates correctly");
    } else {
        eprintln!("  FAIL: Hamming code validation failed");
        all_ok = false;
    }

    // Check expected weight enumerator for [8,4,4]:
    // A_0=1, A_4=14, A_8=1 => total 16 codewords
    let total_cw: usize = we.iter().sum();
    if total_cw == 16 && we[0] == 1 && we[4] == 14 && we[8] == 1 {
        println!(
            "  Weight enumerator: A_0={}, A_4={}, A_8={} (correct for [8,4,4])",
            we[0], we[4], we[8]
        );
    } else {
        eprintln!("  WARN: unexpected weight enumerator for Hamming code");
        all_ok = false;
    }

    println!();
    if all_ok {
        println!("All validation checks passed.");
    } else {
        println!("Some validation checks FAILED. See messages above.");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------

fn cmd_search(args: &[String]) {
    let mut config = search::SearchConfig::default();

    if args.len() > 2 {
        config.target_n = args[2].parse::<usize>().unwrap_or(16);
    }
    if args.len() > 3 {
        config.evo_population = args[3].parse::<usize>().unwrap_or(500);
    }
    if args.len() > 4 {
        config.evo_generations = args[4].parse::<usize>().unwrap_or(500);
    }

    search::search(&config);
}

// ---------------------------------------------------------------------------
// saturate
// ---------------------------------------------------------------------------

fn cmd_saturate(args: &[String]) {
    let n = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(16)
    } else {
        16
    };
    let batch_size = if args.len() > 3 {
        args[3].parse::<usize>().unwrap_or(5000)
    } else {
        5000
    };
    let max_batches = if args.len() > 4 {
        args[4].parse::<usize>().unwrap_or(500)
    } else {
        500
    };

    search::saturate(n, batch_size, max_batches);
}

// ---------------------------------------------------------------------------
// validate-miller
// ---------------------------------------------------------------------------

fn cmd_validate_miller(args: &[String]) {
    let n = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(16)
    } else {
        16
    };

    search::validate_miller(n);
}

// ---------------------------------------------------------------------------
// pipeline
// ---------------------------------------------------------------------------

fn cmd_pipeline(args: &[String]) {
    let json_path = if args.len() > 2 {
        args[2].as_str()
    } else {
        "adinkra_codes_n16.json"
    };

    let output = pipeline::run_pipeline(json_path);
    let json = serde_json::to_string_pretty(&output).expect("Failed to serialize output");
    println!("{}", json);
}

fn cmd_pipeline_k(args: &[String]) {
    let k = parse_usize_arg(args, 2, "pipeline-k <k> [json]");
    let json_path = if args.len() > 3 {
        args[3].as_str()
    } else {
        "adinkra_codes_n16.json"
    };

    let output = pipeline::run_pipeline_k(json_path, k);
    let json = serde_json::to_string_pretty(&output).expect("Failed to serialize output");
    println!("{}", json);
}

fn cmd_decompose_k(args: &[String], allow_disk: bool) {
    let k = parse_usize_arg(args, 2, "decompose-k <k> [json] [--disk] [--f64]");
    // First non-flag positional after k is the json path; flags may appear anywhere.
    let disk_f64 = args.iter().any(|a| a == "--f64");
    // --f64 implies the disk path (the f64 store is too large for RAM).
    let allow_disk = allow_disk || disk_f64 || args.iter().any(|a| a == "--disk");
    let json_path = args
        .iter()
        .skip(3)
        .find(|a| !a.starts_with("--"))
        .map(|s| s.as_str())
        .unwrap_or("adinkra_codes_n16.json");

    let output = pipeline::run_decompose_k_mode(json_path, k, allow_disk, disk_f64);
    let json = serde_json::to_string_pretty(&output).expect("Failed to serialize output");
    println!("{}", json);
}

fn cmd_decompose_structure(args: &[String]) {
    let k = parse_usize_arg(args, 2, "decompose-structure <k> [json]");
    let json_path = if args.len() > 3 {
        args[3].as_str()
    } else {
        "adinkra_codes_n16.json"
    };
    pipeline::run_decompose_structure(json_path, k);
}

fn cmd_lift_scan(args: &[String]) {
    let k = parse_usize_arg(args, 2, "lift-scan <k> [json]");
    let json_path = if args.len() > 3 {
        args[3].as_str()
    } else {
        "adinkra_codes_n16.json"
    };
    pipeline::run_lift_scan(json_path, k);
}

fn cmd_lift_construct(args: &[String]) {
    let k = parse_usize_arg(args, 2, "lift-construct <k> [json]");
    let json_path = if args.len() > 3 {
        args[3].as_str()
    } else {
        "adinkra_codes_n16.json"
    };
    pipeline::run_lift_construct(json_path, k);
}

fn cmd_worldsheet_verify(args: &[String]) {
    let catalog_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("adinkra_codes_n16.json");
    let certificate_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("results/worldsheet_spin_sum_witnesses_n16.json");
    pipeline::run_worldsheet_certificate_verification(catalog_path, certificate_path);
}

fn cmd_enhance_scan(args: &[String]) {
    let k = parse_usize_arg(args, 2, "enhance-scan <k> [json]");
    let json_path = if args.len() > 3 {
        args[3].as_str()
    } else {
        "adinkra_codes_n16.json"
    };
    pipeline::run_enhance_scan(json_path, k);
}

fn cmd_central_charge(args: &[String]) {
    let k = parse_usize_arg(args, 2, "central-charge <k> [json]");
    let json_path = if args.len() > 3 {
        args[3].as_str()
    } else {
        "adinkra_codes_n16.json"
    };
    pipeline::run_central_charge(json_path, k);
}

fn cmd_lift_attack(args: &[String]) {
    let code_index = parse_usize_arg(args, 2, "lift-search <code_index> [iters] [seed] [json]");
    let iters = if args.len() > 3 {
        args[3].parse().unwrap_or(20000)
    } else {
        20000
    };
    let seed = if args.len() > 4 {
        args[4].parse().unwrap_or(1)
    } else {
        1
    };
    let json_path = if args.len() > 5 {
        args[5].as_str()
    } else {
        "adinkra_codes_n16.json"
    };
    pipeline::run_lift_attack(json_path, code_index, iters, seed);
}

fn cmd_q_scan(args: &[String]) {
    let k = parse_usize_arg(args, 2, "q-scan <k> [json] [--no-struct]");
    // --no-struct skips the commutant_dim/Schur label (Q + support only, fast).
    let compute_struct = !args.iter().any(|a| a == "--no-struct");
    let json_path = args
        .iter()
        .skip(3)
        .find(|a| !a.starts_with("--"))
        .map(|s| s.as_str())
        .unwrap_or("adinkra_codes_n16.json");
    pipeline::run_q_scan(json_path, k, compute_struct);
}

fn cmd_decompose_audit(args: &[String]) {
    let k = parse_usize_arg(args, 2, "decompose-audit <k> <sample_reps> [json]");
    let sample = if args.len() > 3 {
        args[3].parse::<usize>().unwrap_or(64)
    } else {
        64
    };
    let json_path = if args.len() > 4 {
        args[4].as_str()
    } else {
        "adinkra_codes_n16.json"
    };
    pipeline::run_decompose_audit(json_path, k, sample);
}

fn cmd_decompose_probe(args: &[String]) {
    let k = parse_usize_arg(args, 2, "decompose-probe <k> <num_reps> [json]");
    let num = if args.len() > 3 {
        args[3].parse::<usize>().unwrap_or(8)
    } else {
        8
    };
    let json_path = if args.len() > 4 {
        args[4].as_str()
    } else {
        "adinkra_codes_n16.json"
    };
    pipeline::run_decompose_probe(json_path, k, num);
}

fn cmd_cls_g_full_build(args: &[String]) {
    use four_color::gmatrix_full::{Side, run_build};
    let side = match args.get(2).map(String::as_str).unwrap_or("L") {
        "L" | "l" => Side::L,
        "R" | "r" => Side::R,
        other => {
            eprintln!(
                "side must be L or R, got '{other}'. Usage: {} cls-g-full-build [side] [blocks] [threads] [cap] [json]",
                args[0]
            );
            std::process::exit(1);
        }
    };
    let m = args
        .get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    if !(1..=3).contains(&m) {
        eprintln!("blocks must be 1..=3, got {m}");
        std::process::exit(1);
    }
    let threads = args
        .get(4)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
    let cap = args.get(5).and_then(|s| s.parse::<u64>().ok());
    let default_path = format!(
        "results/four_color_cls_gmatrix_full_{}_{}blocks.json",
        side.name(),
        m
    );
    let path = args.get(6).map(String::as_str).unwrap_or(&default_path);
    run_build(side, m, threads, cap, path);
}

fn cmd_cls_g_full_verify(args: &[String]) {
    use four_color::gmatrix_full::{Side, run_verify};
    let side = match args.get(2).map(String::as_str).unwrap_or("L") {
        "L" | "l" => Side::L,
        "R" | "r" => Side::R,
        other => {
            eprintln!(
                "side must be L or R, got '{other}'. Usage: {} cls-g-full-verify [side] [blocks] [json]",
                args[0]
            );
            std::process::exit(1);
        }
    };
    let m = args
        .get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    let default_path = format!(
        "results/four_color_cls_gmatrix_full_{}_{}blocks.json",
        side.name(),
        m
    );
    let path = args.get(4).map(String::as_str).unwrap_or(&default_path);
    if !run_verify(side, m, path) {
        std::process::exit(2);
    }
}

fn cmd_cls_g_csp_build(args: &[String]) {
    use four_color::gmatrix_csp::run_build;
    use four_color::gmatrix_full::Side;
    let side = match args.get(2).map(String::as_str).unwrap_or("L") {
        "L" | "l" => Side::L,
        "R" | "r" => Side::R,
        other => {
            eprintln!(
                "side must be L or R, got '{other}'. Usage: {} cls-g-csp-build [side] [blocks] [threads] [cap] [stride] [json]",
                args[0]
            );
            std::process::exit(1);
        }
    };
    let m = parse_opt_num(args.get(3), "blocks").unwrap_or(3);
    if !(1..=3).contains(&m) {
        eprintln!("blocks must be 1..=3, got {m}");
        std::process::exit(1);
    }
    let threads = parse_opt_num(args.get(4), "threads").unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });
    if threads == 0 {
        eprintln!("threads must be >= 1");
        std::process::exit(1);
    }
    let cap: Option<u64> = parse_opt_num(args.get(5).filter(|s| s.as_str() != "-"), "cap");
    let stride = parse_opt_num(args.get(6), "stride").unwrap_or(1);
    if stride == 0 {
        eprintln!("stride must be >= 1");
        std::process::exit(1);
    }
    let default_path = format!(
        "results/four_color_cls_gmatrix_csp_{}_{}blocks.json",
        side.name(),
        m
    );
    let path = args.get(7).map(String::as_str).unwrap_or(&default_path);
    run_build(side, m, threads, cap, stride, path);
}

fn cmd_cls_g_csp_shard(args: &[String]) {
    use four_color::gmatrix_csp::run_shards;
    use four_color::gmatrix_full::Side;
    let side = match args.get(2).map(String::as_str).unwrap_or("L") {
        "L" | "l" => Side::L,
        "R" | "r" => Side::R,
        other => {
            eprintln!(
                "side must be L or R, got '{other}'. Usage: {} cls-g-csp-shard [side] [blocks] [start] [count] [threads] [dir] [stride]",
                args[0]
            );
            std::process::exit(1);
        }
    };
    let m = parse_opt_num(args.get(3), "blocks").unwrap_or(3);
    if !(1..=3).contains(&m) {
        eprintln!("blocks must be 1..=3, got {m}");
        std::process::exit(1);
    }
    let start = parse_opt_num(args.get(4), "start").unwrap_or(0);
    let count = parse_opt_num(args.get(5), "count").unwrap_or(1);
    let threads = parse_opt_num(args.get(6), "threads").unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });
    if threads == 0 {
        eprintln!("threads must be >= 1");
        std::process::exit(1);
    }
    let default_dir = format!("results/cls_g_csp_shards_{}_{}blocks", side.name(), m);
    let dir = args.get(7).map(String::as_str).unwrap_or(&default_dir);
    let stride = parse_opt_num(args.get(8), "stride").unwrap_or(1);
    if stride == 0 {
        eprintln!("stride must be >= 1");
        std::process::exit(1);
    }
    if !run_shards(side, m, start, count, threads, dir, stride) {
        std::process::exit(2);
    }
}

fn cmd_cls_g_csp_status(args: &[String]) {
    use four_color::gmatrix_csp::run_status;
    let dir = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("results/cls_g_csp_shards_L_3blocks");
    if !run_status(dir) {
        std::process::exit(2);
    }
}

fn cmd_cls_g_csp_merge(args: &[String]) {
    use four_color::gmatrix_csp::run_merge;
    use four_color::gmatrix_full::Side;
    let side = match args.get(2).map(String::as_str).unwrap_or("L") {
        "L" | "l" => Side::L,
        "R" | "r" => Side::R,
        other => {
            eprintln!(
                "side must be L or R, got '{other}'. Usage: {} cls-g-csp-merge [side] [blocks] [dir] [json]",
                args[0]
            );
            std::process::exit(1);
        }
    };
    let m = parse_opt_num(args.get(3), "blocks").unwrap_or(3);
    if !(1..=3).contains(&m) {
        eprintln!("blocks must be 1..=3, got {m}");
        std::process::exit(1);
    }
    let default_dir = format!("results/cls_g_csp_shards_{}_{}blocks", side.name(), m);
    let dir = args.get(4).map(String::as_str).unwrap_or(&default_dir);
    let default_path = format!(
        "results/four_color_cls_gmatrix_csp_{}_{}blocks_merged.json",
        side.name(),
        m
    );
    let path = args.get(5).map(String::as_str).unwrap_or(&default_path);
    if !run_merge(side, m, dir, path) {
        std::process::exit(2);
    }
}

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------

fn parse_usize_arg(args: &[String], index: usize, usage_hint: &str) -> usize {
    if args.len() <= index {
        eprintln!("Missing argument. Usage: {} {}", args[0], usage_hint);
        std::process::exit(1);
    }
    match args[index].parse::<usize>() {
        Ok(v) => v,
        Err(_) => {
            eprintln!(
                "Invalid number '{}'. Usage: {} {}",
                args[index], args[0], usage_hint
            );
            std::process::exit(1);
        }
    }
}

/// Optional numeric argument: absent -> None; present but unparseable ->
/// exit 1. Silently defaulting a typo can quietly convert a stratified
/// sample into a full census run (or vice versa), so present-but-invalid
/// is a launch error, never a default.
fn parse_opt_num<T: std::str::FromStr>(arg: Option<&String>, name: &str) -> Option<T> {
    arg.map(|s| {
        s.parse::<T>().unwrap_or_else(|_| {
            eprintln!("{name} must be a number, got '{s}'");
            std::process::exit(1);
        })
    })
}
