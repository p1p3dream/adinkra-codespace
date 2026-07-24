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
mod chromochar;
mod chromotopology;
mod code;
mod dashing;
mod decompose;
mod eleven_dimensional_bridge;
mod eleven_dimensional_clifford;
mod four_color;
mod eleven_dimensional_level16_couplings;
mod eleven_dimensional_prepotential;
mod eleven_dimensional_spinor_bridge;
mod eleven_dimensional_spinor_bridge_kernels;
mod enhance;
mod eval;
mod filters;
mod holoraumy;
mod lorentz;
mod lorentz_intertwiners;
mod lr_matrix;
mod minimal_supergravity_action;
mod minimal_supergravity_curvatures;
mod nauty_canonical;
mod orientation;
mod permutahedron;
mod permutahedron_atlas;
mod permutahedron_fixtures;
mod permutahedron_garden;
mod pipeline;
mod prepotential_curvature;
mod prepotential_gauge;
mod ranking;
mod search;
mod signed_perm;
mod sr_hole;
mod streamed_gadget;
mod supercovariant_derivative;
mod tendim_data;
mod tendim_generate;
mod vector_spinor_intertwiners;
mod viz_export;

use std::time::Instant;

use canonical::{compute_invariants, deduplicate, is_decomposable};
use code::{enumerate_codes, DoublyEvenCode};

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
        "adynkra-11d-clifford-verify" => cmd_adynkra_11d_clifford_verify(),
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
        "adynkra-11d-spinor-bridge-verify" => cmd_adynkra_11d_spinor_bridge_verify(),
        "adynkra-11d-spinor-kernel-verify" => cmd_adynkra_11d_spinor_kernel_verify(),
        "export-3d-assets" => cmd_export_3d_assets(&args),
        "decompose-audit" => cmd_decompose_audit(&args),
        "decompose-probe" => cmd_decompose_probe(&args),
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
    eprintln!("  adynkra-11d-clifford-verify Verify the 11D Clifford and vector-spinor projectors");
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

fn cmd_adynkra_11d_clifford_verify() {
    let report = eleven_dimensional_clifford::verify();
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
