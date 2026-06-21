use std::collections::BTreeMap;
use std::fs;
use std::time::Instant;

use rayon::prelude::*;

use crate::chromotopology::Chromotopology;
use crate::code::DoublyEvenCode;
use crate::dashing::DashingEnumerator;
use crate::decompose::decompose_rep;
use crate::filters::worldsheet_all_splits;
use crate::holoraumy::{dmin, gadget, HoloraumyData};
use crate::lr_matrix::AdinkraRep;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEntry {
    pub k: usize,
    pub generators_raw: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize)]
struct Catalog {
    n: usize,
    #[allow(dead_code)]
    total_classes: usize,
    codes: Vec<CodeEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PipelineResult {
    pub code_index: usize,
    pub n: usize,
    pub k: usize,
    pub d: usize,
    pub num_dashings: usize,
    pub garden_algebra_verified: bool,
    pub worldsheet_trivial: bool,
    pub gadget_self_values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GadgetStratum {
    pub k: usize,
    pub d: usize,
    pub num_reps: usize,
    pub rep_range: (usize, usize),
    pub matrix: Vec<Vec<f64>>,
}

/// Gadget matrix over the IRREDUCIBLE summands of a k-stratum (F8 route b).
///
/// Each reducible valise rep (k < N/2) is decomposed into `d/dmin` irreducible
/// pieces; the gadget is then computed on those pieces. The diagonal is `1.0`
/// (every irreducible has self-gadget 1, a basis-independent fact), in contrast
/// to the flat `d/dmin` diagonal of the reducible [`GadgetStratum`].
///
/// NOTE: off-diagonal (cross-summand) values are computed in the orthonormal
/// basis the decomposition produced and are NOT a basis-invariant classification
/// without a canonical orientation choice — see the [`crate::decompose`] module
/// docs.
#[derive(Debug, Clone, Serialize)]
pub struct IrrepGadgetStratum {
    pub k: usize,
    pub d: usize,
    pub dmin: usize,
    pub num_valise_reps: usize,
    pub num_irreps: usize,
    pub decomposed: bool,
    pub skip_reason: Option<String>,
    /// Maximum Garden-algebra / V²=-I residual over all summands (numerical
    /// health check; ~0 when decomposition is clean). `None` when skipped.
    pub max_summand_residual: Option<f64>,
    pub matrix: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FullPipelineOutput {
    pub n: usize,
    pub num_codes: usize,
    pub total_reps: usize,
    pub results: Vec<PipelineResult>,
    pub gadget_strata: Vec<GadgetStratum>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub irrep_strata: Vec<IrrepGadgetStratum>,
    pub elapsed_secs: f64,
}

fn gadget_stratum_matrix(reps: &[HoloraumyData]) -> Vec<Vec<f64>> {
    let n = reps.len();
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            (0..n).map(|j| gadget(&reps[i], &reps[j])).collect()
        })
        .collect();
    rows
}

pub fn run_pipeline(json_path: &str) -> FullPipelineOutput {
    run_pipeline_filtered(json_path, None)
}

pub fn run_pipeline_k(json_path: &str, only_k: usize) -> FullPipelineOutput {
    run_pipeline_filtered(json_path, Some(only_k))
}

fn run_pipeline_filtered(json_path: &str, only_k: Option<usize>) -> FullPipelineOutput {
    let t0 = Instant::now();

    let data = fs::read_to_string(json_path)
        .unwrap_or_else(|e| panic!("Failed to read codes JSON {json_path:?}: {e}"));
    let catalog: Catalog = serde_json::from_str(&data).unwrap_or_else(|e| {
        panic!("Failed to parse JSON {json_path:?}: {e}. Expected {{n, total_classes, codes:[...]}}")
    });
    let n = catalog.n;

    let codes: Vec<(usize, &CodeEntry)> = catalog
        .codes
        .iter()
        .enumerate()
        .filter(|(_, e)| only_k.map_or(true, |k| e.k == k))
        .collect();

    if let Some(k) = only_k {
        eprintln!(
            "Loaded {} code classes with k={} (N={}, {} total in catalog)",
            codes.len(), k, n, catalog.codes.len()
        );
    } else {
        eprintln!("Loaded {} code classes (N={})", codes.len(), n);
    }

    let per_code: Vec<(PipelineResult, Vec<HoloraumyData>)> = codes
        .par_iter()
        .map(|&(idx, ref entry)| {
            let code = DoublyEvenCode::new(n, entry.generators_raw.clone());
            assert!(code.is_valid(), "code {idx}: invalid doubly-even code");

            let chromo = Chromotopology::from_code(&code);
            let d = chromo.d();
            let dashing_enum = DashingEnumerator::new(&code);
            let color_perms: Vec<Vec<usize>> =
                (0..n).map(|c| chromo.color_perm(c).to_vec()).collect();
            let boson_reps = chromo.boson_reps();

            let mut garden_ok = true;
            let mut gadget_self_vals = Vec::new();
            let mut reps = Vec::new();

            for di in 0..dashing_enum.num_classes() {
                let signs =
                    dashing_enum.get_dashing_for_chromotopology(di, &boson_reps);
                let rep = AdinkraRep::from_parts(n, d, &color_perms, &signs);
                if !rep.verify_garden_algebra() {
                    garden_ok = false;
                    eprintln!(
                        "WARNING: Garden algebra failed for code {idx} dashing {di}"
                    );
                }
                let h = HoloraumyData::from_rep(&rep);
                gadget_self_vals.push(gadget(&h, &h));
                reps.push(h);
            }

            let ws = worldsheet_all_splits(&code);
            let worldsheet_trivial =
                ws.iter().all(|r| (r.p == 0 || r.q == 0) == r.passes);

            let result = PipelineResult {
                code_index: idx,
                n,
                k: entry.k,
                d,
                num_dashings: dashing_enum.num_classes(),
                garden_algebra_verified: garden_ok,
                worldsheet_trivial,
                gadget_self_values: gadget_self_vals,
            };
            (result, reps)
        })
        .collect();

    let mut results = Vec::with_capacity(per_code.len());
    let mut reps_by_k: BTreeMap<usize, Vec<HoloraumyData>> = BTreeMap::new();
    for (result, reps) in per_code {
        reps_by_k.entry(result.k).or_default().extend(reps);
        results.push(result);
    }

    let mut gadget_strata = Vec::new();
    let mut start = 0usize;
    for (&k, reps) in &reps_by_k {
        let d = reps[0].d;
        let num_reps = reps.len();
        eprintln!(
            "Computing {num_reps}x{num_reps} gadget matrix for k={k} stratum (d={d})..."
        );
        let matrix = gadget_stratum_matrix(reps);

        let expected_diag = d as f64 / dmin(n) as f64;
        for (i, row) in matrix.iter().enumerate() {
            debug_assert!(
                (row[i] - expected_diag).abs() < 1e-9,
                "k={k}: gadget diagonal[{i}] = {} != d/dmin = {expected_diag}",
                row[i]
            );
        }

        gadget_strata.push(GadgetStratum {
            k,
            d,
            num_reps,
            rep_range: (start, start + num_reps),
            matrix,
        });
        start += num_reps;
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("Pipeline complete in {elapsed:.2}s");

    FullPipelineOutput {
        n,
        num_codes: results.len(),
        total_reps: start,
        results,
        gadget_strata,
        irrep_strata: Vec::new(),
        elapsed_secs: elapsed,
    }
}

/// Build the `AdinkraRep` for every dashing class of a single code entry.
fn build_reps_for_code(n: usize, entry: &CodeEntry) -> (usize, Vec<AdinkraRep>) {
    let code = DoublyEvenCode::new(n, entry.generators_raw.clone());
    assert!(code.is_valid(), "invalid doubly-even code");
    let chromo = Chromotopology::from_code(&code);
    let d = chromo.d();
    let de = DashingEnumerator::new(&code);
    let color_perms: Vec<Vec<usize>> = (0..n).map(|c| chromo.color_perm(c).to_vec()).collect();
    let boson_reps = chromo.boson_reps();
    let reps = (0..de.num_classes())
        .map(|di| {
            let signs = de.get_dashing_for_chromotopology(di, &boson_reps);
            AdinkraRep::from_parts(n, d, &color_perms, &signs)
        })
        .collect();
    (d, reps)
}

/// Run F8 route (b) on a single k-stratum: decompose every valise rep into its
/// `d/dmin` irreducible summands and compute the gadget matrix over all summands
/// via the memory-bounded streamed Gram (`crate::streamed_gadget`).
///
/// Two guards: the stratum is recorded as skipped (not decomposed) when
/// `d > decompose::MAX_DECOMPOSE_D` (per-rep decomposition infeasible) or when the
/// estimated gadget memory exceeds `decompose::MAX_DECOMPOSE_GADGET_BYTES`
/// (num_irreps too large for RAM — needs a disk-backed/GPU tiled Gram).
pub fn run_decompose_k(json_path: &str, only_k: usize) -> FullPipelineOutput {
    let t0 = Instant::now();

    let data = fs::read_to_string(json_path)
        .unwrap_or_else(|e| panic!("Failed to read codes JSON {json_path:?}: {e}"));
    let catalog: Catalog = serde_json::from_str(&data).unwrap_or_else(|e| {
        panic!("Failed to parse JSON {json_path:?}: {e}. Expected {{n, total_classes, codes:[...]}}")
    });
    let n = catalog.n;
    let dm = dmin(n);

    let codes: Vec<(usize, &CodeEntry)> = catalog
        .codes
        .iter()
        .enumerate()
        .filter(|(_, e)| e.k == only_k)
        .collect();

    eprintln!(
        "decompose-k: {} code classes with k={} (N={}, dmin={})",
        codes.len(),
        only_k,
        n,
        dm
    );

    // Build all valise reps for the stratum (parallel per code).
    let per_code: Vec<(usize, Vec<AdinkraRep>)> = codes
        .par_iter()
        .map(|&(_idx, entry)| build_reps_for_code(n, entry))
        .collect();

    let d = per_code.first().map(|(d, _)| *d).unwrap_or(0);
    let num_valise_reps: usize = per_code.iter().map(|(_, reps)| reps.len()).sum();

    // Scale guard: skip the whole stratum if the dense path is infeasible.
    if d > crate::decompose::MAX_DECOMPOSE_D {
        let reason = format!(
            "d={d} exceeds MAX_DECOMPOSE_D={}; dense decomposition infeasible",
            crate::decompose::MAX_DECOMPOSE_D
        );
        eprintln!("decompose-k: SKIPPED k={only_k}: {reason}");
        let stratum = IrrepGadgetStratum {
            k: only_k,
            d,
            dmin: dm,
            num_valise_reps,
            num_irreps: 0,
            decomposed: false,
            skip_reason: Some(reason),
            max_summand_residual: None,
            matrix: Vec::new(),
        };
        let elapsed = t0.elapsed().as_secs_f64();
        return FullPipelineOutput {
            n,
            num_codes: codes.len(),
            total_reps: num_valise_reps,
            results: Vec::new(),
            gadget_strata: Vec::new(),
            irrep_strata: vec![stratum],
            elapsed_secs: elapsed,
        };
    }

    // Memory guard (separate from the d-guard above): peak memory scales with
    // num_irreps (= reps x d/dmin), NOT with d. The gadget uses the streamed Gram
    // path (crate::streamed_gadget), which holds one flat f32 holoraumy vector per
    // summand (~7.86 MB at N=16): k=8 ~4 GB, k=7 ~18 GB (both fit), but k=6 ~180 GB.
    // Refuse cleanly here instead of OOM-killing; larger strata need a disk-backed
    // or GPU tiled Gram.
    let num_irreps_est = num_valise_reps * (d / dm);
    let est_bytes = crate::decompose::estimated_gadget_bytes(n, num_irreps_est);
    if est_bytes > crate::decompose::MAX_DECOMPOSE_GADGET_BYTES {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let reason = format!(
            "estimated flat holoraumy store {:.1} GiB ({} irreps x {} colour-pairs x {}^2 f32) \
             exceeds budget {:.1} GiB; would OOM. Needs a disk-backed/GPU tiled Gram.",
            gib(est_bytes), num_irreps_est, n * (n - 1) / 2, dm,
            gib(crate::decompose::MAX_DECOMPOSE_GADGET_BYTES)
        );
        eprintln!("decompose-k: SKIPPED k={only_k}: {reason}");
        let stratum = IrrepGadgetStratum {
            k: only_k, d, dmin: dm, num_valise_reps, num_irreps: num_irreps_est,
            decomposed: false, skip_reason: Some(reason), max_summand_residual: None,
            matrix: Vec::new(),
        };
        let elapsed = t0.elapsed().as_secs_f64();
        return FullPipelineOutput {
            n, num_codes: codes.len(), total_reps: num_valise_reps,
            results: Vec::new(), gadget_strata: Vec::new(),
            irrep_strata: vec![stratum], elapsed_secs: elapsed,
        };
    }

    // Decompose each rep and IMMEDIATELY flatten its summands to compact f32
    // holoraumy vectors (dropping the f64 Decomposition per rep), so peak memory is
    // the flat-vector store (~7.86 MB/summand), not the retained dense holoraumy
    // tensors (~15.7 MB/summand) that OOM-killed k=7. See crate::streamed_gadget.
    eprintln!("decompose-k: decomposing {num_valise_reps} valise reps (d={d} -> {} summands each)...", d / dm);
    let all_reps: Vec<&AdinkraRep> = per_code.iter().flat_map(|(_, reps)| reps.iter()).collect();
    let r = d / dm;
    let l = crate::streamed_gadget::flat_len(n);
    let num_irreps = all_reps.len() * r;

    // Build the contiguous W (num_irreps x l, f32) IN PLACE: one chunk of r*l per
    // valise rep, written in parallel to disjoint slices. Each rep's f64
    // Decomposition is dropped at the end of its closure, so peak memory is W
    // itself (no Vec-of-Vec, no 2x copy). Worst residual is reduced via an atomic.
    use std::sync::atomic::{AtomicU64, Ordering};
    let worst_bits = AtomicU64::new(0u64);
    let mut w: Vec<f32> = vec![0.0f32; num_irreps * l];
    w.par_chunks_mut(r * l)
        .enumerate()
        .for_each(|(rep_i, chunk)| {
            let decomp = decompose_rep(all_reps[rep_i])
                .expect("d within guard, decomposition should succeed");
            debug_assert_eq!(decomp.summands.len(), r);
            let mut local_worst = 0.0f64;
            for (s_i, s) in decomp.summands.iter().enumerate() {
                local_worst = local_worst.max(crate::decompose::summand_residual(s));
                crate::streamed_gadget::flatten_summand_into(s, &mut chunk[s_i * l..(s_i + 1) * l]);
            }
            // atomic max of local_worst into worst_bits (f64 bits, CAS loop)
            let mut cur = worst_bits.load(Ordering::Relaxed);
            loop {
                if f64::from_bits(cur) >= local_worst {
                    break;
                }
                match worst_bits.compare_exchange_weak(
                    cur, local_worst.to_bits(), Ordering::Relaxed, Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(observed) => cur = observed,
                }
            }
        });
    let worst_residual = f64::from_bits(worst_bits.load(Ordering::Relaxed));

    eprintln!("decompose-k: computing {num_irreps}x{num_irreps} irreducible gadget matrix (GEMM Gram)...");
    let matrix = crate::streamed_gadget::gram_from_contiguous(&w, num_irreps, l, n);
    drop(w);

    // The irreducible self-gadget must be 1.0 (basis-independent). Check in
    // RELEASE too (not just debug_assert): a drifted diagonal means a broken
    // decomposition. f32 storage allows ~1e-7 drift, so warn above 1e-5.
    let mut worst_diag_dev = 0.0f64;
    for (i, row) in matrix.iter().enumerate() {
        worst_diag_dev = worst_diag_dev.max((row[i] - 1.0).abs());
    }
    if worst_diag_dev > 1e-5 {
        eprintln!(
            "decompose-k: WARNING: worst irrep self-gadget deviation from 1.0 is {worst_diag_dev:.3e} \
             (> 1e-5) — decomposition may be unsound for k={only_k}."
        );
    } else {
        eprintln!("decompose-k: self-gadget diagonal OK (worst dev {worst_diag_dev:.3e}).");
    }

    let stratum = IrrepGadgetStratum {
        k: only_k,
        d,
        dmin: dm,
        num_valise_reps,
        num_irreps,
        decomposed: true,
        skip_reason: None,
        max_summand_residual: Some(worst_residual),
        matrix,
    };

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("decompose-k complete in {elapsed:.2}s (max summand residual {worst_residual:.2e})");

    FullPipelineOutput {
        n,
        num_codes: codes.len(),
        total_reps: num_valise_reps,
        results: Vec::new(),
        gadget_strata: Vec::new(),
        irrep_strata: vec![stratum],
        elapsed_secs: elapsed,
    }
}

/// f32 error audit for the GEMM Gram path. Decomposes a SAMPLE of `sample_reps`
/// valise reps of stratum k (0 = all), then compares, on the resulting summands:
///   (1) the trusted dense f64 gadget (decompose::dense_gadget_matrix),
///   (2) the f64 GEMM Gram, and (3) the f32 GEMM Gram (the production path),
/// plus a Vtilde antisymmetry check. Prints max/mean errors, worst diagonal
/// drift, the f32-vs-f64 spread, the distinct-value gap vs the f32 error, and the
/// antisymmetry residual — so f32-vs-f64 is decided from numbers, not asserted.
pub fn run_decompose_audit(json_path: &str, only_k: usize, sample_reps: usize) {
    use crate::decompose::{dense_gadget_matrix, DenseHoloraumy, IrrepSummand};

    let data = fs::read_to_string(json_path)
        .unwrap_or_else(|e| panic!("Failed to read codes JSON {json_path:?}: {e}"));
    let catalog: Catalog = serde_json::from_str(&data).expect("parse catalog");
    let n = catalog.n;
    let dm = dmin(n);

    let codes: Vec<(usize, &CodeEntry)> = catalog
        .codes
        .iter()
        .enumerate()
        .filter(|(_, e)| e.k == only_k)
        .collect();

    // Sample reps (decompose, collect summands).
    let mut reps_done = 0usize;
    let mut summands: Vec<IrrepSummand> = Vec::new();
    'outer: for (_idx, entry) in &codes {
        let (_d, reps) = build_reps_for_code(n, entry);
        for rep in &reps {
            if sample_reps != 0 && reps_done >= sample_reps {
                break 'outer;
            }
            let decomp = decompose_rep(rep).expect("decompose (audit)");
            summands.extend(decomp.summands);
            reps_done += 1;
        }
    }
    let ni = summands.len();
    let d = if ni > 0 { dm } else { 0 };
    let _ = d;
    println!("=== decompose-audit: N={n} k={only_k} sample_reps={reps_done} summands={ni} dmin={dm} ===");
    if ni == 0 {
        println!("(no summands)");
        return;
    }

    // Antisymmetry of every Vtilde over the sampled summands.
    let mut worst_antisym = 0.0f64;
    let dense: Vec<DenseHoloraumy> = summands
        .iter()
        .map(|s| {
            let dh = DenseHoloraumy::from_summand(s);
            for m in &dh.vtilde {
                let mt = m.transpose();
                let w = m.data.iter().zip(mt.data.iter())
                    .map(|(a, b)| (a + b).abs()).fold(0.0f64, f64::max);
                worst_antisym = worst_antisym.max(w);
            }
            dh
        })
        .collect();

    // (1) trusted dense f64 reference.
    let g_dense = dense_gadget_matrix(&dense);

    // Flat vectors, f32 and f64.
    let l = crate::streamed_gadget::flat_len(n);
    let mut w32 = vec![0.0f32; ni * l];
    let mut w64 = vec![0.0f64; ni * l];
    for (i, s) in summands.iter().enumerate() {
        crate::streamed_gadget::flatten_summand_into(s, &mut w32[i * l..(i + 1) * l]);
        let dh = DenseHoloraumy::from_summand(s);
        let mut idx = i * l;
        for mm in &dh.vtilde {
            for &x in &mm.data {
                w64[idx] = x;
                idx += 1;
            }
        }
    }
    // (3) f32 GEMM and (2) f64 GEMM.
    let g32 = crate::streamed_gadget::gram_from_contiguous(&w32, ni, l, n);
    let g64 = crate::streamed_gadget::gram_from_contiguous_f64(&w64, ni, l, n);

    // Metrics.
    let (mut max_df, mut sum_df, mut cnt) = (0.0f64, 0.0f64, 0usize);
    let (mut max_d64, mut max_3264, mut max_rel) = (0.0f64, 0.0f64, 0.0f64);
    let mut worst_diag = 0.0f64;
    for i in 0..ni {
        worst_diag = worst_diag.max((g32[i][i] - 1.0).abs());
        for j in 0..ni {
            let e_df = (g32[i][j] - g_dense[i][j]).abs();
            max_df = max_df.max(e_df);
            sum_df += e_df;
            cnt += 1;
            max_d64 = max_d64.max((g64[i][j] - g_dense[i][j]).abs());
            let e = (g32[i][j] - g64[i][j]).abs();
            max_3264 = max_3264.max(e);
            if g64[i][j].abs() > 1e-9 {
                max_rel = max_rel.max(e / g64[i][j].abs());
            }
        }
    }
    // Distinct off-diagonal values (rounded) + min adjacent gap, from the f64 ref.
    let round6 = |x: f64| (x * 1e6).round() / 1e6;
    let mut off64: Vec<f64> = Vec::new();
    let mut off32: Vec<f64> = Vec::new();
    for i in 0..ni {
        for j in 0..ni {
            if i != j {
                off64.push(round6(g64[i][j]));
                off32.push(round6(g32[i][j]));
            }
        }
    }
    let distinct = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.dedup();
        v
    };
    let d64 = distinct(off64);
    let d32 = distinct(off32.clone());
    let min_gap = d64.windows(2).map(|w| w[1] - w[0]).fold(f64::INFINITY, f64::min);

    println!("antisymmetry  : max |Vtilde + Vtilde^T| = {worst_antisym:.3e}");
    println!("identity check: max |dense_f64 - GEMM_f64| = {max_d64:.3e}  (validates the Gram identity)");
    println!("f32 vs dense  : max abs = {max_df:.3e}  mean abs = {:.3e}  worst diag drift = {worst_diag:.3e}", sum_df / cnt as f64);
    println!("f32 vs f64    : max abs = {max_3264:.3e}  max rel = {max_rel:.3e}");
    println!("distinct off  : f64 = {}  f32 = {}  (rounded 1e-6)", d64.len(), d32.len());
    println!("nearest gap   : min adjacent f64 gap = {min_gap:.3e}   vs f32 max err = {max_3264:.3e}");
    let safe = max_3264 < min_gap * 0.5;
    println!(
        "VERDICT       : f32 is {} for distinguishing values (max err {} half-gap {:.3e})",
        if safe { "SAFE" } else { "NOT clearly safe" },
        if safe { "<" } else { ">=" },
        min_gap * 0.5
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_entry_parsing() {
        let json = r#"{"n": 4, "total_classes": 1, "codes": [{"k": 1, "generators_raw": [15]}]}"#;
        let catalog: Catalog = serde_json::from_str(json).unwrap();
        assert_eq!(catalog.n, 4);
        assert_eq!(catalog.codes.len(), 1);
        assert_eq!(catalog.codes[0].k, 1);
        assert_eq!(catalog.codes[0].generators_raw, vec![15]);
    }
}

#[cfg(test)]
mod integration_tests {
    use crate::chromotopology::Chromotopology;
    use crate::code::DoublyEvenCode;
    use crate::dashing::DashingEnumerator;
    use crate::holoraumy::{gadget, HoloraumyData};
    use crate::lr_matrix::AdinkraRep;

    fn assert_real_construction_invariants(label: &str, code: &DoublyEvenCode) {
        let chromo = Chromotopology::from_code(code);
        let d = chromo.d();
        let de = DashingEnumerator::new(code);
        let color_perms: Vec<Vec<usize>> =
            (0..code.n).map(|c| chromo.color_perm(c).to_vec()).collect();
        let boson_reps = chromo.boson_reps();

        for di in 0..de.num_classes() {
            let raw = de.get_dashing(di);
            assert!(de.verify_odd(&raw), "{label} dashing {di}: not odd");

            let signs = de.get_dashing_for_chromotopology(di, &boson_reps);
            let rep = AdinkraRep::from_parts(code.n, d, &color_perms, &signs);
            assert!(
                rep.verify_garden_algebra(),
                "{label} dashing {di}: Garden algebra failed"
            );

            let holo = HoloraumyData::from_rep(&rep);
            for (idx, vij) in holo.v.iter().enumerate() {
                assert!(
                    vij.compose(vij).is_neg_identity(),
                    "{label} dashing {di}: V[{idx}]^2 != -I"
                );
            }
            let g = gadget(&holo, &holo);
            assert!(
                (g - 1.0).abs() < 1e-9,
                "{label} dashing {di}: self-gadget {g} != 1.0"
            );
        }
    }

    #[test]
    fn n4_code_4_1_4_real_construction() {
        assert_real_construction_invariants(
            "[4,1,4]",
            &DoublyEvenCode::new(4, vec![0b1111]),
        );
    }

    #[test]
    fn n8_hamming_8_4_4_real_construction() {
        assert_real_construction_invariants(
            "[8,4,4]",
            &DoublyEvenCode::new(
                8,
                vec![0b11100001, 0b11010010, 0b10110100, 0b01111000],
            ),
        );
    }
}
