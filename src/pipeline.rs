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
    /// Worst |self-gadget − 1| over the diagonal (basis-independent; should be ~0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diag_worst_dev: Option<f64>,
    /// Number of distinct off-diagonal values (rounded to 1e-6). NOTE: for k<8
    /// these are orientation-dependent and f32-quantized — a run fingerprint, not
    /// a classification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distinct_off_values: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub off_min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub off_max: Option<f64>,
    /// Most common off-diagonal values: (rounded value, count), top 50 by count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub off_histogram_top: Option<Vec<(f64, usize)>>,
    /// When the full matrix is too large to embed in JSON, it is written here as
    /// raw little-endian f64 (row-major, num_irreps²) and `matrix` is left empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matrix_binary_path: Option<String>,
    /// True if the gadget was computed via the disk-backed tiled Gram (W spilled
    /// to a scratch file) rather than the in-RAM GEMM. None for skipped strata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub used_disk_path: Option<bool>,
    /// True if the f32 numeric gate flagged the disk run as SUSPECT (f32 error
    /// could merge/split distinct off-diagonal values per the gap criterion).
    /// Some(false) = gate passed; None = gate not run (in-RAM / skipped).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numeric_suspect: Option<bool>,
    /// Full gadget matrix, embedded in JSON only for small strata (num_irreps ≤
    /// MATRIX_JSON_CAP); otherwise empty (see `matrix_binary_path`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub matrix: Vec<Vec<f64>>,
}

/// Above this num_irreps the full matrix is dumped to a binary file rather than
/// embedded in the (otherwise multi-hundred-MB) JSON. k=8 (512) embeds; k=7
/// (2304) and k=6 (5888) dump + summarize.
const MATRIX_JSON_CAP: usize = 1024;

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
/// Free bytes on the filesystem containing `dir`, via `df -k` (dependency-free,
/// works on Linux + macOS). `None` if `df` is unavailable/unparseable.
fn available_disk_bytes(dir: &std::path::Path) -> Option<u64> {
    let out = std::process::Command::new("df").arg("-k").arg(dir).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    // Last line: Filesystem 1K-blocks Used Available Capacity% ... ; the 3rd
    // pure-integer token (blocks, used, available) is Available in KiB.
    let last = s.lines().last()?;
    let nums: Vec<u64> = last.split_whitespace().filter_map(|t| t.parse::<u64>().ok()).collect();
    nums.get(2).map(|kb| kb * 1024)
}

/// f32 numeric gate for the disk path: decompose a SAMPLE of reps and compare the
/// f32 GEMM gadget against the trusted dense f64 gadget on those summands.
///
/// Verdict uses the GAP-BASED criterion (the one the audit established, not an
/// arbitrary absolute bound): f32 is SUSPECT iff the max f32-vs-dense error is at
/// least half the smallest gap between distinct off-diagonal values — i.e. iff
/// f32 quantization could merge/split genuinely distinct gadget values. Returns
/// `true` if SUSPECT. Cheap relative to the full run.
fn disk_f32_gate(all_reps: &[&AdinkraRep], n: usize, sample: usize) -> bool {
    use crate::decompose::{dense_gadget_matrix, decompose_rep, DenseHoloraumy};
    let take = sample.min(all_reps.len());
    if take == 0 {
        return false;
    }
    let mut summands = Vec::new();
    for rep in &all_reps[..take] {
        let d = decompose_rep(rep).expect("decompose (f32 gate)");
        summands.extend(d.summands);
    }
    let dense: Vec<DenseHoloraumy> = summands.iter().map(DenseHoloraumy::from_summand).collect();
    let gd = dense_gadget_matrix(&dense);
    let vecs: Vec<Vec<f32>> = summands.iter().map(crate::streamed_gadget::flatten_summand).collect();
    let gf = crate::streamed_gadget::gram_gadget_matrix(&vecs, n);
    let m = gd.len();
    let (mut maxe, mut diagdev) = (0.0f64, 0.0f64);
    let mut off: Vec<f64> = Vec::new();
    for i in 0..m {
        diagdev = diagdev.max((gf[i][i] - 1.0).abs());
        for j in 0..m {
            maxe = maxe.max((gf[i][j] - gd[i][j]).abs());
            if i != j {
                off.push((gd[i][j] * 1e9).round() / 1e9); // round vs f64 reference
            }
        }
    }
    off.sort_by(|a, b| a.partial_cmp(b).unwrap());
    off.dedup();
    let min_gap = off.windows(2).map(|w| w[1] - w[0]).fold(f64::INFINITY, f64::min);
    // SUSPECT iff f32 error could cross half the nearest value gap.
    let suspect = maxe >= 0.5 * min_gap;
    eprintln!(
        "decompose-k: f32 gate ({} summands from {take} reps): max|f32-dense|={maxe:.2e}, \
         worst diag dev={diagdev:.2e}, nearest value gap={min_gap:.2e} -> f32 {}",
        summands.len(),
        if suspect { "SUSPECT" } else { "SAFE" }
    );
    if suspect {
        eprintln!(
            "decompose-k: WARNING: f32 error ({maxe:.2e}) >= half the nearest value gap ({:.2e}); \
             distinct off-diagonal values may be merged/split. Output will be flagged numeric_suspect \
             and the matrix dump stamped _SUSPECT. Consider an f64 build.",
            0.5 * min_gap
        );
    }
    suspect
}

pub fn run_decompose_k(json_path: &str, only_k: usize) -> FullPipelineOutput {
    run_decompose_k_mode(json_path, only_k, false, false)
}

/// `allow_disk = true` routes strata whose flat store exceeds the RAM budget to
/// the disk-backed tiled Gram instead of skipping (the `decompose-k-disk` /
/// `--disk` path). `false` preserves the original behavior (in-RAM or skip).
///
/// `disk_f64 = true` (the `--f64` build) stores W as exact f64 on disk and
/// accumulates the Gram with `dgemm`, so the gadget values are exact to ~1e-13 and
/// the distinct-off-diagonal-value count is trustworthy (vs the f32 path's ~1e-7
/// quantization, which flags `numeric_suspect`). f64 always uses the disk path
/// (the f64 store is double the f32 store and would not fit RAM); it requires
/// `allow_disk`.
pub fn run_decompose_k_mode(
    json_path: &str,
    only_k: usize,
    allow_disk: bool,
    disk_f64: bool,
) -> FullPipelineOutput {
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
            diag_worst_dev: None,
            distinct_off_values: None,
            off_min: None,
            off_max: None,
            off_histogram_top: None,
            matrix_binary_path: None,
            used_disk_path: None,
            numeric_suspect: None,
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

    let all_reps: Vec<&AdinkraRep> = per_code.iter().flat_map(|(_, reps)| reps.iter()).collect();
    let r = d / dm;
    let l = crate::streamed_gadget::flat_len(n);
    let num_irreps = all_reps.len() * r;
    let est_bytes = crate::decompose::estimated_gadget_bytes(n, num_irreps);
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let ram_fits = est_bytes <= crate::decompose::MAX_DECOMPOSE_GADGET_BYTES;

    use std::sync::atomic::{AtomicU64, Ordering};
    // atomic f64-max helper (CAS loop) shared by both build paths.
    let atomic_max = |bits: &AtomicU64, v: f64| {
        let mut cur = bits.load(Ordering::Relaxed);
        loop {
            if f64::from_bits(cur) >= v {
                break;
            }
            match bits.compare_exchange_weak(cur, v.to_bits(), Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(o) => cur = o,
            }
        }
    };

    // A one-shot helper to return a skipped stratum (RAM budget / disk cap).
    let skip = |reason: String| -> FullPipelineOutput {
        eprintln!("decompose-k: SKIPPED k={only_k}: {reason}");
        FullPipelineOutput {
            n,
            num_codes: codes.len(),
            total_reps: num_valise_reps,
            results: Vec::new(),
            gadget_strata: Vec::new(),
            irrep_strata: vec![IrrepGadgetStratum {
                k: only_k, d, dmin: dm, num_valise_reps, num_irreps,
                decomposed: false, skip_reason: Some(reason), max_summand_residual: None,
                diag_worst_dev: None, distinct_off_values: None, off_min: None, off_max: None,
                off_histogram_top: None, matrix_binary_path: None, used_disk_path: None,
                numeric_suspect: None,
                matrix: Vec::new(),
            }],
            elapsed_secs: t0.elapsed().as_secs_f64(),
        }
    };

    if disk_f64 && !allow_disk {
        return skip(
            "the exact f64 build (--f64) requires the disk path; re-run with \
             `decompose-k-disk <k> --f64`."
                .to_string(),
        );
    }
    if !ram_fits && !allow_disk {
        return skip(format!(
            "estimated flat holoraumy store {:.1} GiB ({num_irreps} irreps) exceeds RAM budget {:.1} GiB; \
             re-run with `decompose-k-disk` (--disk) to spill W to a scratch file.",
            gib(est_bytes), gib(crate::decompose::MAX_DECOMPOSE_GADGET_BYTES)
        ));
    }
    // ADINKRA_FORCE_DISK=1 routes a RAM-fitting stratum through the disk path too
    // (for disk-vs-RAM parity validation, e.g. k=6). Requires --disk. The exact
    // f64 build always uses the disk path (the f64 store is too large for RAM).
    let force_disk = std::env::var("ADINKRA_FORCE_DISK").map(|v| v == "1").unwrap_or(false);
    let use_disk = allow_disk && (!ram_fits || force_disk || disk_f64);

    // Build path: in-RAM contiguous W + GEMM when it fits the RAM budget;
    // otherwise (use_disk) stream W to a scratch file and tile the Gram off disk.
    let (matrix, worst_residual, used_disk, numeric_suspect) = if !use_disk {
        eprintln!("decompose-k: decomposing {num_valise_reps} valise reps (d={d} -> {r} summands each), in-RAM GEMM Gram...");
        let worst_bits = AtomicU64::new(0u64);
        let mut w: Vec<f32> = vec![0.0f32; num_irreps * l];
        w.par_chunks_mut(r * l).enumerate().for_each(|(rep_i, chunk)| {
            let decomp = decompose_rep(all_reps[rep_i])
                .expect("d within guard, decomposition should succeed");
            assert_eq!(decomp.summands.len(), r, "decomposition produced != r summands");
            let mut lw = 0.0f64;
            for (s_i, s) in decomp.summands.iter().enumerate() {
                lw = lw.max(crate::decompose::summand_residual(s));
                crate::streamed_gadget::flatten_summand_into(s, &mut chunk[s_i * l..(s_i + 1) * l]);
            }
            atomic_max(&worst_bits, lw);
        });
        let wr = f64::from_bits(worst_bits.load(Ordering::Relaxed));
        eprintln!("decompose-k: computing {num_irreps}x{num_irreps} gadget matrix (GEMM Gram)...");
        let m = crate::streamed_gadget::gram_from_contiguous(&w, num_irreps, l, n);
        (m, wr, false, None)
    } else {
        // Disk-backed path (allow_disk == true; doesn't fit RAM, is forced, or is
        // the exact f64 build). f64 doubles the store (8 bytes/element).
        let elem = if disk_f64 { "f64" } else { "f32" };
        let flat_bytes = if disk_f64 {
            crate::streamed_gadget::flat_store_bytes_f64(n, num_irreps)
        } else {
            crate::streamed_gadget::flat_store_bytes(n, num_irreps)
        };
        if flat_bytes > crate::decompose::MAX_DECOMPOSE_DISK_BYTES {
            return skip(format!(
                "flat store {:.1} GiB exceeds MAX_DECOMPOSE_DISK_BYTES {:.1} GiB.",
                gib(flat_bytes), gib(crate::decompose::MAX_DECOMPOSE_DISK_BYTES)
            ));
        }
        let scratch_dir = match std::env::var("ADINKRA_SCRATCH_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => {
                let d = std::env::temp_dir();
                eprintln!(
                    "decompose-k: WARNING: ADINKRA_SCRATCH_DIR not set; using {} for the {:.1} GiB \
                     scratch. If that is a tmpfs/RAM-backed dir (common for /tmp on Linux), this \
                     defeats the disk path and may OOM — set ADINKRA_SCRATCH_DIR to a real disk.",
                    d.display(), gib(flat_bytes)
                );
                d
            }
        };
        // Free-space preflight (via `df`): refuse before the expensive decomposition
        // if the volume can't hold the scratch + an 8 GiB margin.
        match available_disk_bytes(&scratch_dir) {
            Some(avail) if avail < flat_bytes + (8u64 << 30) => {
                return skip(format!(
                    "scratch dir {} has only {:.1} GiB free; need {:.1} GiB + 8 GiB margin.",
                    scratch_dir.display(), gib(avail), gib(flat_bytes)
                ));
            }
            Some(_) => {}
            None => eprintln!("decompose-k: WARNING: could not determine free space on {}", scratch_dir.display()),
        }
        // Unique filename (pid) so concurrent runs don't collide on one scratch file.
        let scratch_path = scratch_dir.join(format!(
            "adinkra_holoraumy_n{n}_k{only_k}_irreps{num_irreps}_l{l}_pid{}.{elem}scratch",
            std::process::id()
        ));
        eprintln!(
            "decompose-k: DISK path ({elem} store {:.1} GiB): spilling W to {} ...",
            gib(flat_bytes), scratch_path.display()
        );
        eprintln!(
            "decompose-k: (the scratch is removed on normal exit; if you Ctrl-C, reap it with: \
             rm -f {}/adinkra_holoraumy_*_pid*.{elem}scratch)",
            scratch_dir.display()
        );
        // f32 numeric gate on a sample BEFORE committing to the full run. The f64
        // build is exact (~1e-13) by construction, so the gate is skipped there and
        // the stratum is not flagged numeric_suspect.
        let suspect = if disk_f64 {
            eprintln!("decompose-k: f64 disk build (exact ~1e-13); skipping f32 gate.");
            false
        } else {
            disk_f32_gate(&all_reps, n, 32)
        };

        let scratch = crate::streamed_gadget::ScratchW::create(scratch_path, flat_bytes)
            .unwrap_or_else(|e| panic!("decompose-k: failed to create scratch file: {e}"));

        eprintln!("decompose-k: decomposing {num_valise_reps} valise reps (d={d} -> {r} summands each) -> {elem} disk...");
        let worst_bits = AtomicU64::new(0u64);
        let write_err: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
        (0..all_reps.len()).into_par_iter().for_each(|rep_i| {
            if write_err.lock().unwrap().is_some() {
                return;
            }
            let decomp = decompose_rep(all_reps[rep_i])
                .expect("d within guard, decomposition should succeed");
            assert_eq!(decomp.summands.len(), r, "decomposition produced != r summands");
            // f64 build stores exact entries; f32 build halves the disk footprint.
            let mut buf32 = if disk_f64 { Vec::new() } else { vec![0.0f32; l] };
            let mut buf64 = if disk_f64 { vec![0.0f64; l] } else { Vec::new() };
            let mut lw = 0.0f64;
            for (s_i, s) in decomp.summands.iter().enumerate() {
                lw = lw.max(crate::decompose::summand_residual(s));
                let g = rep_i * r + s_i;
                let res = if disk_f64 {
                    crate::streamed_gadget::flatten_summand_into_f64(s, &mut buf64);
                    scratch.write_summand_f64(g, &buf64, l)
                } else {
                    crate::streamed_gadget::flatten_summand_into(s, &mut buf32);
                    scratch.write_summand(g, &buf32, l)
                };
                if let Err(e) = res {
                    write_err.lock().unwrap().get_or_insert_with(|| format!("write_summand g={g}: {e}"));
                    return;
                }
            }
            atomic_max(&worst_bits, lw);
        });
        if let Some(msg) = write_err.into_inner().unwrap() {
            panic!("decompose-k: disk write failed (scratch kept if ADINKRA_KEEP_SCRATCH=1): {msg}");
        }
        scratch.sync().unwrap_or_else(|e| panic!("decompose-k: scratch sync failed: {e}"));
        let wr = f64::from_bits(worst_bits.load(Ordering::Relaxed));
        // f64 tiles are double the bytes of f32 tiles, so halve the default tile to
        // keep the same peak RAM (two 512-row f64 tiles ≈ two 1024-row f32 tiles).
        let default_tile = if disk_f64 {
            crate::streamed_gadget::DEFAULT_TILE_ROWS / 2
        } else {
            crate::streamed_gadget::DEFAULT_TILE_ROWS
        };
        let tile = std::env::var("ADINKRA_GRAM_TILE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(default_tile);
        eprintln!("decompose-k: tiled {elem} GEMM Gram from disk ({num_irreps}x{num_irreps}, tile_rows={tile})...");
        let m = if disk_f64 {
            crate::streamed_gadget::gram_from_disk_f64(&scratch, num_irreps, l, n, tile)
        } else {
            crate::streamed_gadget::gram_from_disk(&scratch, num_irreps, l, n, tile)
        }
        .unwrap_or_else(|e| panic!("decompose-k: gram_from_disk failed: {e}"));
        (m, wr, true, Some(suspect)) // `scratch` drops here -> file removed (unless ADINKRA_KEEP_SCRATCH=1)
    };

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

    // Off-diagonal summary (rounded to 1e-6). For k<8 these values are
    // orientation-dependent + f32-quantized — a run fingerprint, not a
    // classification — but the histogram/extents are a useful at-a-glance signal.
    let round6 = |x: f64| (x * 1e6).round() / 1e6;
    let mut counts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    let (mut omin, mut omax) = (f64::INFINITY, f64::NEG_INFINITY);
    for (i, row) in matrix.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if i != j {
                let r = round6(v);
                *counts.entry((r * 1e6).round() as i64).or_insert(0) += 1;
                omin = omin.min(v);
                omax = omax.max(v);
            }
        }
    }
    let distinct_off = counts.len();
    let mut hist: Vec<(f64, usize)> =
        counts.iter().map(|(&k, &c)| (k as f64 / 1e6, c)).collect();
    hist.sort_by(|a, b| b.1.cmp(&a.1)); // by count desc
    hist.truncate(50);
    eprintln!(
        "decompose-k: off-diagonal: {distinct_off} distinct values (rounded 1e-6), range [{omin:.6}, {omax:.6}]"
    );

    // Embed the full matrix in JSON only for small strata; otherwise dump it to a
    // binary file (raw LE f64, row-major) and keep the JSON to the summary.
    let mut matrix_binary_path = None;
    let mut matrix_out = matrix;
    if num_irreps > MATRIX_JSON_CAP {
        let suspect_tag = if numeric_suspect == Some(true) { "_SUSPECT" } else { "" };
        let path = format!("decompose_k{only_k}_gadget_{num_irreps}{suspect_tag}.f64bin");
        let mut bytes = Vec::with_capacity(num_irreps * num_irreps * 8);
        for row in &matrix_out {
            for &v in row {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        match fs::write(&path, &bytes) {
            Ok(_) => {
                eprintln!("decompose-k: wrote {num_irreps}x{num_irreps} matrix to {path} ({} MiB, raw LE f64)", bytes.len() / (1 << 20));
                matrix_binary_path = Some(path);
            }
            Err(e) => eprintln!("decompose-k: WARNING: failed to write matrix binary {path}: {e}; leaving matrix out of JSON"),
        }
        matrix_out = Vec::new(); // keep JSON small regardless
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
        diag_worst_dev: Some(worst_diag_dev),
        distinct_off_values: Some(distinct_off),
        off_min: if omin.is_finite() { Some(omin) } else { None },
        off_max: if omax.is_finite() { Some(omax) } else { None },
        off_histogram_top: Some(hist),
        matrix_binary_path,
        used_disk_path: Some(used_disk),
        numeric_suspect,
        matrix: matrix_out,
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

    // Entry-level f32 cast error: max |(x as f32) as f64 - x| over every flat
    // holoraumy entry. This is the ONLY metric that decides whether f32 is truly
    // lossless. ~1e-16 => every entry is exactly f32-representable (f32 loses
    // nothing at the entry level; the gadget's 1e-16 is pure accumulation
    // rounding). ~1e-7 => the entries are NOT f32-representable and f32 loses
    // real precision that only happens to cancel in the gadget sum.
    let nflat = ni * l;
    let (mut max_cast, mut sum_cast, mut worst_cast_entry) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..nflat {
        let e = (w32[k] as f64 - w64[k]).abs();
        if e > max_cast {
            max_cast = e;
            worst_cast_entry = w64[k];
        }
        sum_cast += e;
    }
    println!(
        "entry cast    : max |(x as f32) as f64 - x| = {max_cast:.3e}  mean = {:.3e}  (worst entry x={worst_cast_entry:.6e})",
        sum_cast / nflat as f64
    );

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

/// Decomposition-only feasibility probe: decompose the first `num_reps` valise
/// reps of stratum k and report per-rep time, summand count, max residual, and
/// worst self-gadget deviation — WITHOUT building the full gadget matrix (so no
/// num_irreps² memory). Used to test whether large-d decomposition (e.g. d=1024
/// at k=5) is feasible before committing to it.
pub fn run_decompose_probe(json_path: &str, only_k: usize, num_reps: usize) {
    use crate::decompose::{dense_gadget, decompose_rep, summand_residual, DenseHoloraumy};

    let data = fs::read_to_string(json_path)
        .unwrap_or_else(|e| panic!("Failed to read codes JSON {json_path:?}: {e}"));
    let catalog: Catalog = serde_json::from_str(&data).expect("parse catalog");
    let n = catalog.n;
    let dm = dmin(n);
    let codes: Vec<&CodeEntry> = catalog.codes.iter().filter(|e| e.k == only_k).collect();
    let d = if only_k <= 15 { 1usize << (n - only_k - 1) } else { 0 };
    println!(
        "=== decompose-probe N={n} k={only_k} d={d} dmin={dm} r={} (probing {num_reps} reps) ===",
        d / dm.max(1)
    );

    let mut done = 0usize;
    let mut times: Vec<f64> = Vec::new();
    let (mut worst_res, mut worst_dev) = (0.0f64, 0.0f64);
    'outer: for entry in &codes {
        let (_d, reps) = build_reps_for_code(n, entry);
        for rep in &reps {
            if done >= num_reps {
                break 'outer;
            }
            let t = Instant::now();
            let decomp = match decompose_rep(rep) {
                Some(x) => x,
                None => {
                    println!("rep {done}: decompose_rep returned None — d={} exceeds MAX_DECOMPOSE_D", rep.d);
                    return;
                }
            };
            let dt = t.elapsed().as_secs_f64();
            let (mut res, mut dev) = (0.0f64, 0.0f64);
            for s in &decomp.summands {
                res = res.max(summand_residual(s));
                let dh = DenseHoloraumy::from_summand(s);
                dev = dev.max((dense_gadget(&dh, &dh) - 1.0).abs());
            }
            worst_res = worst_res.max(res);
            worst_dev = worst_dev.max(dev);
            times.push(dt);
            println!(
                "rep {done}: {dt:.2}s  summands={}  max_resid={res:.2e}  self-gadget dev={dev:.2e}",
                decomp.summands.len()
            );
            done += 1;
        }
    }
    if times.is_empty() {
        println!("(no reps for k={only_k})");
        return;
    }
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let mx = times.iter().cloned().fold(0.0, f64::max);
    let full_reps = codes.len() * (1usize << only_k); // num_valise_reps = codes * 2^k
    println!(
        "=== {done} reps: avg {avg:.2}s/rep  max {mx:.2}s/rep  worst residual {worst_res:.2e}  worst self-gadget dev {worst_dev:.2e} ===",
    );
    println!(
        "=== extrapolated decomposition of all {full_reps} k={only_k} valise reps: ~{:.0}s ({:.1} min) sequential; /cores in parallel ===",
        avg * full_reps as f64,
        avg * full_reps as f64 / 60.0
    );
}

/// One inferred Schur class for a distinct commutant dimension seen in a stratum.
#[derive(serde::Serialize)]
pub struct StructureClass {
    /// The basis-invariant commutant dimension (`= m²·e` when isotypic).
    pub commutant_dim: usize,
    /// How many of the stratum's valise reps have this commutant dimension.
    pub count: usize,
    /// True iff `commutant_dim == r²·e` for `e ∈ {1,2,4}` (a single irreducible
    /// type with multiplicity `r` over a real/complex/quaternionic division algebra).
    pub isotypic: bool,
    /// `e = commutant_dim / r²` when isotypic (real dimension of the division algebra).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub division_algebra_dim: Option<usize>,
    /// `"R"`, `"C"`, or `"H"` when isotypic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub division_algebra: Option<String>,
    /// The unique real-type (e=1) multiplicity pattern `m_t` (sorted descending,
    /// `Σ m_t = r`, `Σ m_t² = commutant_dim`) when one exists — i.e. how many
    /// INEQUIVALENT irreducible adinkra classes appear and with what multiplicity.
    /// `None` when no such pattern or when it is ambiguous (then see `note`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiplicity_pattern: Option<Vec<usize>>,
    /// Number of inequivalent irreducible classes (`multiplicity_pattern.len()`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_irreducible_types: Option<usize>,
    pub note: String,
}

/// All real-type (e=1) multiplicity patterns: partitions of `r` (sorted parts,
/// descending) whose squares sum to `target = commutant_dim`. For the N=16 minimal
/// irrep (real, e=1; verified at k=8) these are the candidate decompositions of a
/// reducible rep into `Σ m_t` copies across `len()` inequivalent classes.
fn real_multiplicity_patterns(r: usize, target: usize) -> Vec<Vec<usize>> {
    fn rec(rem: usize, sq: usize, max_part: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if rem == 0 {
            if sq == 0 {
                out.push(cur.clone());
            }
            return;
        }
        for p in (1..=max_part.min(rem)).rev() {
            if p * p > sq {
                continue;
            }
            cur.push(p);
            rec(rem - p, sq - p * p, p, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    rec(r, target, r, &mut Vec::new(), &mut out);
    out
}

/// Basis-invariant structural summary of a single k-stratum.
#[derive(serde::Serialize)]
pub struct StructureStratum {
    pub k: usize,
    pub d: usize,
    pub dmin: usize,
    pub r: usize,
    pub num_codes: usize,
    pub num_valise_reps: usize,
    /// Distinct commutant dimensions across the stratum's valise reps, with counts.
    pub commutant_dim_histogram: Vec<(usize, usize)>,
    pub classes: Vec<StructureClass>,
    pub elapsed_secs: f64,
}

/// Compute the BASIS-INVARIANT structural results of a k-stratum WITHOUT the dense
/// eigendecomposition or the (orientation-dependent) gadget matrix: the commutant
/// dimension of each valise rep (exact integer union-find, [`commutant_dim`]), and
/// the multiplicity / division-algebra (Schur) type it implies. Because it never
/// decomposes or forms the gadget, it scales to the high-k strata (k ≤ 3, d up to
/// 16384) that the dense path cannot reach — and it reports exactly the invariant
/// content that survives the orientation gauge (see [`crate::orientation`]): the
/// cross-summand gadget below k=8 is gauge, so the honest classification of a
/// reducible stratum is its multiplicity and Schur structure, not a value spectrum.
pub fn run_decompose_structure(json_path: &str, only_k: usize) {
    use crate::decompose::commutant_dim;
    use rayon::prelude::*;

    let t0 = Instant::now();
    let data = fs::read_to_string(json_path)
        .unwrap_or_else(|e| panic!("Failed to read codes JSON {json_path:?}: {e}"));
    let catalog: Catalog = serde_json::from_str(&data).expect("parse catalog");
    let n = catalog.n;
    let dm = dmin(n);
    let codes: Vec<&CodeEntry> = catalog.codes.iter().filter(|e| e.k == only_k).collect();

    eprintln!(
        "decompose-structure: {} codes with k={only_k} (N={n}, dmin={dm}); commutant only, no eig/gadget",
        codes.len()
    );

    // Build every valise rep of the stratum (parallel per code).
    let per_code: Vec<(usize, Vec<AdinkraRep>)> =
        codes.par_iter().map(|e| build_reps_for_code(n, e)).collect();
    let d = per_code.first().map(|(d, _)| *d).unwrap_or(0);
    let all_reps: Vec<&AdinkraRep> = per_code.iter().flat_map(|(_, reps)| reps.iter()).collect();
    let num_valise_reps = all_reps.len();
    let r = if dm > 0 { d / dm } else { 0 };

    // Each commutant_dim allocates a d² signed disjoint-set (~6 bytes/cell:
    // u32 parent + i8 rel + bool consistent). Cap concurrency so the peak across
    // workers stays under a memory budget (default 24 GiB, env-overridable): at
    // k=1 (d=16384) one DSU is ~1.6 GiB, so unbounded rayon over 8 reps would need
    // ~13 GiB. Batches run in parallel; batches themselves are sequential.
    let dsu_bytes = (d as u64) * (d as u64) * 6;
    let budget_gib: u64 = std::env::var("ADINKRA_STRUCT_MEM_GIB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let budget = budget_gib << 30;
    let max_par = ((budget / dsu_bytes.max(1)).max(1) as usize).min(num_valise_reps.max(1));
    eprintln!(
        "decompose-structure: {num_valise_reps} valise reps (d={d}, r={r}); commutant dims, \
         <= {max_par} concurrent (~{:.1} GiB/DSU, {budget_gib} GiB budget)...",
        dsu_bytes as f64 / (1u64 << 30) as f64
    );

    // Commutant dimension per valise rep, in memory-bounded parallel batches.
    let mut dims: Vec<usize> = Vec::with_capacity(num_valise_reps);
    for chunk in all_reps.chunks(max_par) {
        let part: Vec<usize> = chunk.par_iter().map(|rep| commutant_dim(rep)).collect();
        dims.extend(part);
    }

    // Histogram of distinct commutant dimensions.
    let mut counts: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for &cd in &dims {
        *counts.entry(cd).or_insert(0) += 1;
    }
    let commutant_dim_histogram: Vec<(usize, usize)> =
        counts.iter().map(|(&k, &v)| (k, v)).collect();

    // Infer the Schur class for each distinct commutant dim. Isotypic iff
    // commutant_dim == r²·e with e ∈ {1,2,4}.
    let r2 = r * r;
    let classes: Vec<StructureClass> = commutant_dim_histogram
        .iter()
        .map(|&(cd, count)| {
            let e_opt = if r2 > 0 && cd % r2 == 0 {
                let e = cd / r2;
                if e == 1 || e == 2 || e == 4 {
                    Some(e)
                } else {
                    None
                }
            } else {
                None
            };
            // Real-type (e=1) multiplicity pattern: the MINIMAL-class decomposition
            // (fewest inequivalent irreducible types whose squared multiplicities
            // sum to the commutant dim). Reported when that minimal partition is
            // unique. The commutant dimension does not in general fix the pattern
            // (larger r admits several partitions of the same square-sum), so this
            // is the minimal-complexity decomposition consistent with it — which
            // matches the fully-determined small strata (k=7 [1,1], k=6 [2,2]).
            let patterns = real_multiplicity_patterns(r, cd);
            let (pattern, ambiguous_count) = if patterns.is_empty() {
                (None, 0)
            } else {
                let min_parts = patterns.iter().map(|p| p.len()).min().unwrap();
                let minimal: Vec<&Vec<usize>> =
                    patterns.iter().filter(|p| p.len() == min_parts).collect();
                if minimal.len() == 1 {
                    (Some(minimal[0].clone()), patterns.len())
                } else {
                    (None, patterns.len())
                }
            };
            let num_types = pattern.as_ref().map(|p| p.len());

            let (isotypic, da_dim, da, note) = match e_opt {
                Some(e) => {
                    let label = match e {
                        1 => "R",
                        2 => "C",
                        _ => "H",
                    };
                    (
                        true,
                        Some(e),
                        Some(label.to_string()),
                        format!("isotypic: {r} copies of one GR({dm},{n}) irreducible over {label} (commutant M_{r}({label}))"),
                    )
                }
                None => {
                    let detail = match &pattern {
                        Some(p) => format!(
                            "minimal real (e=1) decomposition: {} inequivalent class(es) with multiplicities {:?}{}",
                            p.len(), p,
                            if ambiguous_count > 1 {
                                format!(" (commutant dim alone admits {ambiguous_count} partitions; this is the fewest-class one)")
                            } else {
                                String::new()
                            }
                        ),
                        None => format!(
                            "no unique minimal real (e=1) pattern; {ambiguous_count} candidate partition(s) tie on class count"
                        ),
                    };
                    (
                        false,
                        None,
                        None,
                        format!(
                            "non-isotypic: commutant_dim {cd} = sum_t m_t²·e_t (r={r}); {detail}"
                        ),
                    )
                }
            };
            StructureClass {
                commutant_dim: cd,
                count,
                isotypic,
                division_algebra_dim: da_dim,
                division_algebra: da,
                multiplicity_pattern: pattern,
                num_irreducible_types: num_types,
                note,
            }
        })
        .collect();

    let stratum = StructureStratum {
        k: only_k,
        d,
        dmin: dm,
        r,
        num_codes: codes.len(),
        num_valise_reps,
        commutant_dim_histogram,
        classes,
        elapsed_secs: t0.elapsed().as_secs_f64(),
    };
    eprintln!(
        "decompose-structure: done in {:.2}s ({} distinct commutant dim(s))",
        stratum.elapsed_secs,
        stratum.classes.len()
    );
    println!("{}", serde_json::to_string_pretty(&stratum).expect("serialize structure"));
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
