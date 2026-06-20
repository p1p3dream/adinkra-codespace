use std::collections::BTreeMap;
use std::fs;
use std::time::Instant;

use rayon::prelude::*;

use crate::chromotopology::Chromotopology;
use crate::code::DoublyEvenCode;
use crate::dashing::DashingEnumerator;
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

#[derive(Debug, Clone, Serialize)]
pub struct FullPipelineOutput {
    pub n: usize,
    pub num_codes: usize,
    pub total_reps: usize,
    pub results: Vec<PipelineResult>,
    pub gadget_strata: Vec<GadgetStratum>,
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
    let t0 = Instant::now();

    let data = fs::read_to_string(json_path)
        .unwrap_or_else(|e| panic!("Failed to read codes JSON {json_path:?}: {e}"));
    let catalog: Catalog = serde_json::from_str(&data).unwrap_or_else(|e| {
        panic!("Failed to parse JSON {json_path:?}: {e}. Expected {{n, total_classes, codes:[...]}}")
    });
    let n = catalog.n;
    eprintln!("Loaded {} code classes (N={})", catalog.codes.len(), n);

    let per_code: Vec<(PipelineResult, Vec<HoloraumyData>)> = catalog
        .codes
        .par_iter()
        .enumerate()
        .map(|(idx, entry)| {
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
        elapsed_secs: elapsed,
    }
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
