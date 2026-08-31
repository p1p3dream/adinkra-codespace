//! Independent component graviton/Riemann to gravitino-curl normalization.
//!
//! This oracle never calls Eq. (25), `Hhat`, Eq. (40), or a gauge-fixed
//! teleparallel section. At fixed null momentum it compares two independent
//! exact compositions from the raw 352-component gravitino frame:
//!
//! 1. hep-th/0101037 Eq. (41),
//!    `D_alpha h_mn = i[(Gamma_m C)_alpha,gamma psi_n^gamma
//!                       +(Gamma_n C)_alpha,gamma psi_m^gamma]`,
//!    followed by the target Pauli-Fierz curvature;
//! 2. the target gravitino curl followed by
//!    `D_alpha Rrepo_ab|cd = i[(p_a Gamma_b-p_b Gamma_a)C_cd
//!                              +(p_c Gamma_d-p_d Gamma_c)C_ab]`.
//!
//! The repository Riemann convention is twice the conventional curvature, so
//! the displayed coefficient is one rather than one half. Both constructions
//! use the same canonical Majorana basis but independent target operators.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use num_rational::Ratio;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::eleven_dimensional_free_complex::{
    ExactCoefficient, SparseExactMatrix, gravitino_complex, graviton_complex,
};
use crate::eleven_dimensional_majorana::{real_charge_conjugation, real_gamma_matrices};
use crate::eleven_dimensional_physical_curvature::ExactQi;

const VECTOR_DIMENSION: usize = 11;
const SPINOR_DIMENSION: usize = 32;
const SYMMETRIC_DIMENSION: usize = 66;
const PAIR_DIMENSION: usize = 55;
const FRAME_DIMENSION: usize = VECTOR_DIMENSION * SPINOR_DIMENSION;
const CURL_DIMENSION: usize = PAIR_DIMENSION * SPINOR_DIMENSION;
const RIEMANN_AMBIENT_DIMENSION: usize = PAIR_DIMENSION * PAIR_DIMENSION;
const D_RIEMANN_DIMENSION: usize = SPINOR_DIMENSION * RIEMANN_AMBIENT_DIMENSION;
const PRIME: u32 = 1_073_741_783;
const NULL_MOMENTUM: [i64; VECTOR_DIMENSION] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];

type SparseColumn = BTreeMap<usize, ExactQi>;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct ComponentGravityRelativeFiberReport {
    pub schema_version: &'static str,
    pub signature: &'static str,
    pub fixed_null_momentum: [i64; VECTOR_DIMENSION],
    pub raw_gravitino_frame_dimension: usize,
    pub curl_dimension: usize,
    pub riemann_ambient_dimension: usize,
    pub d_riemann_ambient_dimension: usize,
    pub component_metric_descendant_coefficient: &'static str,
    pub repository_to_conventional_riemann_ratio: &'static str,
    pub charge_row_adapter_residual_entries: usize,
    pub curvature_identity_residual_entries: usize,
    pub riemann_bianchi_residual_entries: usize,
    pub curl_bianchi_residual_entries: usize,
    pub component_curl_rank: usize,
    pub component_d_riemann_rank: usize,
    pub gravitino_gauge_kernel_dimension: usize,
    pub half_normalization_mutation_residual_entries: usize,
    pub time_metric_mutation_residual_entries: usize,
    pub omitted_pair_mutation_residual_entries: usize,
    pub metric_basis_sha256: String,
    pub riemann_basis_sha256: String,
    pub frame_basis_sha256: String,
    pub curl_basis_sha256: String,
    pub metric_descendant_sha256: String,
    pub d_riemann_stream_sha256: String,
    pub independent_a3_curl_fiber_artifact_sha256: String,
    pub source_sha256: BTreeMap<String, String>,
    pub uses_eq25: bool,
    pub uses_hhat: bool,
    pub uses_eq40: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

fn pairs() -> Vec<(usize, usize)> {
    (0..VECTOR_DIMENSION)
        .flat_map(|left| ((left + 1)..VECTOR_DIMENSION).map(move |right| (left, right)))
        .collect()
}

fn symmetric_pairs() -> Vec<(usize, usize)> {
    (0..VECTOR_DIMENSION)
        .flat_map(|left| (left..VECTOR_DIMENSION).map(move |right| (left, right)))
        .collect()
}

fn exact(coefficient: ExactCoefficient) -> ExactQi {
    ExactQi {
        real: Ratio::new(coefficient.real_numerator, coefficient.real_denominator),
        imaginary: Ratio::new(
            coefficient.imaginary_numerator,
            coefficient.imaginary_denominator,
        ),
    }
}

fn add(output: &mut SparseColumn, row: usize, value: ExactQi) {
    if value.is_zero() {
        return;
    }
    let entry = output.entry(row).or_insert_with(ExactQi::zero);
    entry.add_assign(&value);
    if entry.is_zero() {
        output.remove(&row);
    }
}

fn add_matrix_entry(columns: &mut [SparseColumn], column: usize, row: usize, value: ExactQi) {
    add(&mut columns[column], row, value);
}

fn mixed_lower_gamma_and_charge_adapter_residual() -> (Vec<Vec<Vec<i16>>>, usize) {
    let gammas = real_gamma_matrices();
    let charge = real_charge_conjugation();
    let mixed = (0..VECTOR_DIMENSION)
        .map(|axis| {
            let metric = if axis == 0 { -1_i16 } else { 1_i16 };
            let mut output = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
            for alpha in 0..SPINOR_DIMENSION {
                for spinor in 0..SPINOR_DIMENSION {
                    output[alpha][spinor] = metric * i16::from(gammas[axis][alpha][spinor]);
                }
            }
            output
        })
        .collect::<Vec<_>>();
    let mut residual = 0;
    for axis in 0..VECTOR_DIMENSION {
        let mut literal = vec![vec![0_i16; SPINOR_DIMENSION]; SPINOR_DIMENSION];
        for beta in 0..SPINOR_DIMENSION {
            for pivot in 0..SPINOR_DIMENSION {
                let c = i16::from(charge[beta][pivot]);
                if c == 0 {
                    continue;
                }
                for spinor in 0..SPINOR_DIMENSION {
                    literal[beta][spinor] += c * mixed[axis][pivot][spinor];
                }
            }
        }
        for alpha in 0..SPINOR_DIMENSION {
            for spinor in 0..SPINOR_DIMENSION {
                let adapted = (0..SPINOR_DIMENSION)
                    .map(|beta| i16::from(charge[beta][alpha]) * literal[beta][spinor])
                    .sum::<i16>();
                residual += usize::from(adapted != mixed[axis][alpha][spinor]);
            }
        }
    }
    (mixed, residual)
}

fn metric_descendant_columns(
    gamma_lower: &[Vec<Vec<i16>>],
    half_mutation: bool,
) -> Vec<SparseColumn> {
    let symmetric = symmetric_pairs();
    let symmetric_lookup = symmetric
        .iter()
        .enumerate()
        .map(|(ordinal, &(left, right))| ((left, right), ordinal))
        .collect::<BTreeMap<_, _>>();
    let mut columns = vec![SparseColumn::new(); FRAME_DIMENSION];
    let denominator = if half_mutation { 2 } else { 1 };
    for m in 0..VECTOR_DIMENSION {
        for n in m..VECTOR_DIMENSION {
            let h = symmetric_lookup[&(m, n)];
            for alpha in 0..SPINOR_DIMENSION {
                for spinor in 0..SPINOR_DIMENSION {
                    let left = gamma_lower[m][alpha][spinor];
                    if left != 0 {
                        add_matrix_entry(
                            &mut columns,
                            n * SPINOR_DIMENSION + spinor,
                            alpha * SYMMETRIC_DIMENSION + h,
                            ExactQi::from_rational(i64::from(left), denominator).times_i(),
                        );
                    }
                    let right = gamma_lower[n][alpha][spinor];
                    if right != 0 {
                        add_matrix_entry(
                            &mut columns,
                            m * SPINOR_DIMENSION + spinor,
                            alpha * SYMMETRIC_DIMENSION + h,
                            ExactQi::from_rational(i64::from(right), denominator).times_i(),
                        );
                    }
                }
            }
        }
    }
    columns
}

fn compose_real_operator(
    operator: &SparseExactMatrix,
    inner: &[SparseColumn],
    output_block: usize,
) -> Vec<SparseColumn> {
    let mut by_column = vec![Vec::new(); operator.columns()];
    for (row, column, coefficient) in operator.nonzero_coefficients() {
        by_column[column].push((row, exact(coefficient)));
    }
    inner
        .iter()
        .map(|column| {
            let mut output = SparseColumn::new();
            for (&pivot, value) in column {
                let block = pivot / operator.columns();
                let operator_column = pivot % operator.columns();
                for (row, coefficient) in &by_column[operator_column] {
                    let scaled = value.scaled(&coefficient.real);
                    add(&mut output, block * output_block + row, scaled);
                }
            }
            output
        })
        .collect()
}

fn curl_columns() -> Vec<SparseColumn> {
    let operator = &gravitino_complex(NULL_MOMENTUM).curvature;
    let mut output = vec![SparseColumn::new(); FRAME_DIMENSION];
    for (row, column, coefficient) in operator.nonzero_coefficients() {
        add(&mut output[column], row, exact(coefficient));
    }
    output
}

fn bridge_columns(
    gamma_lower: &[Vec<Vec<i16>>],
    omit_second_pair: bool,
    mutate_time_metric: bool,
) -> Vec<SparseColumn> {
    let pair_basis = pairs();
    let mut output = vec![SparseColumn::new(); CURL_DIMENSION];
    for (input_pair, &(u, v)) in pair_basis.iter().enumerate() {
        for input_spinor in 0..SPINOR_DIMENSION {
            let column = input_pair * SPINOR_DIMENSION + input_spinor;
            for (left_pair, &(a, b)) in pair_basis.iter().enumerate() {
                let first = [(a, b, 1_i64), (b, a, -1_i64)];
                for &(momentum_axis, gamma_axis, sign) in &first {
                    let momentum = NULL_MOMENTUM[momentum_axis];
                    if momentum == 0 {
                        continue;
                    }
                    for alpha in 0..SPINOR_DIMENSION {
                        let mut gamma = gamma_lower[gamma_axis][alpha][input_spinor];
                        if mutate_time_metric && gamma_axis == 0 {
                            gamma = -gamma;
                        }
                        if gamma != 0 {
                            let row = alpha * RIEMANN_AMBIENT_DIMENSION
                                + left_pair * PAIR_DIMENSION
                                + input_pair;
                            add(
                                &mut output[column],
                                row,
                                ExactQi::from_integer(sign * momentum * i64::from(gamma)).times_i(),
                            );
                        }
                    }
                }
            }
            if omit_second_pair {
                continue;
            }
            for (right_pair, &(c, d)) in pair_basis.iter().enumerate() {
                let second = [(c, d, 1_i64), (d, c, -1_i64)];
                for &(momentum_axis, gamma_axis, sign) in &second {
                    let momentum = NULL_MOMENTUM[momentum_axis];
                    if momentum == 0 {
                        continue;
                    }
                    for alpha in 0..SPINOR_DIMENSION {
                        let mut gamma = gamma_lower[gamma_axis][alpha][input_spinor];
                        if mutate_time_metric && gamma_axis == 0 {
                            gamma = -gamma;
                        }
                        if gamma != 0 {
                            let row = alpha * RIEMANN_AMBIENT_DIMENSION
                                + input_pair * PAIR_DIMENSION
                                + right_pair;
                            add(
                                &mut output[column],
                                row,
                                ExactQi::from_integer(sign * momentum * i64::from(gamma)).times_i(),
                            );
                        }
                    }
                }
            }
        }
    }
    output
}

fn compose_sparse(outer: &[SparseColumn], inner: &[SparseColumn]) -> Vec<SparseColumn> {
    inner
        .iter()
        .map(|column| {
            let mut output = SparseColumn::new();
            for (&pivot, value) in column {
                for (&row, coefficient) in &outer[pivot] {
                    let product = ExactQi {
                        real: &coefficient.real * &value.real
                            - &coefficient.imaginary * &value.imaginary,
                        imaginary: &coefficient.real * &value.imaginary
                            + &coefficient.imaginary * &value.real,
                    };
                    add(&mut output, row, product);
                }
            }
            output
        })
        .collect()
}

fn residual_entries(left: &[SparseColumn], right: &[SparseColumn]) -> usize {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            left.keys()
                .chain(right.keys())
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .filter(|row| left.get(row) != right.get(row))
                .count()
        })
        .sum()
}

fn bianchi_residual_entries(
    bianchi: &SparseExactMatrix,
    columns: &[SparseColumn],
    block_size: usize,
) -> usize {
    let mut by_column = vec![Vec::new(); bianchi.columns()];
    for (row, column, coefficient) in bianchi.nonzero_coefficients() {
        by_column[column].push((row, exact(coefficient)));
    }
    columns
        .iter()
        .map(|column| {
            let mut residual = SparseColumn::new();
            for (&pivot, value) in column {
                let block = pivot / block_size;
                let local = pivot % block_size;
                for (row, coefficient) in &by_column[local] {
                    let product = ExactQi {
                        real: &coefficient.real * &value.real
                            - &coefficient.imaginary * &value.imaginary,
                        imaginary: &coefficient.real * &value.imaginary
                            + &coefficient.imaginary * &value.real,
                    };
                    add(&mut residual, block * bianchi.rows() + row, product);
                }
            }
            residual.len()
        })
        .sum()
}

fn mod_inverse(value: u32) -> u32 {
    let mut base = u64::from(value);
    let mut exponent = PRIME - 2;
    let modulus = u64::from(PRIME);
    let mut output = 1_u64;
    while exponent != 0 {
        if exponent & 1 != 0 {
            output = output * base % modulus;
        }
        base = base * base % modulus;
        exponent >>= 1;
    }
    output as u32
}

fn ratio_mod(value: &Ratio<i64>) -> Result<u32, String> {
    let denominator = value.denom().rem_euclid(i64::from(PRIME)) as u32;
    if denominator == 0 {
        return Err("component gravity denominator is inadmissible".to_string());
    }
    let numerator = value.numer().rem_euclid(i64::from(PRIME)) as u32;
    Ok((u64::from(numerator) * u64::from(mod_inverse(denominator)) % u64::from(PRIME)) as u32)
}

fn qi_mod(value: &ExactQi) -> Result<u32, String> {
    let real = ratio_mod(&value.real)?;
    let imaginary = ratio_mod(&value.imaginary)?;
    if real != 0 && imaginary != 0 {
        return Err(
            "component gravity rank expected a purely real or imaginary stream".to_string(),
        );
    }
    Ok(if real != 0 { real } else { imaginary })
}

fn modular_rank(columns: &[SparseColumn]) -> Result<usize, String> {
    let mut pivots = BTreeMap::<usize, BTreeMap<usize, u32>>::new();
    for column in columns {
        let mut reduced = column
            .iter()
            .map(|(&row, value)| Ok((row, qi_mod(value)?)))
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        reduced.retain(|_, value| *value != 0);
        loop {
            let Some((&row, &value)) = reduced.first_key_value() else {
                break;
            };
            let Some(pivot) = pivots.get(&row) else {
                let inverse = mod_inverse(value);
                for entry in reduced.values_mut() {
                    *entry = (u64::from(*entry) * u64::from(inverse) % u64::from(PRIME)) as u32;
                }
                pivots.insert(row, reduced);
                break;
            };
            for (&target, &coefficient) in pivot {
                let subtraction = u64::from(value) * u64::from(coefficient) % u64::from(PRIME);
                let current = reduced.get(&target).copied().unwrap_or(0);
                let next = (u64::from(current) + u64::from(PRIME) - subtraction) % u64::from(PRIME);
                if next == 0 {
                    reduced.remove(&target);
                } else {
                    reduced.insert(target, next as u32);
                }
            }
        }
    }
    Ok(pivots.len())
}

fn basis_hash(domain: &[u8], coordinates: impl IntoIterator<Item = Vec<u8>>) -> String {
    let mut hash = Sha256::new();
    hash.update(domain);
    for coordinate in coordinates {
        hash.update((coordinate.len() as u64).to_le_bytes());
        hash.update(coordinate);
    }
    format!("{:x}", hash.finalize())
}

fn stream_hash(columns: &[SparseColumn]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"adynkra-11d-component-gravity-relative-stream-v1\0");
    for (column, entries) in columns.iter().enumerate() {
        for (&row, value) in entries {
            hash.update((column as u64).to_le_bytes());
            hash.update((row as u64).to_le_bytes());
            hash.update(value.real.numer().to_le_bytes());
            hash.update(value.real.denom().to_le_bytes());
            hash.update(value.imaginary.numer().to_le_bytes());
            hash.update(value.imaginary.denom().to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub(crate) fn build_component_gravity_relative_fiber_report()
-> Result<ComponentGravityRelativeFiberReport, String> {
    let (gamma_lower, charge_adapter_residual) = mixed_lower_gamma_and_charge_adapter_residual();
    let metric_descendant = metric_descendant_columns(&gamma_lower, false);
    let riemann = graviton_complex(NULL_MOMENTUM);
    let gravitino = gravitino_complex(NULL_MOMENTUM);
    let left = compose_real_operator(
        &riemann.curvature,
        &metric_descendant,
        RIEMANN_AMBIENT_DIMENSION,
    );
    let curl = curl_columns();
    let bridge = bridge_columns(&gamma_lower, false, false);
    let right = compose_sparse(&bridge, &curl);
    let residual = residual_entries(&left, &right);

    let half_left = compose_real_operator(
        &riemann.curvature,
        &metric_descendant_columns(&gamma_lower, true),
        RIEMANN_AMBIENT_DIMENSION,
    );
    let time_mutation = compose_sparse(&bridge_columns(&gamma_lower, false, true), &curl);
    let omitted_mutation = compose_sparse(&bridge_columns(&gamma_lower, true, false), &curl);
    let half_residual = residual_entries(&half_left, &right);
    let time_residual = residual_entries(&left, &time_mutation);
    let omitted_residual = residual_entries(&left, &omitted_mutation);

    let riemann_bianchi =
        bianchi_residual_entries(&riemann.bianchi, &left, RIEMANN_AMBIENT_DIMENSION);
    let curl_bianchi = bianchi_residual_entries(&gravitino.bianchi, &curl, CURL_DIMENSION);
    let curl_rank = modular_rank(&curl)?;
    let d_riemann_rank = modular_rank(&left)?;
    let a3_curl_fiber_path = Path::new("results/adynkra_11d_a3_curl_fiber_product.json");
    let independent_a3_curl_fiber_artifact_sha256 = file_sha256(a3_curl_fiber_path)?;
    if independent_a3_curl_fiber_artifact_sha256
        != "53f078a1189555734a9c48f674a0528f620460d0ffe8cd60d461f1533b13558a"
    {
        return Err("independent A3/curl fiber artifact changed".to_string());
    }
    let source_paths = [
        "src/eleven_dimensional_component_gravity_relative_fiber.rs",
        "src/eleven_dimensional_free_complex.rs",
        "src/eleven_dimensional_majorana.rs",
        "src/eleven_dimensional_physical_curvature.rs",
    ];
    let source_sha256 = source_paths
        .into_iter()
        .map(|path| Ok((path.to_string(), file_sha256(Path::new(path))?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;

    let symmetric = symmetric_pairs();
    let pair_basis = pairs();
    let metric_basis_sha256 = basis_hash(
        b"adynkra-11d-symmetric-metric-basis-v1\0",
        symmetric.iter().map(|&(a, b)| vec![a as u8, b as u8]),
    );
    let riemann_basis_sha256 = basis_hash(
        b"adynkra-11d-riemann-pair-pair-basis-v1\0",
        pair_basis.iter().flat_map(|&(a, b)| {
            pair_basis
                .iter()
                .map(move |&(c, d)| vec![a as u8, b as u8, c as u8, d as u8])
        }),
    );
    let frame_basis_sha256 = basis_hash(
        b"adynkra-11d-component-frame-basis-v1\0",
        (0..VECTOR_DIMENSION)
            .flat_map(|a| (0..SPINOR_DIMENSION).map(move |spinor| vec![a as u8, spinor as u8])),
    );
    let curl_basis_sha256 = basis_hash(
        b"adynkra-11d-component-curl-basis-v1\0",
        pair_basis.iter().flat_map(|&(a, b)| {
            (0..SPINOR_DIMENSION).map(move |spinor| vec![a as u8, b as u8, spinor as u8])
        }),
    );
    let passed = charge_adapter_residual == 0
        && residual == 0
        && riemann_bianchi == 0
        && curl_bianchi == 0
        && curl_rank == 320
        && d_riemann_rank == 320
        && FRAME_DIMENSION - curl_rank == 32
        && half_residual > 0
        && time_residual > 0
        && omitted_residual > 0;
    Ok(ComponentGravityRelativeFiberReport {
        schema_version: "adynkra-11d-component-gravity-relative-fiber-v1",
        signature: "mostly plus (-,+,...,+)",
        fixed_null_momentum: NULL_MOMENTUM,
        raw_gravitino_frame_dimension: FRAME_DIMENSION,
        curl_dimension: CURL_DIMENSION,
        riemann_ambient_dimension: RIEMANN_AMBIENT_DIMENSION,
        d_riemann_ambient_dimension: D_RIEMANN_DIMENSION,
        component_metric_descendant_coefficient: "+i",
        repository_to_conventional_riemann_ratio: "2:1",
        charge_row_adapter_residual_entries: charge_adapter_residual,
        curvature_identity_residual_entries: residual,
        riemann_bianchi_residual_entries: riemann_bianchi,
        curl_bianchi_residual_entries: curl_bianchi,
        component_curl_rank: curl_rank,
        component_d_riemann_rank: d_riemann_rank,
        gravitino_gauge_kernel_dimension: FRAME_DIMENSION - curl_rank,
        half_normalization_mutation_residual_entries: half_residual,
        time_metric_mutation_residual_entries: time_residual,
        omitted_pair_mutation_residual_entries: omitted_residual,
        metric_basis_sha256,
        riemann_basis_sha256,
        frame_basis_sha256,
        curl_basis_sha256,
        metric_descendant_sha256: stream_hash(&metric_descendant),
        d_riemann_stream_sha256: stream_hash(&left),
        independent_a3_curl_fiber_artifact_sha256,
        source_sha256,
        uses_eq25: false,
        uses_hhat: false,
        uses_eq40: false,
        passed,
        boundary: "This certifies the independent ordinary on-shell component normalization between the graviton Riemann descendant and the gravitino curl. It does not construct an Hhat source map, extend Eq41 off shell through J/X corrections, identify physical A3 inside Hhat, construct prepotential K into Hhat, or prove irreducibility.",
    })
}

pub(crate) fn write_component_gravity_relative_fiber_report(
    path: &Path,
) -> Result<ComponentGravityRelativeFiberReport, String> {
    let report = build_component_gravity_relative_fiber_report()?;
    if !report.passed {
        return Err("component gravity relative fiber failed its scientific gates".to_string());
    }
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("serialize component gravity relative fiber: {error}"))?;
    let scratch = path.with_extension("json.tmp");
    fs::write(&scratch, bytes).map_err(|error| format!("write {}: {error}", scratch.display()))?;
    fs::rename(&scratch, path).map_err(|error| {
        format!(
            "rename {} to {}: {error}",
            scratch.display(),
            path.display()
        )
    })?;
    Ok(report)
}

pub(crate) fn validate_component_gravity_relative_fiber_report(
    path: &Path,
) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let actual: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let expected = serde_json::to_value(build_component_gravity_relative_fiber_report()?)
        .map_err(|error| format!("serialize expected component gravity report: {error}"))?;
    if actual != expected {
        return Err("component gravity artifact differs from an independent rebuild".to_string());
    }
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_gravity_and_curl_relative_normalization_is_exact() {
        let report = build_component_gravity_relative_fiber_report().unwrap();
        eprintln!("COMPONENT_GRAVITY_RELATIVE {report:#?}");
        assert!(report.passed);
        assert_eq!(report.curvature_identity_residual_entries, 0);
        assert_eq!(report.component_curl_rank, 320);
        assert_eq!(report.component_d_riemann_rank, 320);
        assert!(report.half_normalization_mutation_residual_entries > 0);
        assert!(report.time_metric_mutation_residual_entries > 0);
        assert!(report.omitted_pair_mutation_residual_entries > 0);
    }

    #[test]
    #[ignore = "writes and independently rereads the report-last artifact"]
    fn write_and_reread_component_gravity_relative_fiber() {
        let path = Path::new("results/adynkra_11d_component_gravity_relative_fiber.json");
        let report = write_component_gravity_relative_fiber_report(path).unwrap();
        assert!(report.passed);
        let sha256 = validate_component_gravity_relative_fiber_report(path).unwrap();
        eprintln!("COMPONENT_GRAVITY_RELATIVE_ARTIFACT_SHA {sha256}");
    }
}
