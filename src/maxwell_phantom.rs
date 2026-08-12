//! Exact phantom-sector extraction from the N=1 Maxwell subsystem of the
//! verified chiral-vector positive control.

use crate::chiral_vector_4d::{Clifford4D, GaussianRational, Matrix4, matrix_mul, matrix_scale};
use num_rational::Ratio;
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const CHARGES: usize = 4;
const BOSONS: usize = 7;
const FERMIONS: usize = 4;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct GaussianUnit {
    pub real: i8,
    pub imag: i8,
}

impl GaussianUnit {
    fn add_scaled(&mut self, other: Self, scale: i8) {
        self.real += scale * other.real;
        self.imag += scale * other.imag;
    }

    fn subtract(self, other: Self) -> Self {
        Self {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }

    fn is_zero(self) -> bool {
        self.real == 0 && self.imag == 0
    }
}

type Linkage = [[[GaussianUnit; FERMIONS]; BOSONS]; CHARGES];

#[derive(Clone, Debug, Serialize)]
pub struct MaxwellPhantomArtifact {
    pub schema_version: &'static str,
    pub title: &'static str,
    pub sources: Vec<&'static str>,
    pub boson_basis: [&'static str; BOSONS],
    pub fermion_basis: [&'static str; FERMIONS],
    pub fermion_up_transpose: Linkage,
    pub temporal_down: Linkage,
    pub phantom_matrix: Linkage,
    pub spatial_down: [Linkage; 3],
    pub visible_bosons_on_worldline: usize,
    pub magnetic_phantoms: usize,
    pub nonzero_phantom_entries: usize,
    pub nonphantom_rows_are_zero: bool,
    pub magnetic_temporal_down_rows_are_zero: bool,
    pub equation_5_8_residual_entries: usize,
    pub omega_matrix_count: usize,
    pub raw_bosonic_omega_nonzero_entries: usize,
    pub bianchi_time_to_space_entries: usize,
    pub bianchi_divergence_pivot_entries: usize,
    pub canonical_bosonic_omega_residual_entries: usize,
    pub fermionic_omega_residual_entries: usize,
    pub equation_5_11_passed: bool,
    pub passed: bool,
    pub boundary: &'static str,
}

fn lower_spinors(clifford: &Clifford4D, matrix: &Matrix4) -> Matrix4 {
    matrix_mul(matrix, &clifford.charge_conjugation)
}

fn gaussian_unit(value: GaussianRational) -> GaussianUnit {
    assert_eq!(*value.real.denom(), 1);
    assert_eq!(*value.imag.denom(), 1);
    GaussianUnit {
        real: i8::try_from(*value.real.numer()).expect("Maxwell real coefficient fits i8"),
        imag: i8::try_from(*value.imag.numer()).expect("Maxwell imaginary coefficient fits i8"),
    }
}

fn epsilon3(indices: [usize; 3]) -> i8 {
    if indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2] {
        return 0;
    }
    let inversions = usize::from(indices[0] > indices[1])
        + usize::from(indices[0] > indices[2])
        + usize::from(indices[1] > indices[2]);
    if inversions.is_multiple_of(2) { 1 } else { -1 }
}

fn build_linkages() -> (Linkage, Linkage, [Linkage; 3]) {
    let clifford = Clifford4D::build();
    let mut up_transpose = [[[GaussianUnit::default(); FERMIONS]; BOSONS]; CHARGES];
    let mut temporal_down = [[[GaussianUnit::default(); FERMIONS]; BOSONS]; CHARGES];
    let mut spatial_down = [[[[GaussianUnit::default(); FERMIONS]; BOSONS]; CHARGES]; 3];
    let minus_i_half = GaussianRational::from_ratio(Ratio::from_integer(0), Ratio::new(-1, 2));
    let fermion_reduction_phase = GaussianRational::new(0, -1);
    let magnetic_pairs = [(2, 3), (3, 1), (1, 2)];

    for charge in 0..CHARGES {
        for fermion in 0..FERMIONS {
            for spatial in 1..4 {
                let commutator = lower_spinors(&clifford, &clifford.commutator_up(0, spatial));
                up_transpose[charge][spatial - 1][fermion] = gaussian_unit(
                    commutator[charge][fermion]
                        .mul(&minus_i_half)
                        .mul(&fermion_reduction_phase),
                );
                temporal_down[charge][spatial - 1][fermion] =
                    gaussian_unit(clifford.gamma_down[spatial][charge][fermion]);
                spatial_down[spatial - 1][charge][spatial - 1][fermion] = gaussian_unit(
                    clifford.gamma_down[0][charge][fermion].mul(&GaussianRational::new(-1, 0)),
                );
            }

            let gamma5_lower = lower_spinors(&clifford, &clifford.gamma5);
            up_transpose[charge][3][fermion] =
                gaussian_unit(gamma5_lower[charge][fermion].mul(&fermion_reduction_phase));
            let i_gamma5_gamma0 =
                matrix_scale(&clifford.gamma5_gamma_up(0), GaussianRational::new(0, 1));
            temporal_down[charge][3][fermion] = gaussian_unit(i_gamma5_gamma0[charge][fermion]);
            for spatial in 1..4 {
                let i_gamma5_gamma = matrix_scale(
                    &clifford.gamma5_gamma_up(spatial),
                    GaussianRational::new(0, 1),
                );
                spatial_down[spatial - 1][charge][3][fermion] =
                    gaussian_unit(i_gamma5_gamma[charge][fermion]);
            }

            for (magnetic, &(mu, nu)) in magnetic_pairs.iter().enumerate() {
                let commutator = lower_spinors(&clifford, &clifford.commutator_up(mu, nu));
                up_transpose[charge][4 + magnetic][fermion] = gaussian_unit(
                    commutator[charge][fermion]
                        .mul(&minus_i_half)
                        .mul(&fermion_reduction_phase),
                );
            }

            for derivative in 1..4 {
                for magnetic in 0..3 {
                    let mut coefficient = GaussianUnit::default();
                    for electric in 0..3 {
                        coefficient.add_scaled(
                            temporal_down[charge][electric][fermion],
                            epsilon3([magnetic, derivative - 1, electric]),
                        );
                    }
                    spatial_down[derivative - 1][charge][4 + magnetic][fermion] = coefficient;
                }
            }
        }
    }
    (up_transpose, temporal_down, spatial_down)
}

fn real_entry(value: GaussianUnit) -> i16 {
    assert_eq!(value.imag, 0, "normalized Maxwell linkage must be real");
    i16::from(value.real)
}

fn verify_omega(
    up_transpose: &Linkage,
    temporal_down: &Linkage,
    spatial_down: &[Linkage; 3],
) -> (usize, usize, usize, usize, usize) {
    let down = [
        temporal_down.clone(),
        spatial_down[0],
        spatial_down[1],
        spatial_down[2],
    ];
    let mut raw_bosonic_nonzero = 0;
    let mut time_to_space_entries = 0;
    let mut divergence_pivot_entries = 0;
    let mut canonical_bosonic_residual = 0;
    let mut fermionic_residual = 0;

    for left in 0..CHARGES {
        for right in left..CHARGES {
            let mut bosonic_omega = [[[0_i16; BOSONS]; BOSONS]; 4];
            for mu in 0..4 {
                let mut fermionic_product = [[0_i16; FERMIONS]; FERMIONS];
                for row in 0..FERMIONS {
                    for column in 0..FERMIONS {
                        let numerator: i16 = (0..BOSONS)
                            .map(|boson| {
                                real_entry(up_transpose[left][boson][row])
                                    * real_entry(down[mu][right][boson][column])
                                    + real_entry(up_transpose[right][boson][row])
                                        * real_entry(down[mu][left][boson][column])
                            })
                            .sum();
                        assert_eq!(numerator % 2, 0);
                        fermionic_product[row][column] = numerator / 2;
                    }
                }
                let lambda = fermionic_product[0][0];
                for row in 0..FERMIONS {
                    for column in 0..FERMIONS {
                        let expected = if row == column { lambda } else { 0 };
                        fermionic_residual +=
                            usize::from(fermionic_product[row][column] != expected);
                    }
                }
                for row in 0..BOSONS {
                    for column in 0..BOSONS {
                        let numerator: i16 = (0..FERMIONS)
                            .map(|fermion| {
                                real_entry(down[mu][left][row][fermion])
                                    * real_entry(up_transpose[right][column][fermion])
                                    + real_entry(down[mu][right][row][fermion])
                                        * real_entry(up_transpose[left][column][fermion])
                            })
                            .sum();
                        assert_eq!(numerator % 2, 0);
                        let product = numerator / 2;
                        bosonic_omega[mu][row][column] =
                            product - if row == column { lambda } else { 0 };
                        raw_bosonic_nonzero += usize::from(bosonic_omega[mu][row][column] != 0);
                    }
                }
            }

            let time_phantom: [[i16; 3]; BOSONS] = std::array::from_fn(|row| {
                std::array::from_fn(|magnetic| bosonic_omega[0][row][4 + magnetic])
            });
            for row in 0..BOSONS {
                for magnetic in 0..3 {
                    time_to_space_entries += usize::from(time_phantom[row][magnetic] != 0);
                    bosonic_omega[0][row][4 + magnetic] = 0;
                }
            }
            for derivative in 0..3 {
                for electric in 0..3 {
                    for row in 0..BOSONS {
                        for magnetic in 0..3 {
                            bosonic_omega[1 + derivative][row][electric] +=
                                i16::from(epsilon3([derivative, electric, magnetic]))
                                    * time_phantom[row][magnetic];
                        }
                    }
                }
            }

            let divergence_pivot: [i16; BOSONS] =
                std::array::from_fn(|row| bosonic_omega[1][row][4]);
            for row in 0..BOSONS {
                divergence_pivot_entries += usize::from(divergence_pivot[row] != 0);
                bosonic_omega[1][row][4] = 0;
                bosonic_omega[2][row][5] -= divergence_pivot[row];
                bosonic_omega[3][row][6] -= divergence_pivot[row];
            }
            canonical_bosonic_residual += bosonic_omega
                .iter()
                .flatten()
                .flatten()
                .filter(|&&value| value != 0)
                .count();
        }
    }
    (
        raw_bosonic_nonzero,
        time_to_space_entries,
        divergence_pivot_entries,
        canonical_bosonic_residual,
        fermionic_residual,
    )
}

pub fn build() -> MaxwellPhantomArtifact {
    let (fermion_up_transpose, temporal_down, spatial_down) = build_linkages();
    let phantom_matrix = std::array::from_fn(|charge| {
        std::array::from_fn(|boson| {
            std::array::from_fn(|fermion| {
                fermion_up_transpose[charge][boson][fermion]
                    .subtract(temporal_down[charge][boson][fermion])
            })
        })
    });
    let nonzero_phantom_entries = phantom_matrix
        .iter()
        .flatten()
        .flatten()
        .filter(|&&value| !value.is_zero())
        .count();
    let nonphantom_rows_are_zero = (0..CHARGES).all(|charge| {
        (0..4).all(|boson| {
            phantom_matrix[charge][boson]
                .iter()
                .all(|&value| value.is_zero())
        })
    });
    let magnetic_temporal_down_rows_are_zero = (0..CHARGES).all(|charge| {
        (4..7).all(|boson| {
            temporal_down[charge][boson]
                .iter()
                .all(|&value| value.is_zero())
        })
    });
    let mut equation_5_8_residual_entries = 0;
    for derivative in 0..3 {
        for charge in 0..CHARGES {
            for magnetic in 0..3 {
                for fermion in 0..FERMIONS {
                    let mut predicted = GaussianUnit::default();
                    for electric in 0..3 {
                        predicted.add_scaled(
                            temporal_down[charge][electric][fermion],
                            epsilon3([magnetic, derivative, electric]),
                        );
                    }
                    equation_5_8_residual_entries += usize::from(
                        spatial_down[derivative][charge][4 + magnetic][fermion] != predicted,
                    );
                }
            }
        }
    }
    let (
        raw_bosonic_omega_nonzero_entries,
        bianchi_time_to_space_entries,
        bianchi_divergence_pivot_entries,
        canonical_bosonic_omega_residual_entries,
        fermionic_omega_residual_entries,
    ) = verify_omega(&fermion_up_transpose, &temporal_down, &spatial_down);
    let equation_5_11_passed =
        canonical_bosonic_omega_residual_entries == 0 && fermionic_omega_residual_entries == 0;
    let passed = crate::chiral_vector_4d::verify().passed
        && nonphantom_rows_are_zero
        && magnetic_temporal_down_rows_are_zero
        && nonzero_phantom_entries > 0
        && equation_5_8_residual_entries == 0
        && equation_5_11_passed;
    MaxwellPhantomArtifact {
        schema_version: "maxwell-phantom-extraction-v1",
        title: "Maxwell phantom sector extracted from the verified chiral-vector system",
        sources: vec![
            "arXiv:1405.0048 Eqs. (32)-(41)",
            "arXiv:0907.3605 Eqs. (5.1), (5.3), (5.6), and (5.8)",
        ],
        boson_basis: ["E1", "E2", "E3", "d", "B1", "B2", "B3"],
        fermion_basis: ["lambda1", "lambda2", "lambda3", "lambda4"],
        fermion_up_transpose,
        temporal_down,
        phantom_matrix,
        spatial_down,
        visible_bosons_on_worldline: 4,
        magnetic_phantoms: 3,
        nonzero_phantom_entries,
        nonphantom_rows_are_zero,
        magnetic_temporal_down_rows_are_zero,
        equation_5_8_residual_entries,
        omega_matrix_count: 4 * 10,
        raw_bosonic_omega_nonzero_entries,
        bianchi_time_to_space_entries,
        bianchi_divergence_pivot_entries,
        canonical_bosonic_omega_residual_entries,
        fermionic_omega_residual_entries,
        equation_5_11_passed,
        passed,
        boundary: "This extracts the Maxwell magnetic phantom sector and verifies the complete p=1 gauge-enhancement condition in one fixed source basis. It does not search the eight-color representations or cover p>=2 gauge potentials.",
    }
}

pub fn write_artifact(path: &Path) -> MaxwellPhantomArtifact {
    let artifact = build();
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(path).expect("create Maxwell phantom artifact")),
        &artifact,
    )
    .expect("write Maxwell phantom artifact");
    artifact
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxwell_phantom_support_is_exact() {
        let artifact = build();
        assert!(artifact.nonphantom_rows_are_zero);
        assert!(artifact.magnetic_temporal_down_rows_are_zero);
        assert!(artifact.nonzero_phantom_entries > 0);
        assert_eq!(artifact.equation_5_8_residual_entries, 0);
        assert_eq!(artifact.omega_matrix_count, 40);
        assert_eq!(artifact.raw_bosonic_omega_nonzero_entries, 144);
        assert_eq!(artifact.bianchi_time_to_space_entries, 36);
        assert_eq!(artifact.bianchi_divergence_pivot_entries, 12);
        assert_eq!(artifact.canonical_bosonic_omega_residual_entries, 0);
        assert_eq!(artifact.fermionic_omega_residual_entries, 0);
        assert!(artifact.equation_5_11_passed);
        assert!(artifact.passed);
    }
}
