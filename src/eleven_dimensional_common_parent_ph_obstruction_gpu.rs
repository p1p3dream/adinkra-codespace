//! Persistent CUDA substrate for common-parent `P_H` obstruction screens.
//!
//! The scientific candidate inventory is supplied by an immutable manifest.
//! Column width is runtime-configurable, so the engine is not restricted to
//! the already-closed scalar-factorizing line or the old 56/77 fixtures.

use std::collections::BTreeMap;
use std::time::Duration;
#[cfg(feature = "cuda")]
use std::time::Instant;

pub const PINNED_PRIMES: [u32; 3] = [1_073_741_783, 1_073_741_723, 1_073_741_719];
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
pub const AUGMENTED_RANK_ALLOWANCE: usize = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fp2 {
    pub real: u32,
    pub imaginary: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ThreePrimeFp2 {
    pub lane: [u32; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackedPhCooEntry {
    pub row_key: u64,
    pub column: u32,
    pub reserved: u32,
    pub value: ThreePrimeFp2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeviceStatus {
    pub ranks: [u32; 3],
    pub obstruction_prime: u32,
    pub obstruction_pivot_column: u32,
    pub stopped: u32,
    pub invalid: u32,
    pub obstruction_row_key: u64,
    pub batches_submitted: u64,
    pub input_entries: u64,
    pub reduced_entries: u64,
    pub rows_visited: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateBlock {
    pub name: String,
    pub columns: usize,
    pub semantic_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateManifest {
    pub schema_version: String,
    pub manifest_sha256: String,
    pub canonical_row_schema_sha256: String,
    pub denominator_ledger_sha256: String,
    pub blocks: Vec<CandidateBlock>,
    pub augmented_obstruction_columns: usize,
    pub expected_rank: usize,
    pub scalar_factorizing_line_is_negative_fixture_only: bool,
    pub cross_block_coupled_equations: bool,
}

impl CandidateManifest {
    pub fn candidate_columns(&self) -> Result<usize, String> {
        self.blocks.iter().try_fold(0_usize, |sum, block| {
            if block.columns == 0 || !is_sha256(&block.semantic_sha256) {
                return Err("candidate manifest has an empty or unbound block".to_string());
            }
            sum.checked_add(block.columns)
                .ok_or_else(|| "candidate manifest column count overflow".to_string())
        })
    }

    pub fn columns(&self) -> Result<usize, String> {
        self.candidate_columns()?
            .checked_add(self.augmented_obstruction_columns)
            .ok_or_else(|| "augmented candidate column count overflow".to_string())
    }

    pub fn validate(&self) -> Result<usize, String> {
        if self.schema_version != "adynkra-11d-common-parent-ph-candidates-v1"
            || !is_sha256(&self.manifest_sha256)
            || !is_sha256(&self.canonical_row_schema_sha256)
            || !is_sha256(&self.denominator_ledger_sha256)
            || !self.scalar_factorizing_line_is_negative_fixture_only
            || !self.cross_block_coupled_equations
            || self.augmented_obstruction_columns != 1
        {
            return Err("candidate manifest identity or scope is invalid".to_string());
        }
        let columns = self.columns()?;
        if self.expected_rank >= columns || columns > 4096 {
            return Err("candidate manifest rank bound is invalid".to_string());
        }
        Ok(columns)
    }

    pub fn direct_parent_leading_fixture() -> Self {
        let digest = "1111111111111111111111111111111111111111111111111111111111111111";
        Self {
            schema_version: "adynkra-11d-common-parent-ph-candidates-v1".to_string(),
            manifest_sha256: digest.to_string(),
            canonical_row_schema_sha256: digest.to_string(),
            denominator_ledger_sha256: digest.to_string(),
            blocks: [
                ("Hhat direct spinor", 12),
                ("graviton h", 2),
                ("three-form A3", 8),
                ("gravitino psi", 8),
            ]
            .into_iter()
            .map(|(name, columns)| CandidateBlock {
                name: name.to_string(),
                columns,
                semantic_sha256: digest.to_string(),
            })
            .collect(),
            augmented_obstruction_columns: 1,
            expected_rank: 30,
            scalar_factorizing_line_is_negative_fixture_only: true,
            cross_block_coupled_equations: true,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Checkpoint {
    pub schema_version: String,
    pub manifest_sha256: String,
    pub next_batch_ordinal: u64,
    pub last_row_key: Option<u64>,
    pub status: DeviceStatus,
    pub basis: Vec<Fp2>,
    pub pivot_for_column: Vec<i32>,
    pub pivot_columns: Vec<u32>,
    pub pivot_row_keys: Vec<u64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Heartbeat {
    pub schema_version: String,
    pub manifest_sha256: String,
    pub elapsed_seconds: f64,
    pub phase: String,
    pub batches_submitted: u64,
    pub input_entries: u64,
    pub reduced_entries: u64,
    pub rows_visited: u64,
    pub ranks: [u32; 3],
    pub obstruction_found: bool,
    pub entries_per_second: f64,
    pub rows_per_second: f64,
    pub resident_bytes: u64,
    pub high_water_bytes: u64,
    pub checkpoint_ordinal: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Batch {
    pub ordinal: u64,
    pub first_row_key: u64,
    pub last_row_key: u64,
    pub entries: Vec<PackedPhCooEntry>,
}

impl Batch {
    pub fn validate(&self, columns: usize, previous_last: Option<u64>) -> Result<(), String> {
        if self.entries.is_empty()
            || self.first_row_key > self.last_row_key
            || previous_last.is_some_and(|previous| self.first_row_key <= previous)
        {
            return Err("batch row interval is empty, overlapping, or out of order".to_string());
        }
        for entry in &self.entries {
            if entry.reserved != 0
                || usize::try_from(entry.column).unwrap_or(usize::MAX) >= columns
                || entry.row_key < self.first_row_key
                || entry.row_key > self.last_row_key
            {
                return Err("packed P_H entry violates its manifest or batch".to_string());
            }
            for (prime_index, prime) in PINNED_PRIMES.iter().copied().enumerate() {
                if entry.value.lane[2 * prime_index] >= prime
                    || entry.value.lane[2 * prime_index + 1] >= prime
                {
                    return Err("packed P_H entry has a noncanonical residue".to_string());
                }
            }
        }
        Ok(())
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn add_mod(left: u32, right: u32, prime: u32) -> u32 {
    ((u64::from(left) + u64::from(right)) % u64::from(prime)) as u32
}

fn subtract_mod(left: u32, right: u32, prime: u32) -> u32 {
    ((u64::from(left) + u64::from(prime) - u64::from(right)) % u64::from(prime)) as u32
}

fn multiply_mod(left: u32, right: u32, prime: u32) -> u32 {
    (u64::from(left) * u64::from(right) % u64::from(prime)) as u32
}

fn power_mod(mut base: u32, mut exponent: u32, prime: u32) -> u32 {
    let mut output = 1_u32;
    while exponent != 0 {
        if exponent & 1 != 0 {
            output = multiply_mod(output, base, prime);
        }
        base = multiply_mod(base, base, prime);
        exponent >>= 1;
    }
    output
}

fn fp2_subtract(left: Fp2, right: Fp2, prime: u32) -> Fp2 {
    Fp2 {
        real: subtract_mod(left.real, right.real, prime),
        imaginary: subtract_mod(left.imaginary, right.imaginary, prime),
    }
}

fn fp2_multiply(left: Fp2, right: Fp2, prime: u32) -> Fp2 {
    Fp2 {
        real: subtract_mod(
            multiply_mod(left.real, right.real, prime),
            multiply_mod(left.imaginary, right.imaginary, prime),
            prime,
        ),
        imaginary: add_mod(
            multiply_mod(left.real, right.imaginary, prime),
            multiply_mod(left.imaginary, right.real, prime),
            prime,
        ),
    }
}

fn fp2_inverse(value: Fp2, prime: u32) -> Fp2 {
    let norm = add_mod(
        multiply_mod(value.real, value.real, prime),
        multiply_mod(value.imaginary, value.imaginary, prime),
        prime,
    );
    assert_ne!(norm, 0);
    let inverse_norm = power_mod(norm, prime - 2, prime);
    Fp2 {
        real: multiply_mod(value.real, inverse_norm, prime),
        imaginary: multiply_mod(
            if value.imaginary == 0 {
                0
            } else {
                prime - value.imaginary
            },
            inverse_norm,
            prime,
        ),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuParityResult {
    pub status: DeviceStatus,
    pub pivot_columns: Vec<Vec<u32>>,
    pub pivot_row_keys: Vec<Vec<u64>>,
}

/// Reconstruct the modular right kernel from a downloaded retained-echelon
/// checkpoint.  Characteristic-zero lifting and exact Q(i) replay remain a
/// CPU proof boundary and are intentionally not performed on device.
pub fn modular_nullspace_from_checkpoint(
    checkpoint: &Checkpoint,
    columns: usize,
    expected_rank: usize,
) -> Result<Vec<Vec<Vec<Fp2>>>, String> {
    let maximum_rank = expected_rank + AUGMENTED_RANK_ALLOWANCE;
    if checkpoint.basis.len() != 3 * maximum_rank * columns
        || checkpoint.pivot_columns.len() != 3 * maximum_rank
        || checkpoint.status.invalid != 0
    {
        return Err("nullspace checkpoint shape is invalid".to_string());
    }
    let mut output = Vec::with_capacity(3);
    for (prime_index, prime) in PINNED_PRIMES.iter().copied().enumerate() {
        let rank = checkpoint.status.ranks[prime_index] as usize;
        if rank > maximum_rank {
            return Err("nullspace checkpoint rank exceeds retained storage".to_string());
        }
        let pivots = &checkpoint.pivot_columns
            [prime_index * maximum_rank..prime_index * maximum_rank + rank];
        if pivots.iter().any(|pivot| *pivot as usize >= columns)
            || pivots.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err("nullspace checkpoint pivots are not canonical".to_string());
        }
        let pivot_set = pivots
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let free_columns = (0..columns)
            .filter(|column| !pivot_set.contains(&(*column as u32)))
            .collect::<Vec<_>>();
        let mut prime_basis = Vec::with_capacity(free_columns.len());
        for free in free_columns {
            let mut vector = vec![Fp2::default(); columns];
            vector[free].real = 1;
            for basis_row in (0..rank).rev() {
                let pivot = pivots[basis_row] as usize;
                let stored_offset = (prime_index * maximum_rank + basis_row) * columns;
                let stored = &checkpoint.basis[stored_offset..stored_offset + columns];
                let mut sum = Fp2::default();
                for column in (pivot + 1)..columns {
                    let product = fp2_multiply(stored[column], vector[column], prime);
                    sum.real = add_mod(sum.real, product.real, prime);
                    sum.imaginary = add_mod(sum.imaginary, product.imaginary, prime);
                }
                vector[pivot] = Fp2 {
                    real: if sum.real == 0 { 0 } else { prime - sum.real },
                    imaginary: if sum.imaginary == 0 {
                        0
                    } else {
                        prime - sum.imaginary
                    },
                };
            }
            prime_basis.push(vector);
        }
        output.push(prime_basis);
    }
    Ok(output)
}

/// Exact CPU oracle for the packed sort/reduce and retained expected+1 RREF.
pub fn cpu_retained_rref(
    columns: usize,
    expected_rank: usize,
    entries: &[PackedPhCooEntry],
) -> Result<CpuParityResult, String> {
    if columns == 0 || expected_rank >= columns {
        return Err("CPU parity dimensions are invalid".to_string());
    }
    let mut reduced = BTreeMap::<(u64, u32), ThreePrimeFp2>::new();
    for entry in entries {
        if entry.reserved != 0 || entry.column as usize >= columns {
            return Err("CPU parity entry is invalid".to_string());
        }
        let target = reduced.entry((entry.row_key, entry.column)).or_default();
        for (prime_index, prime) in PINNED_PRIMES.iter().copied().enumerate() {
            target.lane[2 * prime_index] = add_mod(
                target.lane[2 * prime_index],
                entry.value.lane[2 * prime_index],
                prime,
            );
            target.lane[2 * prime_index + 1] = add_mod(
                target.lane[2 * prime_index + 1],
                entry.value.lane[2 * prime_index + 1],
                prime,
            );
        }
    }
    let mut by_row = BTreeMap::<u64, Vec<(u32, ThreePrimeFp2)>>::new();
    for ((row, column), value) in &reduced {
        by_row.entry(*row).or_default().push((*column, *value));
    }
    let maximum_rank = expected_rank + AUGMENTED_RANK_ALLOWANCE;
    let mut bases = vec![Vec::<Vec<Fp2>>::new(); 3];
    let mut pivots = vec![Vec::<u32>::new(); 3];
    let mut pivot_rows = vec![Vec::<u64>::new(); 3];
    let mut status = DeviceStatus {
        obstruction_prime: u32::MAX,
        obstruction_pivot_column: u32::MAX,
        obstruction_row_key: u64::MAX,
        input_entries: entries.len() as u64,
        reduced_entries: reduced.len() as u64,
        ..DeviceStatus::default()
    };
    for (row_key, sparse) in by_row {
        for (prime_index, prime) in PINNED_PRIMES.iter().copied().enumerate() {
            let mut row = vec![Fp2::default(); columns];
            for (column, value) in &sparse {
                row[*column as usize] = Fp2 {
                    real: value.lane[2 * prime_index],
                    imaginary: value.lane[2 * prime_index + 1],
                };
            }
            for (pivot, basis) in pivots[prime_index].iter().zip(&bases[prime_index]) {
                let factor = row[*pivot as usize];
                if factor != Fp2::default() {
                    for column in *pivot as usize..columns {
                        row[column] = fp2_subtract(
                            row[column],
                            fp2_multiply(factor, basis[column], prime),
                            prime,
                        );
                    }
                }
            }
            let Some(pivot) = row.iter().position(|value| *value != Fp2::default()) else {
                continue;
            };
            if bases[prime_index].len() >= maximum_rank {
                return Err("CPU retained basis exceeded expected+1".to_string());
            }
            let inverse = fp2_inverse(row[pivot], prime);
            for value in &mut row {
                *value = fp2_multiply(*value, inverse, prime);
            }
            bases[prime_index].push(row);
            pivots[prime_index].push(pivot as u32);
            pivot_rows[prime_index].push(row_key);
            status.ranks[prime_index] += 1;
            if status.ranks[prime_index] as usize > expected_rank && status.stopped == 0 {
                status.stopped = 1;
                status.obstruction_prime = prime_index as u32;
                status.obstruction_pivot_column = pivot as u32;
                status.obstruction_row_key = row_key;
            }
        }
        status.rows_visited += 1;
        if status.stopped != 0 {
            break;
        }
    }
    Ok(CpuParityResult {
        status,
        pivot_columns: pivots,
        pivot_row_keys: pivot_rows,
    })
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use std::ffi::{c_char, c_void, CStr};
    use std::ptr::NonNull;

    unsafe extern "C" {
        fn adynkra_common_parent_ph_obstruction_create(
            columns: u32,
            expected_rank: u32,
            batch_capacity: u64,
            device_hard_cap: u64,
            error: *mut c_char,
            error_capacity: u64,
        ) -> *mut c_void;
        fn adynkra_common_parent_ph_obstruction_submit(
            context: *mut c_void,
            batch_ordinal: u64,
            entries: *const PackedPhCooEntry,
            entry_count: u64,
            first_row_key: u64,
            last_row_key: u64,
            error: *mut c_char,
            error_capacity: u64,
        ) -> i32;
        fn adynkra_common_parent_ph_obstruction_poll(
            context: *mut c_void,
            status: *mut DeviceStatus,
            error: *mut c_char,
            error_capacity: u64,
        ) -> i32;
        fn adynkra_common_parent_ph_obstruction_checkpoint(
            context: *mut c_void,
            basis: *mut Fp2,
            basis_capacity: u64,
            pivot_for_column: *mut i32,
            pivot_map_capacity: u64,
            pivot_columns: *mut u32,
            pivot_capacity: u64,
            pivot_row_keys: *mut u64,
            row_key_capacity: u64,
            status: *mut DeviceStatus,
            error: *mut c_char,
            error_capacity: u64,
        ) -> i32;
        fn adynkra_common_parent_ph_obstruction_restore(
            context: *mut c_void,
            basis: *const Fp2,
            basis_count: u64,
            pivot_for_column: *const i32,
            pivot_map_count: u64,
            pivot_columns: *const u32,
            pivot_count: u64,
            pivot_row_keys: *const u64,
            row_key_count: u64,
            status: *const DeviceStatus,
            next_batch_ordinal: u64,
            last_row_key: u64,
            has_last_row: u32,
            error: *mut c_char,
            error_capacity: u64,
        ) -> i32;
        fn adynkra_common_parent_ph_obstruction_resident_bytes(context: *const c_void) -> u64;
        fn adynkra_common_parent_ph_obstruction_high_water_bytes(context: *const c_void) -> u64;
        fn adynkra_common_parent_ph_obstruction_primes(output: *mut u32);
        fn adynkra_common_parent_ph_obstruction_destroy(context: *mut c_void);
    }

    fn message(error: &[i8]) -> String {
        unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    pub struct Engine {
        context: NonNull<c_void>,
        manifest: CandidateManifest,
        columns: usize,
        batch_capacity: usize,
        input_entries: u64,
        started: Instant,
        last_heartbeat: Instant,
        checkpoint_ordinal: u64,
        last_row_key: Option<u64>,
    }

    impl Engine {
        pub fn create(
            manifest: CandidateManifest,
            batch_capacity: usize,
            device_hard_cap: u64,
        ) -> Result<Self, String> {
            let columns = manifest.validate()?;
            if batch_capacity == 0 {
                return Err("P_H batch capacity is zero".to_string());
            }
            let mut error = vec![0_i8; 1024];
            let context = unsafe {
                adynkra_common_parent_ph_obstruction_create(
                    columns as u32,
                    manifest.expected_rank as u32,
                    batch_capacity as u64,
                    device_hard_cap,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            };
            let context = NonNull::new(context).ok_or_else(|| message(&error))?;
            let mut device_primes = [0_u32; 3];
            unsafe { adynkra_common_parent_ph_obstruction_primes(device_primes.as_mut_ptr()) };
            if device_primes != PINNED_PRIMES {
                unsafe { adynkra_common_parent_ph_obstruction_destroy(context.as_ptr()) };
                return Err("P_H host/device prime mismatch".to_string());
            }
            let now = Instant::now();
            Ok(Self {
                context,
                manifest,
                columns,
                batch_capacity,
                input_entries: 0,
                started: now,
                last_heartbeat: now,
                checkpoint_ordinal: 0,
                last_row_key: None,
            })
        }

        pub fn adopt(
            manifest: CandidateManifest,
            batch_capacity: usize,
            device_hard_cap: u64,
            checkpoint: &Checkpoint,
        ) -> Result<Self, String> {
            let mut engine = Self::create(manifest, batch_capacity, device_hard_cap)?;
            if checkpoint.schema_version != "adynkra-11d-common-parent-ph-checkpoint-v1"
                || checkpoint.manifest_sha256 != engine.manifest.manifest_sha256
            {
                return Err("P_H checkpoint identity does not match the manifest".to_string());
            }
            let maximum_rank = engine.manifest.expected_rank + AUGMENTED_RANK_ALLOWANCE;
            if checkpoint.basis.len() != 3 * maximum_rank * engine.columns
                || checkpoint.pivot_for_column.len() != 3 * engine.columns
                || checkpoint.pivot_columns.len() != 3 * maximum_rank
                || checkpoint.pivot_row_keys.len() != 3 * maximum_rank
                || checkpoint.status.batches_submitted != checkpoint.next_batch_ordinal
                || checkpoint.status.invalid != 0
            {
                return Err("P_H checkpoint shape or status is invalid".to_string());
            }
            let mut error = vec![0_i8; 1024];
            let code = unsafe {
                adynkra_common_parent_ph_obstruction_restore(
                    engine.context.as_ptr(),
                    checkpoint.basis.as_ptr(),
                    checkpoint.basis.len() as u64,
                    checkpoint.pivot_for_column.as_ptr(),
                    checkpoint.pivot_for_column.len() as u64,
                    checkpoint.pivot_columns.as_ptr(),
                    checkpoint.pivot_columns.len() as u64,
                    checkpoint.pivot_row_keys.as_ptr(),
                    checkpoint.pivot_row_keys.len() as u64,
                    &checkpoint.status,
                    checkpoint.next_batch_ordinal,
                    checkpoint.last_row_key.unwrap_or_default(),
                    u32::from(checkpoint.last_row_key.is_some()),
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            };
            if code != 0 {
                return Err(format!("P_H CUDA restore {code}: {}", message(&error)));
            }
            engine.input_entries = checkpoint.status.input_entries;
            engine.checkpoint_ordinal = checkpoint.next_batch_ordinal;
            engine.last_row_key = checkpoint.last_row_key;
            Ok(engine)
        }

        pub fn submit(&mut self, batch: &Batch) -> Result<(), String> {
            if batch.ordinal != self.checkpoint_ordinal || batch.entries.len() > self.batch_capacity
            {
                return Err("P_H batch ordinal or capacity is invalid".to_string());
            }
            batch.validate(self.columns, self.last_row_key)?;
            let mut error = vec![0_i8; 1024];
            let status = unsafe {
                adynkra_common_parent_ph_obstruction_submit(
                    self.context.as_ptr(),
                    batch.ordinal,
                    batch.entries.as_ptr(),
                    batch.entries.len() as u64,
                    batch.first_row_key,
                    batch.last_row_key,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            };
            if status != 0 {
                return Err(format!("P_H CUDA submit {status}: {}", message(&error)));
            }
            self.input_entries += batch.entries.len() as u64;
            self.checkpoint_ordinal += 1;
            self.last_row_key = Some(batch.last_row_key);
            Ok(())
        }

        pub fn poll(&mut self, phase: &str) -> Result<(DeviceStatus, Heartbeat), String> {
            let mut status = DeviceStatus::default();
            let mut error = vec![0_i8; 1024];
            let code = unsafe {
                adynkra_common_parent_ph_obstruction_poll(
                    self.context.as_ptr(),
                    &mut status,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            };
            if code != 0 || status.invalid != 0 {
                return Err(format!("P_H CUDA poll {code}: {}", message(&error)));
            }
            status.input_entries = self.input_entries;
            let elapsed = self.started.elapsed().as_secs_f64().max(f64::EPSILON);
            let resident_bytes = unsafe {
                adynkra_common_parent_ph_obstruction_resident_bytes(self.context.as_ptr())
            };
            let high_water_bytes = unsafe {
                adynkra_common_parent_ph_obstruction_high_water_bytes(self.context.as_ptr())
            };
            Ok((
                status,
                Heartbeat {
                    schema_version: "adynkra-11d-common-parent-ph-heartbeat-v1".to_string(),
                    manifest_sha256: self.manifest.manifest_sha256.clone(),
                    elapsed_seconds: elapsed,
                    phase: phase.to_string(),
                    batches_submitted: self.checkpoint_ordinal,
                    input_entries: self.input_entries,
                    reduced_entries: status.reduced_entries,
                    rows_visited: status.rows_visited,
                    ranks: status.ranks,
                    obstruction_found: status.stopped != 0,
                    entries_per_second: self.input_entries as f64 / elapsed,
                    rows_per_second: status.rows_visited as f64 / elapsed,
                    resident_bytes,
                    high_water_bytes,
                    checkpoint_ordinal: self.checkpoint_ordinal,
                },
            ))
        }

        pub fn heartbeat_due(&self) -> bool {
            self.last_heartbeat.elapsed() >= HEARTBEAT_INTERVAL
        }

        /// Run canonical batches through the two-slot pipeline.  Polling and
        /// checkpoint publication occur at least every five seconds at batch
        /// boundaries.  Polling every second submitted slot also gives the
        /// device an exact witness-first stop point without disabling overlap.
        pub fn run_batches<I, H, C>(
            &mut self,
            batches: I,
            mut publish_heartbeat: H,
            mut publish_checkpoint: C,
        ) -> Result<DeviceStatus, String>
        where
            I: IntoIterator<Item = Batch>,
            H: FnMut(&Heartbeat) -> Result<(), String>,
            C: FnMut(&Checkpoint) -> Result<(), String>,
        {
            for batch in batches {
                self.submit(&batch)?;
                let heartbeat_due = self.heartbeat_due();
                let poll_boundary = self.checkpoint_ordinal % 2 == 0 || heartbeat_due;
                if poll_boundary {
                    let (status, heartbeat) = self.poll("sort_reduce_retained_rref")?;
                    if heartbeat_due || status.stopped != 0 {
                        publish_heartbeat(&heartbeat)?;
                        let checkpoint = self.checkpoint()?;
                        publish_checkpoint(&checkpoint)?;
                        self.last_heartbeat = Instant::now();
                    }
                    if status.stopped != 0 {
                        return Ok(status);
                    }
                }
            }
            let (status, heartbeat) = self.poll("finalize")?;
            publish_heartbeat(&heartbeat)?;
            let checkpoint = self.checkpoint()?;
            publish_checkpoint(&checkpoint)?;
            self.last_heartbeat = Instant::now();
            Ok(status)
        }

        pub fn checkpoint(&mut self) -> Result<Checkpoint, String> {
            let maximum_rank = self.manifest.expected_rank + AUGMENTED_RANK_ALLOWANCE;
            let mut basis = vec![Fp2::default(); 3 * maximum_rank * self.columns];
            let mut pivot_for_column = vec![-1_i32; 3 * self.columns];
            let mut pivot_columns = vec![0_u32; 3 * maximum_rank];
            let mut pivot_row_keys = vec![0_u64; 3 * maximum_rank];
            let mut status = DeviceStatus::default();
            let mut error = vec![0_i8; 1024];
            let code = unsafe {
                adynkra_common_parent_ph_obstruction_checkpoint(
                    self.context.as_ptr(),
                    basis.as_mut_ptr(),
                    basis.len() as u64,
                    pivot_for_column.as_mut_ptr(),
                    pivot_for_column.len() as u64,
                    pivot_columns.as_mut_ptr(),
                    pivot_columns.len() as u64,
                    pivot_row_keys.as_mut_ptr(),
                    pivot_row_keys.len() as u64,
                    &mut status,
                    error.as_mut_ptr(),
                    error.len() as u64,
                )
            };
            if code != 0 || status.invalid != 0 {
                return Err(format!("P_H CUDA checkpoint {code}: {}", message(&error)));
            }
            status.input_entries = self.input_entries;
            Ok(Checkpoint {
                schema_version: "adynkra-11d-common-parent-ph-checkpoint-v1".to_string(),
                manifest_sha256: self.manifest.manifest_sha256.clone(),
                next_batch_ordinal: self.checkpoint_ordinal,
                last_row_key: self.last_row_key,
                status,
                basis,
                pivot_for_column,
                pivot_columns,
                pivot_row_keys,
            })
        }
    }

    impl Drop for Engine {
        fn drop(&mut self) {
            unsafe { adynkra_common_parent_ph_obstruction_destroy(self.context.as_ptr()) };
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::Engine;

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(row: u64, column: u32, real: u32) -> PackedPhCooEntry {
        let mut value = ThreePrimeFp2::default();
        for prime_index in 0..3 {
            value.lane[2 * prime_index] = real;
        }
        PackedPhCooEntry {
            row_key: row,
            column,
            reserved: 0,
            value,
        }
    }

    #[test]
    fn abi_sizes_are_pinned() {
        assert_eq!(std::mem::size_of::<Fp2>(), 8);
        assert_eq!(std::mem::size_of::<ThreePrimeFp2>(), 24);
        assert_eq!(std::mem::size_of::<PackedPhCooEntry>(), 40);
        assert_eq!(std::mem::size_of::<DeviceStatus>(), 72);
    }

    #[test]
    fn variable_direct_parent_manifest_has_all_leading_blocks() {
        let manifest = CandidateManifest::direct_parent_leading_fixture();
        assert_eq!(manifest.candidate_columns().unwrap(), 30);
        assert_eq!(manifest.validate().unwrap(), 31);
        assert_eq!(
            manifest
                .blocks
                .iter()
                .map(|block| block.columns)
                .collect::<Vec<_>>(),
            [12, 2, 8, 8]
        );
        assert!(manifest.cross_block_coupled_equations);
        assert!(manifest.scalar_factorizing_line_is_negative_fixture_only);
    }

    #[test]
    fn duplicate_reduce_and_expected_plus_one_witness_are_exact() {
        let mut entries = Vec::new();
        for row in 0..5_u64 {
            entries.push(entry(row, row as u32, 2));
            entries.push(entry(row, row as u32, 3));
        }
        let result = cpu_retained_rref(6, 4, &entries).unwrap();
        assert_eq!(result.status.ranks, [5, 5, 5]);
        assert_eq!(result.status.stopped, 1);
        assert_eq!(result.status.obstruction_row_key, 4);
        assert_eq!(result.status.obstruction_pivot_column, 4);
        assert_eq!(result.status.reduced_entries, 5);
        assert_eq!(result.pivot_columns[0], [0, 1, 2, 3, 4]);
    }

    #[test]
    fn scalar_line_is_only_a_negative_parity_fixture() {
        let entries = (0..8).map(|row| entry(row, 0, 1)).collect::<Vec<_>>();
        let result = cpu_retained_rref(2, 1, &entries).unwrap();
        assert_eq!(result.status.ranks, [1, 1, 1]);
        assert_eq!(result.status.stopped, 0);
    }

    #[test]
    fn cuda_benchmark_vandermonde_contract_has_the_same_witness() {
        let mut entries = Vec::new();
        for row in 0..100_u64 {
            for column in 0..20_u32 {
                let mut packed = PackedPhCooEntry {
                    row_key: row,
                    column,
                    reserved: 0,
                    value: ThreePrimeFp2::default(),
                };
                for (prime_index, prime) in PINNED_PRIMES.iter().copied().enumerate() {
                    packed.value.lane[2 * prime_index] =
                        power_mod((row as u32 + 1) % prime, column, prime);
                }
                entries.push(packed);
            }
        }
        let result = cpu_retained_rref(24, 12, &entries).unwrap();
        assert_eq!(result.status.ranks, [13, 13, 13]);
        assert_eq!(result.status.obstruction_row_key, 12);
        assert_eq!(result.status.obstruction_pivot_column, 12);
    }

    #[test]
    fn checkpoint_echelon_reconstructs_modular_kernel() {
        let columns = 3;
        let expected_rank = 2;
        let maximum_rank = 3;
        let mut checkpoint = Checkpoint {
            schema_version: "adynkra-11d-common-parent-ph-checkpoint-v1".to_string(),
            status: DeviceStatus {
                ranks: [2, 2, 2],
                ..DeviceStatus::default()
            },
            basis: vec![Fp2::default(); 3 * maximum_rank * columns],
            pivot_columns: vec![0; 3 * maximum_rank],
            ..Checkpoint::default()
        };
        for prime_index in 0..3 {
            let offset = prime_index * maximum_rank * columns;
            checkpoint.basis[offset] = Fp2 {
                real: 1,
                imaginary: 0,
            };
            checkpoint.basis[offset + 1] = Fp2 {
                real: 2,
                imaginary: 0,
            };
            checkpoint.basis[offset + 2] = Fp2 {
                real: 3,
                imaginary: 0,
            };
            checkpoint.basis[offset + columns + 1] = Fp2 {
                real: 1,
                imaginary: 0,
            };
            checkpoint.basis[offset + columns + 2] = Fp2 {
                real: 4,
                imaginary: 0,
            };
            checkpoint.pivot_columns[prime_index * maximum_rank] = 0;
            checkpoint.pivot_columns[prime_index * maximum_rank + 1] = 1;
        }
        let kernels =
            modular_nullspace_from_checkpoint(&checkpoint, columns, expected_rank).unwrap();
        for (prime_index, kernel) in kernels.iter().enumerate() {
            let prime = PINNED_PRIMES[prime_index];
            assert_eq!(kernel.len(), 1);
            assert_eq!(kernel[0][2].real, 1);
            assert_eq!(kernel[0][1].real, prime - 4);
            assert_eq!(kernel[0][0].real, 5);
        }
    }
}
