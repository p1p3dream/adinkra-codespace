#![allow(dead_code)] // primitive-library module: much of its API surface is exercised by the test suite, not the binary main path

use std::collections::{HashMap, HashSet, VecDeque};

use crate::code::{rref, DoublyEvenCode};

fn reduce(u: u32, rref_gens: &[u32], pivot_positions: &[usize]) -> u32 {
    let mut v = u;
    for (i, &pivot) in pivot_positions.iter().enumerate() {
        if v & (1 << pivot) != 0 {
            v ^= rref_gens[i];
        }
    }
    v
}

/// Enumerates all 2^k equivalence classes of odd dashings on the Adinkra
/// chromotopology defined by a doubly-even code.
///
/// Algorithm:
///   1. Build the quotient graph I^N / C from RREF-reduced coset representatives.
///   2. BFS spanning tree assigns consistent "lift" values (cube preimages).
///   3. Tree edge dashings computed from lift staircase.
///   4. Non-tree edge values determined by face-constraint propagation.
///   5. Remaining k free variables form the cocycle basis for H^1.
pub struct DashingEnumerator {
    n: usize,
    k: usize,
    num_edges: usize,
    base_dashing: Vec<u8>,
    cocycle_basis: Vec<Vec<u8>>,
    coset_reps: Vec<u32>,
    rep_to_index: HashMap<u32, usize>,
    edge_list: Vec<(usize, usize, usize)>,
    rref_gens: Vec<u32>,
    pivot_positions: Vec<usize>,
}

impl DashingEnumerator {
    pub fn new(code: &DoublyEvenCode) -> Self {
        let n = code.n;
        let k = code.k();

        let rref_gens = rref(&code.generators);
        let pivot_positions: Vec<usize> = rref_gens
            .iter()
            .map(|g| g.trailing_zeros() as usize)
            .collect();
        let pivot_set: HashSet<usize> = pivot_positions.iter().copied().collect();
        let free_positions: Vec<usize> = (0..n).filter(|p| !pivot_set.contains(p)).collect();
        let num_free = n - k;

        let num_cosets = 1usize << num_free;
        let mut coset_reps = Vec::with_capacity(num_cosets);
        for mask in 0..num_cosets {
            let mut v = 0u32;
            for (bit_idx, &pos) in free_positions.iter().enumerate() {
                if mask & (1 << bit_idx) != 0 {
                    v |= 1 << pos;
                }
            }
            let r = reduce(v, &rref_gens, &pivot_positions);
            coset_reps.push(r);
        }
        coset_reps.sort_unstable();
        coset_reps.dedup();

        let rep_to_index: HashMap<u32, usize> = coset_reps
            .iter()
            .enumerate()
            .map(|(i, &r)| (r, i))
            .collect();

        // Build edge list
        let mut edge_list: Vec<(usize, usize, usize)> = Vec::new();
        let mut edge_set = HashSet::new();
        let mut edge_lookup: HashMap<(usize, usize, usize), usize> = HashMap::new();

        for color in 0..n {
            for &v in &coset_reps {
                let w = reduce(v ^ (1 << color), &rref_gens, &pivot_positions);
                let v_idx = rep_to_index[&v];
                let w_idx = rep_to_index[&w];
                if v_idx == w_idx {
                    continue;
                }
                let edge_key = (v_idx.min(w_idx), v_idx.max(w_idx), color);
                if edge_set.insert(edge_key) {
                    let idx = edge_list.len();
                    edge_list.push(edge_key);
                    edge_lookup.insert(edge_key, idx);
                }
            }
        }

        let num_edges = edge_list.len();

        // BFS spanning tree with lifts
        let num_vertices = coset_reps.len();
        let mut adj: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); num_vertices];
        for (edge_idx, &(v, w, color)) in edge_list.iter().enumerate() {
            adj[v].push((w, edge_idx, color));
            adj[w].push((v, edge_idx, color));
        }

        let mut visited = vec![false; num_vertices];
        let mut lift: Vec<u32> = vec![0u32; num_vertices];
        let mut is_tree_edge = vec![false; num_edges];

        visited[0] = true;
        lift[0] = coset_reps[0];

        let mut queue = VecDeque::new();
        queue.push_back(0usize);

        while let Some(u_idx) = queue.pop_front() {
            for &(w_idx, edge_idx, color) in &adj[u_idx] {
                if !visited[w_idx] {
                    visited[w_idx] = true;
                    lift[w_idx] = lift[u_idx] ^ (1 << color);
                    is_tree_edge[edge_idx] = true;
                    queue.push_back(w_idx);
                }
            }
        }

        // Compute tree edge dashing values via lift staircase
        let mut edge_constant = vec![0u8; num_edges];
        let mut edge_free_mask = vec![0u32; num_edges];
        let mut edge_determined = vec![false; num_edges];

        for (edge_idx, &(v_idx, _w_idx, color)) in edge_list.iter().enumerate() {
            if is_tree_edge[edge_idx] {
                let lv = lift[v_idx];
                let src = if lv & (1 << color) == 0 { lv } else { lv ^ (1 << color) };
                let mask = if color == 0 { 0 } else { (1u32 << color) - 1 };
                edge_constant[edge_idx] = ((src & mask).count_ones() % 2) as u8;
                edge_determined[edge_idx] = true;
            }
        }

        // Build face list for constraint propagation
        let mut faces: Vec<[usize; 4]> = Vec::new();
        for &v_rep in &coset_reps {
            let v_idx = rep_to_index[&v_rep];
            for color_i in 0..n {
                for color_j in (color_i + 1)..n {
                    let v_ei = reduce(v_rep ^ (1 << color_i), &rref_gens, &pivot_positions);
                    let v_ej = reduce(v_rep ^ (1 << color_j), &rref_gens, &pivot_positions);
                    let v_ei_ej = reduce(v_ei ^ (1 << color_j), &rref_gens, &pivot_positions);

                    let v_ei_idx = rep_to_index[&v_ei];
                    let v_ej_idx = rep_to_index[&v_ej];
                    let v_ei_ej_idx = rep_to_index[&v_ei_ej];

                    if v_idx > v_ei_idx || v_idx > v_ej_idx || v_idx > v_ei_ej_idx {
                        continue;
                    }

                    let e1 = Self::lookup_edge(&edge_lookup, v_idx, v_ei_idx, color_i);
                    let e2 = Self::lookup_edge(&edge_lookup, v_ei_idx, v_ei_ej_idx, color_j);
                    let e3 = Self::lookup_edge(&edge_lookup, v_idx, v_ej_idx, color_j);
                    let e4 = Self::lookup_edge(&edge_lookup, v_ej_idx, v_ei_ej_idx, color_i);

                    faces.push([e1, e2, e3, e4]);
                }
            }
        }

        // Build edge-to-faces index for efficient propagation
        let mut edge_to_faces: Vec<Vec<usize>> = vec![Vec::new(); num_edges];
        for (face_idx, face) in faces.iter().enumerate() {
            for &e in face {
                edge_to_faces[e].push(face_idx);
            }
        }

        // Face-constraint propagation with work queue
        let mut free_var_count = 0usize;
        let mut face_queue: VecDeque<usize> = (0..faces.len()).collect();
        let mut in_queue = vec![true; faces.len()];

        loop {
            // Process work queue
            while let Some(face_idx) = face_queue.pop_front() {
                in_queue[face_idx] = false;
                let face = &faces[face_idx];

                let mut undetermined_count = 0usize;
                let mut undetermined_edge = 0usize;
                for &e in face {
                    if !edge_determined[e] {
                        undetermined_count += 1;
                        undetermined_edge = e;
                    }
                }

                if undetermined_count == 1 {
                    let mut c = 1u8;
                    let mut fm = 0u32;
                    for &e in face {
                        if e != undetermined_edge {
                            c ^= edge_constant[e];
                            fm ^= edge_free_mask[e];
                        }
                    }
                    edge_constant[undetermined_edge] = c;
                    edge_free_mask[undetermined_edge] = fm;
                    edge_determined[undetermined_edge] = true;

                    for &fi in &edge_to_faces[undetermined_edge] {
                        if !in_queue[fi] {
                            in_queue[fi] = true;
                            face_queue.push_back(fi);
                        }
                    }
                }
            }

            // Check for undetermined edges
            match (0..num_edges).find(|&e| !edge_determined[e]) {
                None => break,
                Some(e) => {
                    edge_constant[e] = 0;
                    edge_free_mask[e] = 1u32 << free_var_count;
                    edge_determined[e] = true;
                    free_var_count += 1;

                    for &fi in &edge_to_faces[e] {
                        if !in_queue[fi] {
                            in_queue[fi] = true;
                            face_queue.push_back(fi);
                        }
                    }
                }
            }
        }

        debug_assert_eq!(
            free_var_count, k,
            "expected k={} free variables, got {}",
            k, free_var_count
        );

        let base_dashing = edge_constant.clone();

        let mut cocycle_basis = Vec::with_capacity(k);
        for j in 0..k {
            let cocycle: Vec<u8> = edge_free_mask
                .iter()
                .map(|&fm| ((fm >> j) & 1) as u8)
                .collect();
            cocycle_basis.push(cocycle);
        }

        DashingEnumerator {
            n,
            k,
            num_edges,
            base_dashing,
            cocycle_basis,
            coset_reps,
            rep_to_index,
            edge_list,
            rref_gens,
            pivot_positions,
        }
    }

    fn lookup_edge(
        edge_lookup: &HashMap<(usize, usize, usize), usize>,
        v: usize,
        w: usize,
        color: usize,
    ) -> usize {
        let key = (v.min(w), v.max(w), color);
        *edge_lookup
            .get(&key)
            .unwrap_or_else(|| panic!("edge ({}, {}, color={}) not found", key.0, key.1, color))
    }

    pub fn num_classes(&self) -> usize {
        1 << self.k
    }

    pub fn get_dashing(&self, class_index: usize) -> Vec<i8> {
        assert!(
            class_index < self.num_classes(),
            "class_index {} out of range [0, {})",
            class_index,
            self.num_classes()
        );

        let mut gf2 = self.base_dashing.clone();

        for (j, cocycle) in self.cocycle_basis.iter().enumerate() {
            if class_index & (1 << j) != 0 {
                for (e, val) in gf2.iter_mut().enumerate() {
                    *val ^= cocycle[e];
                }
            }
        }

        gf2.iter()
            .map(|&b| if b == 0 { 1i8 } else { -1i8 })
            .collect()
    }

    pub fn get_dashing_for_chromotopology(
        &self,
        class_index: usize,
        boson_reps: &[u32],
    ) -> Vec<i8> {
        let raw_dashing = self.get_dashing(class_index);

        let mut edge_lookup: HashMap<(usize, usize, usize), usize> = HashMap::new();
        for (idx, &(v, w, c)) in self.edge_list.iter().enumerate() {
            edge_lookup.insert((v, w, c), idx);
        }

        let d = boson_reps.len();
        let mut result = vec![1i8; self.n * d];

        for color in 0..self.n {
            for (rank, &rep) in boson_reps.iter().enumerate() {
                let v = reduce(rep, &self.rref_gens, &self.pivot_positions);
                let w = reduce(v ^ (1 << color), &self.rref_gens, &self.pivot_positions);
                let v_idx = self.rep_to_index[&v];
                let w_idx = self.rep_to_index[&w];
                let key = (v_idx.min(w_idx), v_idx.max(w_idx), color);
                if let Some(&edge_idx) = edge_lookup.get(&key) {
                    result[color * d + rank] = raw_dashing[edge_idx];
                }
            }
        }

        result
    }

    pub fn verify_odd(&self, dashing: &[i8]) -> bool {
        assert_eq!(dashing.len(), self.num_edges, "dashing length mismatch");

        let mut edge_index: HashMap<(usize, usize, usize), usize> = HashMap::new();
        for (idx, &(v, w, c)) in self.edge_list.iter().enumerate() {
            edge_index.insert((v, w, c), idx);
        }

        for &v in &self.coset_reps {
            let v_idx = self.rep_to_index[&v];

            for color_i in 0..self.n {
                for color_j in (color_i + 1)..self.n {
                    let v_ei = reduce(v ^ (1 << color_i), &self.rref_gens, &self.pivot_positions);
                    let v_ej = reduce(v ^ (1 << color_j), &self.rref_gens, &self.pivot_positions);
                    let v_ei_ej = reduce(
                        v_ei ^ (1 << color_j),
                        &self.rref_gens,
                        &self.pivot_positions,
                    );

                    let v_ei_idx = self.rep_to_index[&v_ei];
                    let v_ej_idx = self.rep_to_index[&v_ej];
                    let v_ei_ej_idx = self.rep_to_index[&v_ei_ej];

                    let e1 = self.find_edge(&edge_index, v_idx, v_ei_idx, color_i);
                    let e2 = self.find_edge(&edge_index, v_ei_idx, v_ei_ej_idx, color_j);
                    let e3 = self.find_edge(&edge_index, v_idx, v_ej_idx, color_j);
                    let e4 = self.find_edge(&edge_index, v_ej_idx, v_ei_ej_idx, color_i);

                    let product = dashing[e1] as i32
                        * dashing[e2] as i32
                        * dashing[e3] as i32
                        * dashing[e4] as i32;

                    if product != -1 {
                        return false;
                    }
                }
            }
        }

        true
    }

    fn find_edge(
        &self,
        edge_index: &HashMap<(usize, usize, usize), usize>,
        v_idx: usize,
        w_idx: usize,
        color: usize,
    ) -> usize {
        let key = (v_idx.min(w_idx), v_idx.max(w_idx), color);
        *edge_index
            .get(&key)
            .unwrap_or_else(|| panic!("edge ({}, {}, color={}) not found", key.0, key.1, color))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::DoublyEvenCode;

    fn trivial_n4() -> DoublyEvenCode {
        DoublyEvenCode::trivial(4)
    }

    fn code_4_1_4() -> DoublyEvenCode {
        DoublyEvenCode::new(4, vec![0b1111])
    }

    fn hamming_8_4() -> DoublyEvenCode {
        DoublyEvenCode::new(8, vec![0b11100001, 0b11010010, 0b10110100, 0b01111000])
    }

    #[test]
    fn trivial_code_one_class_and_odd() {
        let code = trivial_n4();
        let de = DashingEnumerator::new(&code);

        assert_eq!(de.num_classes(), 1);

        let d = de.get_dashing(0);
        assert!(de.verify_odd(&d), "base dashing should be odd");

        for &s in &d {
            assert!(s == 1 || s == -1);
        }
    }

    #[test]
    fn code_4_1_two_classes_both_odd_and_distinct() {
        let code = code_4_1_4();
        let de = DashingEnumerator::new(&code);

        assert_eq!(de.num_classes(), 2);

        let d0 = de.get_dashing(0);
        let d1 = de.get_dashing(1);

        assert!(de.verify_odd(&d0), "class 0 dashing should be odd");
        assert!(de.verify_odd(&d1), "class 1 dashing should be odd");
        assert_ne!(d0, d1, "the two classes should produce distinct dashings");
    }

    #[test]
    fn hamming_16_classes_all_odd_and_distinct() {
        let code = hamming_8_4();
        let de = DashingEnumerator::new(&code);

        assert_eq!(de.num_classes(), 16);

        let mut seen: HashSet<Vec<i8>> = HashSet::new();
        for i in 0..16 {
            let d = de.get_dashing(i);
            assert!(de.verify_odd(&d), "class {i} dashing should be odd");
            assert!(seen.insert(d), "class {i} produced a duplicate dashing");
        }
    }

    #[test]
    fn reduce_is_idempotent() {
        let code = hamming_8_4();
        let rref_gens = rref(&code.generators);
        let pivot_positions: Vec<usize> = rref_gens
            .iter()
            .map(|g| g.trailing_zeros() as usize)
            .collect();

        for v in 0..256u32 {
            let r1 = reduce(v, &rref_gens, &pivot_positions);
            let r2 = reduce(r1, &rref_gens, &pivot_positions);
            assert_eq!(r1, r2, "reduce is not idempotent for v={v:#010b}");
        }
    }

    #[test]
    fn reduce_codeword_to_zero() {
        let code = hamming_8_4();
        let rref_gens = rref(&code.generators);
        let pivot_positions: Vec<usize> = rref_gens
            .iter()
            .map(|g| g.trailing_zeros() as usize)
            .collect();

        for cw in code.codewords() {
            assert_eq!(
                reduce(cw, &rref_gens, &pivot_positions),
                0,
                "codeword {cw:#010b} did not reduce to 0"
            );
        }
    }

    #[test]
    fn edge_count_formula() {
        let cases: Vec<(DoublyEvenCode, &str)> = vec![
            (trivial_n4(), "trivial N=4"),
            (code_4_1_4(), "[4,1,4]"),
            (hamming_8_4(), "[8,4,4]"),
        ];

        for (code, label) in &cases {
            let de = DashingEnumerator::new(code);
            let expected = code.n * (1 << (code.n - code.k() - 1));
            assert_eq!(
                de.num_edges, expected,
                "{label}: expected {expected} edges, got {}",
                de.num_edges
            );
        }
    }

    #[test]
    fn num_classes_equals_two_to_the_k() {
        let cases: Vec<(DoublyEvenCode, &str)> = vec![
            (trivial_n4(), "trivial N=4"),
            (code_4_1_4(), "[4,1,4]"),
            (hamming_8_4(), "[8,4,4]"),
        ];

        for (code, label) in &cases {
            let de = DashingEnumerator::new(code);
            let expected = 1usize << code.k();
            assert_eq!(
                de.num_classes(),
                expected,
                "{label}: expected {expected} classes, got {}",
                de.num_classes()
            );
        }
    }
}
