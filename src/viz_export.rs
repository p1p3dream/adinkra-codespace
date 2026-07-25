use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::code::DoublyEvenCode;
use crate::dashing::DashingEnumerator;

const MAGIC: &[u8; 4] = b"AD3D";
const FORMAT_VERSION: u16 = 1;

#[derive(Debug, Deserialize)]
struct Catalog {
    n: usize,
    codes: Vec<CodeEntry>,
}

#[derive(Debug, Deserialize)]
struct CodeEntry {
    index: usize,
    k: usize,
    generators_raw: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub struct ExportManifest {
    schema: &'static str,
    version: u16,
    n: usize,
    code_count: usize,
    total_dashing_classes: usize,
    assets: Vec<AssetRecord>,
}

#[derive(Debug, Serialize)]
struct AssetRecord {
    code_index: usize,
    k: usize,
    vertices: usize,
    edges: usize,
    dashing_classes: usize,
    file: String,
    bytes: usize,
}

pub fn export(catalog_path: &str, output_dir: &str) -> ExportManifest {
    let catalog_text = fs::read_to_string(catalog_path)
        .unwrap_or_else(|e| panic!("failed to read catalog {catalog_path}: {e}"));
    let catalog: Catalog = serde_json::from_str(&catalog_text)
        .unwrap_or_else(|e| panic!("failed to parse catalog {catalog_path}: {e}"));

    assert!(catalog.n <= 16, "3D binary format supports n <= 16");
    let output = Path::new(output_dir);
    fs::create_dir_all(output)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", output.display()));

    let mut assets = Vec::with_capacity(catalog.codes.len());
    let mut total_dashing_classes = 0usize;

    for entry in &catalog.codes {
        let code = DoublyEvenCode::new(catalog.n, entry.generators_raw.clone());
        assert!(code.is_valid(), "catalog code {} is invalid", entry.index);
        assert_eq!(
            code.k(),
            entry.k,
            "catalog k mismatch at code {}",
            entry.index
        );

        let enumerator = DashingEnumerator::new(&code);
        let packed = enumerator.packed_edge_dashings();
        let vertices = 1usize << (catalog.n - entry.k);
        let edges = catalog.n * vertices / 2;
        assert_eq!(
            packed.len(),
            edges * 2,
            "edge count mismatch at code {}",
            entry.index
        );

        let filename = format!("{:03}.ad3d", entry.index);
        let path = output.join(&filename);
        write_asset(
            &path,
            catalog.n,
            entry.k,
            entry.index,
            vertices,
            edges,
            &packed,
        );
        let bytes = 20 + packed.len();
        let dashing_classes = 1usize << entry.k;
        total_dashing_classes += dashing_classes;
        assets.push(AssetRecord {
            code_index: entry.index,
            k: entry.k,
            vertices,
            edges,
            dashing_classes,
            file: filename,
            bytes,
        });
    }

    let manifest = ExportManifest {
        schema: "adinkra-3d-dashing-assets",
        version: FORMAT_VERSION,
        n: catalog.n,
        code_count: assets.len(),
        total_dashing_classes,
        assets,
    };
    let manifest_path = output.join("manifest.json");
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).expect("serialize 3D manifest");
    write_atomic(&manifest_path, &manifest_bytes);
    manifest
}

fn write_asset(
    path: &Path,
    n: usize,
    k: usize,
    code_index: usize,
    vertices: usize,
    edges: usize,
    packed: &[u8],
) {
    let mut bytes = Vec::with_capacity(20 + packed.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes.push(n as u8);
    bytes.push(k as u8);
    bytes.extend_from_slice(&(code_index as u16).to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&(vertices as u32).to_le_bytes());
    bytes.extend_from_slice(&(edges as u32).to_le_bytes());
    bytes.extend_from_slice(packed);
    write_atomic(path, &bytes);
}

fn write_atomic(path: &Path, bytes: &[u8]) {
    let mut temporary = PathBuf::from(path);
    let temporary_name = format!(
        ".{}.tmp",
        path.file_name().and_then(|v| v.to_str()).unwrap_or("asset")
    );
    temporary.set_file_name(temporary_name);

    let mut file = fs::File::create(&temporary)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", temporary.display()));
    file.write_all(bytes)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", temporary.display()));
    file.sync_all()
        .unwrap_or_else(|e| panic!("failed to sync {}: {e}", temporary.display()));
    fs::rename(&temporary, path).unwrap_or_else(|e| {
        panic!(
            "failed to move {} to {}: {e}",
            temporary.display(),
            path.display()
        )
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_header_and_payload_are_stable() {
        let root = std::env::temp_dir().join(format!(
            "adinkra-3d-export-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("sample.ad3d");
        write_asset(&path, 4, 1, 7, 8, 16, &[0, 1, 1, 0]);
        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], MAGIC);
        assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), FORMAT_VERSION);
        assert_eq!(bytes[6], 4);
        assert_eq!(bytes[7], 1);
        assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), 7);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 8);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 16);
        assert_eq!(&bytes[20..], &[0, 1, 1, 0]);
        fs::remove_dir_all(root).unwrap();
    }
}
