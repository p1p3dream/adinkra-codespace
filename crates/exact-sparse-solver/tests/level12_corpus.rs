use adynkra_exact_sparse::level12::build_level12_matrix;
use adynkra_exact_sparse::{CsrMatrix, PRIME, field_from_i64};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const CHECKPOINT: &str = "results/adynkra_11d_level12_second_momentum_kernel_generation.json";
const CHECKPOINT_SCHEMA: &str = "adynkra-11d-level12-second-momentum-kernel-generation-v1";
const COMPLETED_SYSTEMS: u64 = 17;
const COMPLETED_KERNELS: u64 = 39;

#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Json>),
    Object(BTreeMap<String, Json>),
}

impl Json {
    fn field(&self, name: &str) -> &Json {
        self.as_object()
            .get(name)
            .unwrap_or_else(|| panic!("missing JSON field {name}"))
    }

    fn as_object(&self) -> &BTreeMap<String, Json> {
        match self {
            Self::Object(value) => value,
            _ => panic!("expected JSON object, got {self:?}"),
        }
    }

    fn as_array(&self) -> &[Json] {
        match self {
            Self::Array(value) => value,
            _ => panic!("expected JSON array, got {self:?}"),
        }
    }

    fn as_str(&self) -> &str {
        match self {
            Self::String(value) => value,
            _ => panic!("expected JSON string, got {self:?}"),
        }
    }

    fn as_u64(&self) -> u64 {
        match self {
            Self::Number(value) => value
                .parse()
                .unwrap_or_else(|_| panic!("expected unsigned integer, got {value}")),
            _ => panic!("expected JSON number, got {self:?}"),
        }
    }

    fn as_bool(&self) -> bool {
        match self {
            Self::Bool(value) => *value,
            _ => panic!("expected JSON boolean, got {self:?}"),
        }
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> JsonParser<'a> {
    fn parse(input: &'a str) -> Json {
        let mut parser = Self {
            bytes: input.as_bytes(),
            cursor: 0,
        };
        let value = parser.value();
        parser.whitespace();
        assert_eq!(parser.cursor, parser.bytes.len(), "trailing JSON input");
        value
    }

    fn value(&mut self) -> Json {
        self.whitespace();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Json::String(self.string()),
            Some(b't') => {
                self.literal(b"true");
                Json::Bool(true)
            }
            Some(b'f') => {
                self.literal(b"false");
                Json::Bool(false)
            }
            Some(b'n') => {
                self.literal(b"null");
                Json::Null
            }
            Some(b'-' | b'0'..=b'9') => Json::Number(self.number()),
            other => panic!("unexpected JSON byte {other:?} at {}", self.cursor),
        }
    }

    fn object(&mut self) -> Json {
        self.expect(b'{');
        let mut fields = BTreeMap::new();
        self.whitespace();
        if self.take(b'}') {
            return Json::Object(fields);
        }
        loop {
            self.whitespace();
            let key = self.string();
            self.whitespace();
            self.expect(b':');
            let previous = fields.insert(key, self.value());
            assert!(previous.is_none(), "duplicate JSON object key");
            self.whitespace();
            if self.take(b'}') {
                return Json::Object(fields);
            }
            self.expect(b',');
        }
    }

    fn array(&mut self) -> Json {
        self.expect(b'[');
        let mut values = Vec::new();
        self.whitespace();
        if self.take(b']') {
            return Json::Array(values);
        }
        loop {
            values.push(self.value());
            self.whitespace();
            if self.take(b']') {
                return Json::Array(values);
            }
            self.expect(b',');
        }
    }

    fn string(&mut self) -> String {
        self.expect(b'"');
        let mut output = String::new();
        loop {
            let byte = self.next().expect("unterminated JSON string");
            match byte {
                b'"' => return output,
                b'\\' => match self.next().expect("unterminated JSON escape") {
                    b'"' => output.push('"'),
                    b'\\' => output.push('\\'),
                    b'/' => output.push('/'),
                    b'b' => output.push('\u{0008}'),
                    b'f' => output.push('\u{000c}'),
                    b'n' => output.push('\n'),
                    b'r' => output.push('\r'),
                    b't' => output.push('\t'),
                    b'u' => {
                        let mut code = 0_u32;
                        for _ in 0..4 {
                            code = code * 16 + hex(self.next().expect("short Unicode escape"));
                        }
                        output.push(char::from_u32(code).expect("invalid Unicode escape"));
                    }
                    escape => panic!("invalid JSON escape {escape}"),
                },
                0x00..=0x1f => panic!("control byte in JSON string"),
                0x20..=0x7f => output.push(char::from(byte)),
                _ => {
                    let start = self.cursor - 1;
                    let width = utf8_width(byte);
                    let end = start + width;
                    let text = std::str::from_utf8(&self.bytes[start..end])
                        .expect("invalid UTF-8 in JSON string");
                    output.push_str(text);
                    self.cursor = end;
                }
            }
        }
    }

    fn number(&mut self) -> String {
        let start = self.cursor;
        if self.peek() == Some(b'-') {
            self.cursor += 1;
        }
        self.digits();
        if self.take(b'.') {
            self.digits();
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.cursor += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.cursor += 1;
            }
            self.digits();
        }
        std::str::from_utf8(&self.bytes[start..self.cursor])
            .unwrap()
            .to_owned()
    }

    fn digits(&mut self) {
        let start = self.cursor;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.cursor += 1;
        }
        assert!(self.cursor > start, "expected JSON digits");
    }

    fn literal(&mut self, expected: &[u8]) {
        let end = self.cursor + expected.len();
        assert_eq!(&self.bytes[self.cursor..end], expected);
        self.cursor = end;
    }

    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.cursor += 1;
        }
    }

    fn take(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.cursor += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: u8) {
        let actual = self.next();
        assert_eq!(actual, Some(expected), "unexpected JSON byte");
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.cursor).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let value = self.peek()?;
        self.cursor += 1;
        Some(value)
    }
}

fn hex(byte: u8) -> u32 {
    match byte {
        b'0'..=b'9' => u32::from(byte - b'0'),
        b'a'..=b'f' => u32::from(byte - b'a' + 10),
        b'A'..=b'F' => u32::from(byte - b'A' + 10),
        _ => panic!("invalid hexadecimal digit in JSON escape"),
    }
}

fn utf8_width(leading: u8) -> usize {
    match leading {
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => panic!("invalid leading UTF-8 byte in JSON string"),
    }
}

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn checkpoint() -> Json {
    let path = repository_root().join(CHECKPOINT);
    JsonParser::parse(&fs::read_to_string(path).expect("read level-12 checkpoint"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn decode_coefficients(bytes: &[u8], width: usize) -> Vec<i64> {
    match width {
        2 => bytes
            .chunks_exact(2)
            .map(|chunk| i64::from(i16::from_le_bytes([chunk[0], chunk[1]])))
            .collect(),
        4 => bytes
            .chunks_exact(4)
            .map(|chunk| i64::from(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
            .collect(),
        _ => panic!("unsupported kernel coefficient width {width}"),
    }
}

fn verify_exact_integer_residual(label: &str, copy: u64, matrix: &CsrMatrix, kernel: &[i64]) {
    for row in 0..matrix.rows() as usize {
        let start = matrix.row_offsets()[row] as usize;
        let end = matrix.row_offsets()[row + 1] as usize;
        let residual = (start..end).fold(0_i128, |sum, index| {
            sum + i128::from(matrix.coefficients()[index])
                * i128::from(kernel[matrix.column_indices()[index] as usize])
        });
        assert_eq!(
            residual, 0,
            "{label} copy {copy} has nonzero exact residual at row {row}"
        );
    }
}

fn stable_digests(label: &str) -> Option<(&'static str, &'static str)> {
    match label {
        "00010" => Some((
            "041f2dcb423e807b70e0952f71f163113ef26673a00e46f0e959436100c9f6ee",
            "8b6d7e58ea61e4b71c3824820711d8a2f81e69047f01443b85fd371c7b0d7d62",
        )),
        "01002" => Some((
            "6127e200bd993d39c16f9d1a90fb6a972e63c8e92551cd866f80d17e992a583b",
            "3ec6b091f9299b85b6858f6a1be21d6e4434e6e1f32c8ddf6b2f6977b719f4c0",
        )),
        "01100" => Some((
            "1115d770b1e37f36c18aada86ddea087db0acff4103250ea128684cd78b70eea",
            "eeed2f4549f8a328fb05c557bdf838ea98e09e4081efbbac6e0077088e8cc3e1",
        )),
        "02000" => Some((
            "42a510a6b5ee1ca1756ed027c1167f62f0eacb8520f98c4af5a9c0d21cebeeff",
            "12652b189b5a66f5ae0ad18fd18e600174851e357f5d76d314a96f7488cf14fb",
        )),
        "10002" => Some((
            "9ee504fcd076b3368f45646c0276df6d9b9832df80904e746c405c2dc8e1d5b8",
            "174abdb80f54adb3635cf131369cdf0b1c7ab054f1af2f1bb6390af3f06dac75",
        )),
        "11002" => Some((
            "848fac577735265acfff8cd76d24898956ba192a6e1cc7f5fab1e3d7387b951b",
            "492bdee8a851bc9b9bbb553a72ccbf044e0aef961f53a5c82b8cd7c60667bd9e",
        )),
        "11010" => Some((
            "d9c054e95822139ce8d9f9d9e2504565cc07ee086a54b002d9b5bdb60b38a114",
            "81a50ba37241504628f885bf50c4a2e135e4df03c91ff9fd13e8803ae5ce9d2c",
        )),
        "11100" => Some((
            "ab13659933283663560b904ac10c95ca765ccbcb56683717818d708f001a3632",
            "3774221b97bdb3840b133cd1a7788ef7612f0a6fd5c0e404bbca257450d0ff09",
        )),
        "12000" => Some((
            "829b4269bc2d379932eaecb9c00565ae06a643d2e017e040fe4aae21d1b6f6cf",
            "bacc9d2607a0675258685a40f8a1fa3a362fbeeb419e713cbf2c51247137abea",
        )),
        "20002" => Some((
            "6e30c040669d694701805c2fed2d4b1ba8a51d7cc32368ebf27c26f5543560e1",
            "3bc9ff655048200fd5999630de7caf4750f08aa1fa36df06a8a74ed572fa86cf",
        )),
        "20010" => Some((
            "d61a7ba7d315bf6c34476a5e47f54c2c3b522627bbc12bf480609a155a10904f",
            "c5f88f2273a067d74ed891fc916acb475c184d0a61712d8cbfac2e9613277ef9",
        )),
        "20100" => Some((
            "8bf3d7084d5d8ed368f8e342b9742a872d35eddc27e082aac2594945c06b6e8a",
            "1c6fd91724286fda2133c1179ddbbde3b149f4cc5964452051dd827edc85461d",
        )),
        // This coordinate digest was also independently cross-checked against Python.
        "30002" => Some((
            "0d6012f3735d4696f3f7e1fdec2e146a8d1ba08cdd8779d859f41653cdc382a6",
            "4a0c2ce6b463fa6eded11aaabe2833c9f69b3dd3fba37493469609bbf48b01d5",
        )),
        "30010" => Some((
            "edce270df1437c54643ddfdd05f3e6ab027f0949ff7392510390b02068e65bff",
            "ba40c5ada6d7ea49c43752c8d5f0ca13014c1e2c7f96f5c7a32bef73c165f25c",
        )),
        "30100" => Some((
            "8675d60de29b1b3e86b8d333e968d9dd711955b9625ee3a68a2950db995fb9a1",
            "bd84fabde0e6ad114f43fd332c99522c55545bfd8710431c321a5abfce00c169",
        )),
        "31000" => Some((
            "63787bf5b80560e6de1540d65f6c0b2cc70a788b903fcd8857caeccea1d79e89",
            "01d54144024aa4431140ccb3cd3d2371a9a1aca8123efc222e47507d36482bad",
        )),
        "40000" => Some((
            "3d41893ab6b4178cb5d35296e9b49a49ffef71b5467de9bd19b6e95ddcc3c173",
            "3936b65aadb20a773f013c111225e4b4106ae925643c508b80189d21c9228d74",
        )),
        _ => None,
    }
}

#[test]
fn checkpoint_declares_the_completed_corpus_contract() {
    let artifact = checkpoint();
    assert_eq!(artifact.field("schema_version").as_str(), CHECKPOINT_SCHEMA);
    assert!(artifact.field("passed").as_bool());
    assert!(!artifact.field("inventory_complete").as_bool());
    assert_eq!(
        artifact.field("completed_systems").as_u64(),
        COMPLETED_SYSTEMS
    );
    assert_eq!(
        artifact.field("completed_kernel_copies").as_u64(),
        COMPLETED_KERNELS
    );
    assert_eq!(
        artifact.field("systems").as_array().len() as u64,
        COMPLETED_SYSTEMS
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "exhaustive corpus construction is intended for `cargo test --release --test level12_corpus`"
)]
fn completed_manifest_shapes_and_all_published_kernels_match() {
    let root = repository_root();
    let artifact = checkpoint();
    let systems = artifact.field("systems").as_array();
    let mut verified_outputs = 0_u64;

    for system in systems {
        let label = system.field("dynkin_label").as_str();
        assert_eq!(system.field("exterior_degree").as_u64(), 12, "{label}");
        assert_eq!(system.field("prime").as_u64(), u64::from(PRIME), "{label}");
        assert!(system.field("passed").as_bool(), "{label}");

        let matrix = build_level12_matrix(label).unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(
            u64::from(matrix.raising.columns()),
            system.field("source_columns").as_u64(),
            "{label} source columns"
        );
        assert_eq!(
            u64::from(matrix.raising.rows()),
            system.field("raising_rows").as_u64(),
            "{label} raising rows"
        );
        assert_eq!(
            matrix.raising.nonzeros() as u64,
            system.field("nonzero_entries").as_u64(),
            "{label} nonzero entries"
        );
        if let Some((numeric, source_labeled)) = stable_digests(label) {
            assert_eq!(matrix.canonical_sha256(), numeric, "{label} numeric digest");
            assert_eq!(
                matrix.source_labeled_sha256(),
                source_labeled,
                "{label} source-labeled digest"
            );
        }

        let width = system.field("coefficient_width_bytes").as_u64() as usize;
        assert!(matches!(width, 2 | 4), "{label} width {width}");
        let outputs = system.field("outputs").as_array();
        assert_eq!(
            outputs.len() as u64,
            system.field("exact_nullity").as_u64(),
            "{label} output count"
        );

        for (ordinal, output) in outputs.iter().enumerate() {
            let copy = output.field("copy").as_u64();
            assert_eq!(copy, ordinal as u64 + 1, "{label} copy order");
            let relative = Path::new(output.field("path").as_str());
            assert!(
                relative.is_relative(),
                "{label} kernel path must be relative"
            );
            assert!(
                !relative
                    .components()
                    .any(|part| matches!(part, std::path::Component::ParentDir)),
                "{label} kernel path escapes repository"
            );
            let bytes = fs::read(root.join(relative))
                .unwrap_or_else(|error| panic!("{label} copy {copy}: {error}"));
            assert_eq!(
                bytes.len() as u64,
                output.field("bytes").as_u64(),
                "{label} copy {copy} byte count"
            );
            assert_eq!(
                bytes.len(),
                matrix.raising.columns() as usize * width,
                "{label} copy {copy} shape"
            );
            assert_eq!(
                sha256_hex(&bytes),
                output.field("sha256").as_str(),
                "{label} copy {copy} fixture digest"
            );

            let coefficients = decode_coefficients(&bytes, width);
            assert_eq!(
                coefficients.iter().filter(|&&value| value != 0).count() as u64,
                output.field("nonzero_coefficients").as_u64(),
                "{label} copy {copy} nonzero coefficients"
            );
            assert_eq!(
                coefficients
                    .iter()
                    .map(|value| value.unsigned_abs())
                    .max()
                    .unwrap_or(0),
                output.field("maximum_absolute_coefficient").as_u64(),
                "{label} copy {copy} maximum coefficient"
            );
            verify_exact_integer_residual(label, copy, &matrix.raising, &coefficients);

            let modular: Vec<_> = coefficients
                .iter()
                .map(|&value| field_from_i64(value))
                .collect();
            let modular_residual = matrix.raising.spmv(&modular).unwrap();
            assert!(
                modular_residual.iter().all(|&value| value == 0),
                "{label} copy {copy} has a nonzero modular residual"
            );
            verified_outputs += 1;
        }
    }

    assert_eq!(systems.len() as u64, COMPLETED_SYSTEMS);
    assert_eq!(verified_outputs, COMPLETED_KERNELS);
}
