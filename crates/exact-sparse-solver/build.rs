use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/exact_sparse_cuda.cu");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=ADYNKRA_CUDA_ARCH");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() || env::var_os("DOCS_RS").is_some() {
        return;
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "linux" {
        panic!("the `cuda` feature currently supports Linux targets only");
    }

    let cuda_home = env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let nvcc = cuda_home.join("bin/nvcc");
    if !nvcc.is_file() {
        panic!(
            "CUDA compiler not found at {}; set CUDA_HOME to a CUDA toolkit",
            nvcc.display()
        );
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    let object = out_dir.join("exact_sparse_cuda.o");
    let archive = out_dir.join("libadynkra_exact_sparse_cuda.a");
    let architecture = env::var("ADYNKRA_CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_owned());

    run(
        Command::new(&nvcc)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("--threads=0")
            .arg(format!("-arch={architecture}"))
            .arg("-Xcompiler=-fPIC")
            .arg("-c")
            .arg("cuda/exact_sparse_cuda.cu")
            .arg("-o")
            .arg(&object),
        "compile CUDA backend",
    );
    run(
        Command::new(&nvcc)
            .arg("--lib")
            .arg(&object)
            .arg("-o")
            .arg(&archive),
        "archive CUDA backend",
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=adynkra_exact_sparse_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    let cuda_lib = cuda_library_dir(&cuda_home);
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
}

fn cuda_library_dir(cuda_home: &Path) -> PathBuf {
    let lib64 = cuda_home.join("lib64");
    if lib64.is_dir() {
        return lib64;
    }
    let target = cuda_home.join("targets/x86_64-linux/lib");
    if target.is_dir() {
        return target;
    }
    panic!(
        "CUDA runtime library directory not found under {}",
        cuda_home.display()
    );
}

fn run(command: &mut Command, action: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to {action}: {error}"));
    assert!(status.success(), "failed to {action}: {status}");
}
