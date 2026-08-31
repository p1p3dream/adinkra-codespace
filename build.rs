use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/second_momentum_fx_cuda.cu");
    println!("cargo:rerun-if-changed=cuda/complete_f_sparse_cuda.cu");
    println!("cargo:rerun-if-changed=cuda/four_form_56_stream_cuda.cu");
    println!("cargo:rerun-if-changed=cuda/d21_invariant_diagrams_cuda.cu");
    println!("cargo:rerun-if-changed=cuda/d21_witness_constructor_cuda.cu");
    println!("cargo:rerun-if-changed=cuda/teleparallel_lorentz_descent_cuda.cu");
    println!("cargo:rerun-if-changed=cuda/common_parent_ph_obstruction_cuda.cu");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=ADYNKRA_CUDA_ARCH");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() || env::var_os("DOCS_RS").is_some() {
        return;
    }
    if env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() != "linux" {
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
    let second_momentum_object = out_dir.join("second_momentum_fx_cuda.o");
    let complete_f_object = out_dir.join("complete_f_sparse_cuda.o");
    let four_form_56_object = out_dir.join("four_form_56_stream_cuda.o");
    let d21_diagrams_object = out_dir.join("d21_invariant_diagrams_cuda.o");
    let d21_witness_object = out_dir.join("d21_witness_constructor_cuda.o");
    let lorentz_descent_object = out_dir.join("teleparallel_lorentz_descent_cuda.o");
    let common_parent_ph_object = out_dir.join("common_parent_ph_obstruction_cuda.o");
    let archive = out_dir.join("libadynkra_second_momentum_fx_cuda.a");
    let architecture = env::var("ADYNKRA_CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_owned());
    run(
        Command::new(&nvcc)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-lineinfo")
            .arg("-Xptxas=-warn-spills")
            .arg("--threads=0")
            .arg(format!("-arch={architecture}"))
            .arg("-Xcompiler=-fPIC")
            .arg("-c")
            .arg("cuda/second_momentum_fx_cuda.cu")
            .arg("-o")
            .arg(&second_momentum_object),
        "compile second-momentum CUDA backend",
    );
    run(
        Command::new(&nvcc)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-lineinfo")
            .arg("-Xptxas=-warn-spills")
            .arg("--threads=0")
            .arg(format!("-arch={architecture}"))
            .arg("-Xcompiler=-fPIC")
            .arg("-c")
            .arg("cuda/complete_f_sparse_cuda.cu")
            .arg("-o")
            .arg(&complete_f_object),
        "compile complete-F sparse CUDA backend",
    );
    run(
        Command::new(&nvcc)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-lineinfo")
            .arg("-Xptxas=-warn-spills")
            .arg("--threads=0")
            .arg(format!("-arch={architecture}"))
            .arg("-Xcompiler=-fPIC")
            .arg("-c")
            .arg("cuda/four_form_56_stream_cuda.cu")
            .arg("-o")
            .arg(&four_form_56_object),
        "compile four-form 56-column CUDA backend",
    );
    run(
        Command::new(&nvcc)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-lineinfo")
            .arg("-Xptxas=-warn-spills")
            .arg("--threads=0")
            .arg(format!("-arch={architecture}"))
            .arg("-Xcompiler=-fPIC")
            .arg("-c")
            .arg("cuda/d21_invariant_diagrams_cuda.cu")
            .arg("-o")
            .arg(&d21_diagrams_object),
        "compile D21 invariant-diagram CUDA backend",
    );
    run(
        Command::new(&nvcc)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-lineinfo")
            .arg("-Xptxas=-warn-spills")
            .arg("--threads=0")
            .arg(format!("-arch={architecture}"))
            .arg("-Xcompiler=-fPIC")
            .arg("-c")
            .arg("cuda/d21_witness_constructor_cuda.cu")
            .arg("-o")
            .arg(&d21_witness_object),
        "compile D21 witness-constructor CUDA backend",
    );
    run(
        Command::new(&nvcc)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-lineinfo")
            .arg("-Xptxas=-warn-spills")
            .arg("--threads=0")
            .arg(format!("-arch={architecture}"))
            .arg("-Xcompiler=-fPIC")
            .arg("-c")
            .arg("cuda/teleparallel_lorentz_descent_cuda.cu")
            .arg("-o")
            .arg(&lorentz_descent_object),
        "compile teleparallel Lorentz-descent CUDA backend",
    );
    run(
        Command::new(&nvcc)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-lineinfo")
            .arg("-Xptxas=-warn-spills")
            .arg("--threads=0")
            .arg(format!("-arch={architecture}"))
            .arg("-Xcompiler=-fPIC")
            .arg("-c")
            .arg("cuda/common_parent_ph_obstruction_cuda.cu")
            .arg("-o")
            .arg(&common_parent_ph_object),
        "compile common-parent P_H obstruction CUDA backend",
    );
    run(
        Command::new(&nvcc)
            .arg("--lib")
            .arg(&second_momentum_object)
            .arg(&complete_f_object)
            .arg(&four_form_56_object)
            .arg(&d21_diagrams_object)
            .arg(&d21_witness_object)
            .arg(&lorentz_descent_object)
            .arg(&common_parent_ph_object)
            .arg("-o")
            .arg(&archive),
        "archive second-momentum CUDA backend",
    );
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=adynkra_second_momentum_fx_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_library_dir(&cuda_home).display()
    );
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
