use std::env;

fn main() {
    // On macOS, Python extension modules are usually built without linking to libpython.
    // The loader resolves Python symbols at runtime from the hosting interpreter.
    //
    // When `cargo test` (or cross-compilation builds) also builds the `cdylib` artifact,
    // we need to allow unresolved Python symbols at link time.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rustc-link-arg-cdylib=-undefined");
        println!("cargo:rustc-link-arg-cdylib=dynamic_lookup");
    }
}
