/**
 * The C-ABI version this package is built against. Both backends call
 * `lnk_abi_version()` on load and assert the loaded artifact matches — cheap
 * insurance against a stale `.dylib`/`.wasm` whose symbol shapes have drifted.
 *
 * Bump in lockstep with `lnk_abi_version()` in `crates/lenke-engine/src/ffi.rs`.
 */
export const ABI_VERSION = 19;

export const assertAbi = (loaded: number): void => {
  if (loaded !== ABI_VERSION) {
    throw new Error(
      `lenke native ABI mismatch: artifact reports ${loaded}, package expects ${ABI_VERSION}. ` +
        `Rebuild the Rust crate (bun run build:rust / build:wasm).`,
    );
  }
};
