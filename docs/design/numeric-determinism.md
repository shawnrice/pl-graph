# Numeric determinism across builds

Status: **open decision, deliberately not taken.** This records what was measured so the choice can be made on numbers rather than re-derived. User-facing behavior is documented in [choosing-your-build](../guides/choosing-your-build.md#numeric-results-across-builds); this note is the cost analysis behind it.

## The situation

Exact arithmetic agrees everywhere. The transcendentals (`exp`, `ln`, `log`, `log10`, `power`, `sin`, `cos`, `tan`, `cot`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`) do not, because IEEE 754 does not require correct rounding for them and each build reaches a different implementation: the JS runtime's `Math.*` for pure-TS, the **host OS's** libm for native, and Rust's compiled-in math for wasm (`lenke_core.wasm` imports nothing from the host — verified by reading its import section).

The consequence that actually matters is not accuracy. A 1-ulp difference is far below any tolerance anyone would set, but it still breaks `=`, `DISTINCT` and `GROUP BY` on a computed double — two engines put the same logical value in different buckets. That is why this is a correctness question for a graph database and not merely a numerics nicety.

## Measured magnitude

Worst single-call disagreement between pure-TS and native is **4 ulp — 15.3 significant digits**:

```
power(0.1, 10)   ts=1.0000000000000011e-10   native=1.0000000000000006e-10
```

It does not meaningfully compound. Over 2000 rows, 118 individual `exp` values differ, yet the aggregate still agrees to ~14 significant digits (summation rounding absorbs most of it — though not always: a naive sum of 2000 `sin` values does differ in the last few digits, so it can survive into an aggregate).

A ~7–12% disagreement rate across arguments is normal for two independent implementations. A good libm targets under 1 ulp of error and glibc's is nearer half that, so the two differ exactly on the arguments where they fall on opposite sides of a rounding boundary.

## The options, with costs

Measured on Linux/glibc. `libm` here means routing the Rust engine through the `libm` crate on **all** targets, not just wasm.

|                                 | native ↔ wasm, any OS    | pure-TS ↔ native | binary | speed                   |
| ------------------------------- | ------------------------ | ---------------- | ------ | ----------------------- |
| **today** — host libm on native | 93%, varies by OS        | **98.7%**        | —      | —                       |
| **swap** — `libm` crate on both | **100%, OS-independent** | 88%              | +0.6%  | −38% on transcendentals |
| **fdlibm in both engines**      | **100%**                 | **100%**         | +0.6%  | −38% on transcendentals |

Binary cost of the swap, measured by building it: `liblenke_core.so` 3,168,120 → 3,187,000 B (**+18,880**, +0.60%); `lenke_core.wasm` 2,282,023 → 2,296,181 B (**+14,158**, +0.62%). The wasm growth is avoidable in principle — calling `libm::` explicitly pulls in the crate's copy alongside the one std already used — but is not worth chasing.

Speed cost, on a deliberately math-saturated query (20k rows × 4 transcendentals): 14.81 → 20.44 ms/query, **+38%**. glibc's routines are hand-optimized and vectorized; the pure-Rust ones are not. Ordinary graph queries do essentially no transcendental math, so the end-to-end effect there is ~0; numeric analytics pays the full 38%.

The swap was verified to do what it claims: with it applied, native and wasm agreed on **272/272** (function, argument) pairs.

## Why nothing was changed

The swap does not eliminate divergence — it **moves** it, buying agreement between the two Rust builds at the price of agreement between the two engines (98.7% → 88%). That is not obviously a good trade, and it is not reversible for free once results are persisted or compared downstream.

The only option that removes the seam entirely is the third row, which is the industry answer: name one reference implementation and use it everywhere. Java does exactly this — `java.lang.Math` is fast and platform-dependent while `StrictMath` is specified to reproduce the published [fdlibm](https://www.netlib.org/fdlibm/readme) algorithms bit-for-bit, over very nearly this same function list. JS engines converged the same way: ECMA-262 recommends fdlibm for `sin`/`cos`/`tan`, and V8 and SpiderMonkey adopted it partly for reproducibility. For lenke that means porting ~15 functions into the TS engine so it matches the Rust one — bounded work, but real work.

**Take the swap only as part of the third row.** On its own it trades one seam for a wider one; as the Rust half of a shared-implementation plan it is the right first step.

## Reproducing the measurements

The call sites are one dispatch block in `crates/lenke-core/src/gql/eval/scalar_fns.rs` plus three in `src/gremlin/exec.rs`. Cross-build agreement is guarded continuously by `packages/native/src/backend-parity-fuzz.test.ts`, which excludes these functions and says why.
