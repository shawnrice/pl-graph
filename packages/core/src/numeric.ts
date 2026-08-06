/**
 * Numeric primitives whose whole purpose is to mean the SAME thing in every
 * engine.
 *
 * These are not conveniences — each exists because the obvious built-in differs
 * between JavaScript and Rust, so the engines have to agree on one explicit
 * form. Keeping that form in one place is the point: a copy per engine is a copy
 * that can drift, and drift here is a byte-identity bug.
 */

/**
 * ISO GQL / Gremlin `sign` → `-1 | 0 | 1`, with NaN passing through.
 *
 * NOT `Math.sign`, whose signed-zero result (`Math.sign(-0) === -0`) and Rust's
 * `f64::signum` (`+1` for `0.0`) both diverge from this and from each other.
 *
 * Both TS engines had their own identical copy, each with a comment explaining
 * that it had to match the other.
 */
export const mathSign = (x: number): number => {
  if (Number.isNaN(x)) {
    return Number.NaN;
  }

  if (x > 0) {
    return 1;
  }

  return x < 0 ? -1 : 0;
};
