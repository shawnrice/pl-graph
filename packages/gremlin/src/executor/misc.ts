// `math(expr)` — evaluate a tiny infix arithmetic expression. Supports numeric
// literals, parens, `+ - * / % ^`, unary `- +`, the constants `pi`/`e`, a set of
// math functions (`sin`, `cos`, …, `pow`, `log`, `atan2`), the identifier `_`
// (the current traverser value), and other identifiers referencing `as_`-bound
// labels. Operands are projected by the `.by(...)` modulator(s) in
// first-appearance order, cycling — so `math('a + b').by('age')` sums the `age`
// of the values tagged `a` and `b`.

import { mathSign } from '@lenke/core';
import type { Graph } from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';

import type { By, Step } from '../ast.js';
import { byOr, evalBy, extend, recallTag, type RunContext, type Traverser } from './runtime.js';

const IDENT = /[A-Za-z_][A-Za-z0-9_]*/g;

// The distinct *operand* identifiers in `expr`, in first-appearance order. Used
// to map `by()` modulators to operands (TinkerPop cycles them in this order).
// An identifier immediately followed by `(` (whitespace allowed) is a function
// call, not an operand, and is excluded — so it neither consumes a by()
// modulator nor is looked up as an unbound tag. Mirrors the native `math_vars`.
const mathVars = (expr: string): string[] => {
  const seen = new Set<string>();
  const out: string[] = [];

  for (const m of expr.matchAll(IDENT)) {
    let j = (m.index ?? 0) + m[0].length;

    while (j < expr.length && /\s/.test(expr[j])) {
      j++;
    }

    if (expr[j] === '(') {
      continue; // function name — not an operand variable
    }

    if (!seen.has(m[0])) {
      seen.add(m[0]);
      out.push(m[0]);
    }
  }

  return out;
};

// Numeric constants recognized in `math()` (mXparser's `pi`/`e`). Only used when
// the name is not shadowed by a bound variable — the parser resolves first.
const MATH_CONSTS: Record<string, number> = {
  pi: Math.PI,
  e: Math.E,
};

// `math()` `signum`: -1 | 0 | 1 with NaN passing through. Matches the GQL `sign`
// kernel (NOT `Math.sign`, whose signed-zero result diverges) and native.

// Dispatch a `math()` function call. `b` is defined for the 2-arg forms. Every
// op is the SAME underlying primitive the GQL kernel uses (`Math.sin`/`**`/…),
// so `math()` stays bit-identical to GQL and to the native twin. Arity mismatch
// → `undefined` (fault). Note: `log(base, value)` and `ln` (natural) follow GQL
// naming; TinkerPop/mXparser spells natural log `log`.
const UNARY_FN: Record<string, (n: number) => number> = {
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  asin: Math.asin,
  acos: Math.acos,
  atan: Math.atan,
  sinh: Math.sinh,
  cosh: Math.cosh,
  tanh: Math.tanh,
  sqrt: Math.sqrt,
  abs: Math.abs,
  ceil: Math.ceil,
  floor: Math.floor,
  exp: Math.exp,
  ln: Math.log,
  log10: Math.log10,
  signum: mathSign,
};

const BINARY_FN: Record<string, (a: number, b: number) => number> = {
  atan2: (y, x) => Math.atan2(y, x),
  pow: (x, y) => x ** y,
  log: (base, value) => Math.log(value) / Math.log(base),
};

const mathCall = (name: string, a: number, b: number | undefined): number | undefined => {
  if (b !== undefined) {
    return BINARY_FN[name]?.(a, b);
  }

  return UNARY_FN[name]?.(a);
};

// Every malformed-`math()` fault carries ONE code — `InvalidValue` — matching
// the native engine's `set_type_fault` (E_INVALID_VALUE). Byte-identical error
// codes are part of cross-engine parity, so this must NOT diverge (an earlier
// `Unsupported` here broke it on the bare `sin _` form).
const mathFault = (msg: string): LenkeError =>
  new LenkeError(`math: ${msg}`, { code: ErrorCode.InvalidValue });

export const mathStep = function* (
  stream: Iterable<Traverser<unknown>>,
  step: Extract<Step, { kind: 'math' }>,
  graph: Graph,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  const bys = step.bys ?? [];
  const vars = mathVars(step.expr);
  const byFor = (name: string): By =>
    bys.length > 0 ? bys[vars.indexOf(name) % bys.length] : { kind: 'identity' };

  for (const t of stream) {
    // Resolve an operand name to a number, or `undefined` when unbound (the
    // parser may then fall back to a constant, else fault). `_` is the current
    // value; any other name is an as_-bound tag.
    const resolve = (name: string): number | undefined => {
      let base: unknown;

      if (name === '_') {
        base = t.value;
      } else {
        const r = recallTag(t.tags, name, 'last');

        if (!r.ok) {
          return undefined;
        }

        base = r.value;
      }

      // A no-value by() coerces to `undefined` → NaN (unchanged from before NO_VALUE;
      // `Number(NO_VALUE)` would otherwise throw on the symbol).
      return Number(byOr(evalBy(byFor(name), base, graph, ctx)));
    };

    yield extend(t, evalMath(step.expr, resolve));
  }
};

// Recursive-descent parser for the `math()` grammar. Precedence, loosest to
// tightest (mXparser / TinkerPop): `+ -` < `* / %` < `^` (right-assoc) < unary
// `- +` < primary. Primary = numeric literal, parenthesized expr, `name(args)`
// function call, bare/juxtaposition unary application (`sin _`), constant
// (`pi`/`e`), or an identifier resolved via `resolve` (variables win over
// constants and function names). Every fault throws `InvalidValue` — the SAME
// code the native `MathP` raises. A faithful twin of the native `MathP`.
export const evalMath = (expr: string, resolve: (name: string) => number | undefined): number => {
  let pos = 0;
  const peek = (): string => expr[pos] ?? '';
  const skip = () => {
    while (pos < expr.length && /\s/.test(expr[pos])) {
      pos++;
    }
  };
  const parsePrimary = (): number => {
    skip();
    const ch = peek();

    if (ch === '(') {
      pos++;
      const v = parseAdd();
      skip();

      if (peek() !== ')') {
        throw mathFault(`expected ')' in ${expr}`);
      }

      pos++;

      return v;
    }

    // Identifier: variable (`_`/as_-bound), function call, or constant.
    if (/[A-Za-z_]/.test(ch)) {
      const idStart = pos;

      while (pos < expr.length && /[A-Za-z0-9_]/.test(expr[pos])) {
        pos++;
      }

      const name = expr.slice(idStart, pos);

      // Variables win over constants and function names.
      const v = resolve(name);

      if (v !== undefined) {
        return v;
      }

      // Function call: identifier immediately followed by `(`.
      skip();

      if (peek() === '(') {
        pos++; // consume '('
        const a = parseAdd();
        skip();
        let b: number | undefined;

        if (peek() === ',') {
          pos++;
          b = parseAdd();
          skip();
        }

        if (peek() !== ')') {
          throw mathFault(`expected ')' in ${expr}`);
        }

        pos++;
        const r = mathCall(name, a, b);

        if (r === undefined) {
          throw mathFault(`unknown function '${name}' in ${expr}`);
        }

        return r;
      }

      // Bare/juxtaposition form (TinkerPop): a unary function name NOT followed
      // by `(` applies to the next unary expression. Binds tighter than binary
      // ops (`sin _ + 1` == `(sin _) + 1`) and chains right-associatively
      // (`sin cos _` == `sin(cos(_))`); the unary arg also allows a leading sign
      // (`abs -3` == `abs(-3)`). Multi-arg functions still require parens
      // (handled above), so they fall through here to a fault.
      const fn = UNARY_FN[name];

      if (fn) {
        return fn(parseUnary());
      }

      // Unshadowed constant (`pi`/`e`), else an unbound identifier (fault).
      const c = MATH_CONSTS[name];

      if (c !== undefined) {
        return c;
      }

      throw mathFault(`unbound variable '${name}' in '${expr}'`);
    }

    // Number literal (a leading sign is handled by `parseUnary`).
    const start = pos;

    while (pos < expr.length && /[0-9.]/.test(expr[pos])) {
      pos++;
    }

    if (start === pos) {
      throw mathFault(`unexpected '${ch}' in ${expr}`);
    }

    const lit = expr.slice(start, pos);
    const n = Number(lit);

    if (Number.isNaN(n)) {
      throw mathFault(`bad number '${lit}' in ${expr}`);
    }

    return n;
  };
  const parseUnary = (): number => {
    skip();
    const ch = peek();

    if (ch === '-') {
      pos++;

      return -parseUnary();
    }

    if (ch === '+') {
      pos++;

      return parseUnary();
    }

    return parsePrimary();
  };
  const parsePower = (): number => {
    // Unary binds tighter than `^` (mXparser): `-2 ^ 2` == `(-2) ^ 2` == 4.
    const base = parseUnary();
    skip();

    if (peek() === '^') {
      pos++;

      // Right-associative: `2 ^ 3 ^ 2` == `2 ^ (3 ^ 2)` == 512.
      return base ** parsePower();
    }

    return base;
  };
  const parseMul = (): number => {
    let left = parsePower();
    skip();

    while (peek() === '*' || peek() === '/' || peek() === '%') {
      const op = peek();
      pos++;
      const right = parsePower();

      if (op === '*') {
        left *= right;
      } else if (op === '/') {
        left /= right;
      } else {
        left %= right;
      }

      skip();
    }

    return left;
  };
  const parseAdd = (): number => {
    let left = parseMul();
    skip();

    while (peek() === '+' || peek() === '-') {
      const op = peek();
      pos++;
      const right = parseMul();
      left = op === '+' ? left + right : left - right;
      skip();
    }

    return left;
  };
  const result = parseAdd();
  skip();

  if (pos < expr.length) {
    throw mathFault(`trailing input '${expr.slice(pos)}' in ${expr}`);
  }

  return result;
};
