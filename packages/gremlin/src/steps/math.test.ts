import { describe, expect, test } from 'bun:test';

import { ErrorCode, hasErrorCode } from '@lenke/errors';

import { run } from '../executor.js';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import { as_, constant, out, V, hasId, inject, math, project, values } from '../steps.js';
import { traversal } from '../traversal.js';

const arr = (r: Iterable<unknown>): unknown[] => [...r];

describe('math tests', () => {
  const g = createTestTinkerGraph();

  test('math: simple expression on injected number', () => {
    const r = arr(run(traversal(inject(10), math('_ + 5')), g));
    expect(r).toEqual([15]);
  });

  test('math: precedence and parens', () => {
    const r = arr(run(traversal(inject(2), math('(_ + 3) * 4')), g));
    expect(r).toEqual([20]);
  });

  test('math: applied to a property value', () => {
    // marko.age = 29; 29 + 100 = 129
    const r = arr(run(traversal(V(), hasId('1'), values('age'), math('_ + 100')), g));
    expect(r).toEqual([129]);
  });

  // doc: g.V().as('a').out('knows').as('b').math('a + b').by('age')
  test('math: resolves as_-bound names projected by by()', () => {
    // marko(29) —knows→ vadas(27) and josh(32): 29+27=56, 29+32=61.
    const r = arr(
      run(traversal(V(), as_('a'), out('KNOWS'), as_('b'), math('a + b').by('age')), g),
    );
    expect((r as number[]).sort((x, y) => x - y)).toEqual([56, 61]);
  });

  test('math: a single by() cycles across all named operands', () => {
    // a (marko, 29); same `by('age')` applies to a and b.
    const r = arr(
      run(traversal(V(), hasId('1'), as_('a'), out('KNOWS'), as_('b'), math('b - a').by('age')), g),
    );
    // josh 32-29=3, vadas 27-29=-2.
    expect((r as number[]).sort((x, y) => x - y)).toEqual([-2, 3]);
  });

  // doc: g.V().hasId(1).values('age').math('_ / 12') — marko's age in dozens.
  test('math: division on a property value', () => {
    const r = arr(run(traversal(V(), hasId('1'), values('age'), math('_ / 10')), g));
    expect(r).toEqual([2.9]);
  });

  // doc: math with negative result.
  test('math: subtraction yields signed value', () => {
    const r = arr(run(traversal(inject(5), math('_ - 10')), g));
    expect(r).toEqual([-5]);
  });

  // doc: math chained with another math.
  test('math: chained transformations compose', () => {
    const r = arr(run(traversal(inject(2), math('_ * 3'), math('_ + 1')), g));
    expect(r).toEqual([7]);
  });

  // --- functions, `^`/`%`, unary, constants ---

  const one = (expr: string, v: number): number =>
    arr(run(traversal(inject(v), math(expr)), g))[0] as number;

  test('math: functions use the shared f64 kernel', () => {
    expect(one('sin(_)', 0.7)).toBe(Math.sin(0.7));
    expect(one('cos(_)', 0.7)).toBe(Math.cos(0.7));
    expect(one('tan(_)', 0.7)).toBe(Math.tan(0.7));
    expect(one('asin(_)', 0.7)).toBe(Math.asin(0.7));
    expect(one('acos(_)', 0.7)).toBe(Math.acos(0.7));
    expect(one('atan(_)', 0.7)).toBe(Math.atan(0.7));
    expect(one('sinh(_)', 0.7)).toBe(Math.sinh(0.7));
    expect(one('cosh(_)', 0.7)).toBe(Math.cosh(0.7));
    expect(one('tanh(_)', 0.7)).toBe(Math.tanh(0.7));
    expect(one('sqrt(_)', 0.7)).toBe(Math.sqrt(0.7));
    expect(one('abs(-_)', 0.7)).toBe(0.7);
    expect(one('ceil(_)', 0.7)).toBe(1);
    expect(one('floor(_)', 0.7)).toBe(0);
    expect(one('exp(_)', 0.7)).toBe(Math.exp(0.7));
    expect(one('ln(_)', 0.7)).toBe(Math.log(0.7));
    expect(one('log10(_)', 0.7)).toBe(Math.log10(0.7));
    expect(one('signum(_)', 0.7)).toBe(1);
    expect(one('signum(-_)', 0.7)).toBe(-1);
  });

  test('math: two-arg functions', () => {
    expect(one('pow(_, 10)', 2)).toBe(1024);
    expect(one('atan2(_, 1)', 2)).toBe(Math.atan2(2, 1));
    // log(base, value): log base 2 of 8 == 3.
    expect(one('log(_, 8)', 2)).toBe(Math.log(8) / Math.log(2));
  });

  test('math: power operator is right-associative, above `*`, below unary', () => {
    expect(one('2 ^ 3 ^ 2', 0)).toBe(512);
    expect(one('2 * 3 ^ 2', 0)).toBe(18);
    expect(one('-2 ^ 2', 0)).toBe(4);
    expect(one('2 ^ -1', 0)).toBe(0.5);
  });

  test('math: modulo and unary', () => {
    expect(one('_ % 3', 10)).toBe(1);
    expect(one('-_ + 3', 10)).toBe(-7);
    expect(one('- -_', 10)).toBe(10);
    expect(one('2 * 3 % 4', 0)).toBe(2);
  });

  test('math: constants pi and e', () => {
    expect(one('pi', 0)).toBe(Math.PI);
    expect(one('e', 0)).toBe(Math.E);
    expect(one('2 * pi', 0)).toBe(2 * Math.PI);
  });

  test('math: a bound tag shadows a function name', () => {
    // `sin` here is a variable (as_-bound), not the sine function.
    const r = arr(run(traversal(inject(42), as_('sin'), math('sin + 1')), g));
    expect(r).toEqual([43]);
  });

  test('math: unknown function faults', () => {
    expect(() => arr(run(traversal(inject(1), math('nope(_)')), g))).toThrow();
  });

  // --- bare/juxtaposition function form (TinkerPop): `sin _` == `sin(_)` ---

  test('math: bare function application agrees with the paren form', () => {
    expect(one('sin _', 0.7)).toBe(Math.sin(0.7));
    expect(one('sin _', 0.7)).toBe(one('sin(_)', 0.7));
    expect(one('sin _ + 1', 0.7)).toBe(Math.sin(0.7) + 1); // binds tighter than `+`
    expect(one('sin _ * 2', 0.7)).toBe(Math.sin(0.7) * 2); // binds tighter than `*`
    expect(one('-sin _', 0.7)).toBe(-Math.sin(0.7));
    expect(one('abs -3', 0)).toBe(3); // unary arg
    expect(one('sin cos _', 0.7)).toBe(Math.sin(Math.cos(0.7))); // right-assoc chain
    expect(one('sqrt _', 2)).toBe(Math.sqrt(2));
    expect(one('sin (_ + 1)', 0.7)).toBe(Math.sin(0.7 + 1));
  });

  test('math: bare form is unary-only — multi-arg needs parens', () => {
    expect(() => arr(run(traversal(inject(1), math('atan2 _')), g))).toThrow();
  });

  test('math: a bound tag shadows a function name in the bare position', () => {
    // `sin` alone resolves to the variable; `sin _` leaves `_` trailing → fault.
    const r = arr(run(traversal(inject(42), as_('sin'), math('sin')), g));
    expect(r).toEqual([42]);
    expect(() => arr(run(traversal(inject(42), as_('sin'), math('sin _')), g))).toThrow();
  });

  // doc: g.V().project('a','b').by('age').by(constant(1)).math('a + b') — 30
  // math() reads a named variable from the current project()/group() Map row (TinkerPop
  // MathStep scoping), so a project can feed math without an explicit as()-tag.
  test('math: named variables resolve from a project() row', () => {
    // marko.age = 29; a=29, b=29 → 58.
    expect(
      arr(run(traversal(V(), hasId('1'), project('a', 'b').by('age').by('age'), math('a + b')), g)),
    ).toEqual([58]);
    // The canonical TinkerPop form with a constant second key: 29 + 1 = 30.
    expect(
      arr(
        run(
          traversal(V(), hasId('1'), project('a', 'b').by('age').by(constant(1)), math('a + b')),
          g,
        ),
      ),
    ).toEqual([30]);
  });

  test('math: an unbound name on a non-map frontier still faults', () => {
    // No project/group scope and no as() tag → `a` is unbound (a value error).
    expect(() => arr(run(traversal(V(), hasId('1'), math('a + b')), g))).toThrow();
  });

  // Every malformed-math fault carries the same code as native: E_INVALID_VALUE.
  test('math: faults carry ErrorCode.InvalidValue', () => {
    for (const expr of ['_ +', 'nope(_)', 'atan2 _', '( _', 'sin _ )']) {
      let code: unknown;

      try {
        arr(run(traversal(inject(1), math(expr)), g));
      } catch (e) {
        code = hasErrorCode(e, ErrorCode.InvalidValue) ? ErrorCode.InvalidValue : e;
      }

      expect(code).toBe(ErrorCode.InvalidValue);
    }
  });
});
