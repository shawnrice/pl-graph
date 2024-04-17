import { equals } from './equals';

function* range(start: number, stop: number) {
  let x = start;
  while (x <= stop) {
    yield x++;
  }
}

describe('functional iterator tests', () => {
  test('equals works', () => {
    expect(equals([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])).toBeTruthy();
  });

  test('equals works with a comparator', () => {
    const a = [1, 2, 3, 4, 5];
    const b = ['1', '2', '3', '4', '5'];
    expect(equals<string | number>(a, b)).toBe(false);
    // eslint-disable-next-line eqeqeq
    expect(equals<string | number>(a, b, (x, y) => x == y)).toBe(true);
  });

  test('equals finds different sizes', () => {
    const a = [1, 2, 3, 4, 5];
    const b = [1, 2, 3];
    expect(equals(a, b)).toBe(false);
    expect(equals(b, a)).toBe(false);
    expect(equals(a.slice(0, 3), b)).toBe(true);
  });

  test('using generators works', () => {
    expect(equals(range(0, 10), range(0, 10))).toBe(true);
  });

  test('overflow protection works', () => {
    // We expect to give up before getting there and return false
    expect(equals(range(0, 1_000_500), range(0, 1_000_500))).toBe(false);
  });
});
