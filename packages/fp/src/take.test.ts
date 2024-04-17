import { take } from './take';

describe('functional iterator tests', () => {
  test('take works', () => {
    expect(Array.from(take(2, [1, 2, 3, 4, 5]))).toEqual([1, 2]);
  });
});
