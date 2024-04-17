import { after } from './after';

describe('functional iterator tests', () => {
  test('after works', () => {
    const isFive = (x: number) => x === 5;
    expect(Array.from(after(isFive, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))).toEqual([6, 7, 8, 9, 10]);
  });
});
