import { skip } from './skip';

describe('functional iterator tests', () => {
  test('skip works', () => {
    expect(Array.from(skip(2, [1, 2, 3, 4, 5]))).toEqual([3, 4, 5]);
  });
});
