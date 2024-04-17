import { NumberPredicate } from './types';

export const gt =
  <D extends number>(x: D): NumberPredicate<D> =>
  y =>
    y > x;
