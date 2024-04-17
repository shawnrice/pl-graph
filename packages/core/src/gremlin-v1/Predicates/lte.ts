import { NumberPredicate } from './types';

export const lte =
  <D extends number>(x: D): NumberPredicate<D> =>
  y =>
    y <= x;
