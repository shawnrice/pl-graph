import { NumberPredicate } from './types';

export const inside =
  <D extends number>(x: D, y: D): NumberPredicate<D> =>
  z =>
    x < z && z < y;
