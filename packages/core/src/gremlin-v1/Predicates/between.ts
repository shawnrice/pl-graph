import { NumberPredicate } from './types';

export const between =
  <D extends number>(x: D, y: D): NumberPredicate<D> =>
  z =>
    x <= z && z < y;
