import { NumberPredicate } from './types';

export const outside =
  <D extends number>(x: D, y: D): NumberPredicate<D> =>
  z =>
    z < x && y < z;
