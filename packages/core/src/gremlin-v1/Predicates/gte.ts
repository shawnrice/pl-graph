import { NumberPredicate } from './types';

export const gte =
  <D extends number>(x: D): NumberPredicate<D> =>
  y =>
    y >= x;
