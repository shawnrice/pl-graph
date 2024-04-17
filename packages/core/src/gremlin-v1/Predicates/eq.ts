import { ObjectPredicate } from './types';

export const eq =
  <D extends Record<string, unknown>>(x: D): ObjectPredicate<D> =>
  y =>
    Object.is(x, y);
