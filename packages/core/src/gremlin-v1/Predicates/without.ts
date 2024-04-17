import { ObjectPredicate } from './types';

export const without =
  <D extends Record<string, unknown>>(...x: D[]): ObjectPredicate<D> =>
  y =>
    x.length === 1 && Array.isArray(x[0]) ? !x[0].includes(y) : !x.includes(y);
