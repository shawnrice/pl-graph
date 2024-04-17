import { StringPredicate } from './types';

export const startingWith =
  <D extends string>(x0: D): StringPredicate<D> =>
  (x1: string) =>
    x1.startsWith(x0);
