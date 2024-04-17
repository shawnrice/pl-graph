/* eslint-disable func-names */

import { List } from '../List';

export function of<T>(...args: T[] | readonly T[]): List<T> {
  return new List<T>(function* () {
    yield* args;
  }, args.length);
}
