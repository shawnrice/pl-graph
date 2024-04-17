export type ObjectPredicate<D extends Record<string, unknown>> = (obj: D) => boolean;

export type ArrayOfObjectsPredicate<D extends Record<string, unknown>[]> = (obj: D) => boolean;

export type NumberPredicate<D extends number> = (x: D) => boolean;

export type StringPredicate<D extends string> = (str: D) => boolean;

export type Predicate<D> = D extends Record<string, unknown>
  ? ObjectPredicate<D>
  : D extends Record<string, unknown>[]
  ? ArrayOfObjectsPredicate<D>
  : D extends number
  ? NumberPredicate<D>
  : D extends string
  ? StringPredicate<D>
  : never;
