export function pick<TKeys extends keyof TSource, TSource extends object>(
  keys: TKeys[],
  object: TSource,
): Pick<TSource, TKeys> {
  const result = {} as Pick<TSource, TKeys>;
  for (const key of keys) {
    result[key] = object[key];
  }
  return result;
}
