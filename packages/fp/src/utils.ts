type TypedArray = ArrayBufferView & ArrayLike<number>;

const isTypedArray = (x: unknown): x is TypedArray => {
  // @ts-expect-error: this is a type check
  return x && x.buffer instanceof ArrayBuffer && x.BYTES_PER_ELEMENT;
};

const isString = (x: unknown): x is string => {
  return typeof x === 'string' || x instanceof String;
};

export const isIndexed = <T>(x: unknown): x is Iterable<T> & { [key: number]: T; length: number } =>
  Array.isArray(x) || isTypedArray(x) || isString(x);

export const wrapIndexedIterable = <T>(
  x: Iterable<T> & { [key: number]: T; length: number },
): Iterable<T> => ({
  [Symbol.iterator](): Iterator<T> {
    const { length } = x;
    let count = -1; // eslint-disable-line
    return {
      next(): IteratorResult<T> {
        return ++count < length ? { value: x[count]!, done: false } : { value: void 0, done: true };
      },
    };
  },
});

export const maybeOptimizeIterable = <T>(x0: Iterable<T>): Iterable<T> =>
  isIndexed<T>(x0) ? wrapIndexedIterable(x0) : x0;
