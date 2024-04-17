export const isGeneratorFunction = <T>(x: unknown): x is () => Generator<T> => {
  if (typeof x !== 'function') {
    return false;
  }

  if (process?.versions?.bun) {
    // For Bun (at least for 1.30.0), the constructor name is not 'GeneratorFunction'
    const regex = /^(async\s+)?function *\*/;

    return !!(
      typeof x === 'function' &&
      String.prototype.match.call(Function.prototype.toString.call(x), regex)
    );
  }

  return x.constructor.name === 'GeneratorFunction';
};
