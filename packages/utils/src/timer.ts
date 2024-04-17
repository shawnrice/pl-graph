/* eslint-disable no-console */

// ts-expect-error: we're not defining the __DEV__ flag on globalThis
const isPerformanceAvailable =
  // Do not expose timings on production
  !(globalThis as any).__DEV__ &&
  // `performance.now` does not exist in Jest
  typeof performance !== 'undefined' &&
  typeof performance.now !== 'undefined';

export const timer = (name: string): (() => void) => {
  if (!isPerformanceAvailable) {
    return () => {};
  }

  const start = performance.now();

  return () => {
    const end = performance.now();

    console.info(`[TIMER] ${name} took ${end - start}ms`);
  };
};
