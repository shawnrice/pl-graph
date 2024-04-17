export type SampleTimer = {
  getTimer: () => () => void;
  stats: () => void;
};

// ts-expect-error: we're not defining the __DEV__ flag on globalThis
const isPerformanceAvailable = !(globalThis as any).__DEV__ &&
  typeof performance !== 'undefined' &&
  typeof performance.now !== 'undefined';

export const sampleTimer = (name: string): SampleTimer => {
  // `performance.now` does not exist in Jest
  if (!isPerformanceAvailable) {
    return {
      getTimer: () => () => { },
      stats: () => { },
    };
  }

  console.group(name);

  const samples: number[] = [];
  const getTimer = () => {
    const start = performance.now();
    const endTimer = () => {
      const end = performance.now();
      const diff = end - start;
      // eslint-disable-next-line functional/immutable-data
      samples.push(diff);

      console.info(`[TIMER] ${name} took ${end - start}ms`);
    };

    return endTimer;
  };

  const stats = () => {
    const n = samples.length;
    const total = samples.reduce((acc, sample) => acc + sample, 0);
    const mean = total / n;
    const squareDistances = samples.map(x => (x - mean) ** 2);
    const totalSquareDistances = squareDistances.reduce((acc, x) => acc + x, 0);
    const stdDeviation = Math.sqrt(totalSquareDistances / n);
    const sorted = [...samples].sort((a, b) => a - b);
    const median = n % 2 === 0 ?
      (sorted[Math.floor(n / 2) - 1] + sorted[Math.floor(n / 2)]) / 2 :
      sorted[Math.floor(n / 2)];

    const min = Math.min(...samples);
    const max = Math.max(...samples);

    console.groupEnd();

    console.info(
      `[TIMER] ${name} ${n} samples. Mean: ${mean}, Median: ${median}, StdDev: ${stdDeviation}, Min: ${min}, Max: ${max}`,
    );
  };

  return { getTimer, stats };
};
