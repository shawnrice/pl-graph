import { useRef, useSyncExternalStore, useDebugValue } from 'react';

import { Traversal } from '@pl-graph/core/src';
import { arraysAreEqual } from '@pl-graph/utils/src';

import { useGraphContext } from './GraphContext';

type Equality<T> = (a: T, b: T) => boolean;

const defaultIsEqual = <T>(a: T, b: T): boolean => {
  if (a instanceof Traversal && b instanceof Traversal) {
    return a.isEqual(b);
  }

  if (Array.isArray(a) && Array.isArray(b)) {
    return arraysAreEqual(a, b);
  }

  return a === b;
};

export const useGraphTraversal = <T>(
  traversal: (graph: Traversal<any, any>) => T,
  isEqual: Equality<T> = defaultIsEqual,
): T => {
  const { graph } = useGraphContext();
  const cache = useRef<T | null>(null);

  const snapshot = useSyncExternalStore(
    (onStoreChange: () => any) => graph.subscribe(onStoreChange),
    () => graph.snapshot(), // store.getState
    undefined,
  );

  const next = snapshot.traverse(traversal);

  if (cache.current === null) {
    // @ts-expect-error type: these are a few TS2345 errors
    cache.current = next;
    // @ts-expect-error type
  } else if (!isEqual(cache.current, next)) {
    // @ts-expect-error type
    cache.current = next;
  }

  useDebugValue(cache.current);

  return cache.current!;
};
