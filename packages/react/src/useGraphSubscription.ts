import React from 'react';

import type { NullaryFn } from '@pl-graph/fp/src';

import { useGraphContext } from './GraphContext';
import { useForceUpdate } from './useForceUpdate';

/**
 * Subscribes to the graph
 *
 * This will force a re-render whenever the graph creates a new snapshot
 */
export const useGraphSubscription = (listener?: NullaryFn): void => {
  const { graph } = useGraphContext();
  const update = useForceUpdate();

  React.useEffect(() => graph.subscribe(listener ?? update), [graph, listener, update]);
};
