/**
 * @jest-environment jsdom
 */
import * as React from 'react';

import { act, renderHook, waitFor } from '@testing-library/react';

import { createTestTinkerGraph } from './fixtures/createTestTinkerGraph';
import { GraphProvider } from './GraphProvider';
import { useGraphTraversal } from './useGraphTraversal';

const createTinkerWrapper = () => {
  const tinkerGraph = createTestTinkerGraph();

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <GraphProvider graph={tinkerGraph}>{children}</GraphProvider>
  );

  return { tinkerGraph, wrapper };
};

describe('useGraphTraversal Hooks', () => {
  test('we can query for marko', () => {
    const tinkerGraph = createTestTinkerGraph();

    const wrapper = ({ children }) => <GraphProvider graph={tinkerGraph}>{children}</GraphProvider>;

    const { result } = renderHook(() => useGraphTraversal(x => x.V('1').values('name').toArray()), {
      wrapper,
    });

    expect(result.current).toEqual(['marko']);
  });

  test('mutating the graph works', () => {
    vi.useFakeTimers();
    const { tinkerGraph, wrapper } = createTinkerWrapper();

    const { result } = renderHook(
      () => useGraphTraversal(x => x.V('15').values('name').toArray()),
      {
        wrapper,
      },
    );

    expect(result.current).toEqual([]);

    act(() => {
      tinkerGraph.addVertex({
        id: '15',
        labels: ['PERSON'],
        properties: {
          name: 'Shawn',
          age: 39,
        },
      });
    });

    vi.runOnlyPendingTimers();

    expect(result.current).toEqual(['Shawn']);
  });

  test.skip('mutating the graph works with a legacy root', () => {
    vi.useFakeTimers();
    const { tinkerGraph, wrapper } = createTinkerWrapper();

    const { result } = renderHook(
      () => useGraphTraversal(x => x.V('15').values('name').toArray()),
      {
        wrapper,
        legacyRoot: true, // this causes react to show errors in the console
      },
    );

    expect(result.current).toEqual([]);

    act(() => {
      tinkerGraph.addVertex({
        id: '15',
        labels: ['PERSON'],
        properties: {
          name: 'Shawn',
          age: 39,
        },
      });
      vi.runOnlyPendingTimers();
    });

    expect(result.current).toEqual(['Shawn']);
  });

  test('we can unsubscribe when the component unmounts', () => {
    const { tinkerGraph, wrapper } = createTinkerWrapper();

    const { result, unmount } = renderHook(
      () => useGraphTraversal(x => x.V('1').values('name').toArray()),
      {
        wrapper,
      },
    );

    expect(result.current).toEqual(['marko']);
    expect(tinkerGraph.listeners.size).toBe(1);

    unmount();

    expect(tinkerGraph.listeners.size).toBe(0);
  });

  test('we can prevent default', () => {
    const { tinkerGraph, wrapper } = createTinkerWrapper();

    const preventNewVertices = vi.fn(event => {
      if (event.type === '@graph/VertexAdded') {
        return event.preventDefault();
      }
    });

    tinkerGraph.on('@graph/VertexAdded', preventNewVertices);

    const { result } = renderHook(
      () => useGraphTraversal(x => x.V('15').values('name').toArray()),
      {
        wrapper,
      },
    );

    expect(result.current).toEqual([]);

    act(() => {
      tinkerGraph.addVertex({
        id: '15',
        labels: ['PERSON'],
        properties: {
          name: 'Shawn',
          age: 39,
        },
      });
      vi.runOnlyPendingTimers();
    });

    expect(result.current).toEqual([]);
    expect(preventNewVertices).toHaveBeenCalledTimes(1);
  });
});
