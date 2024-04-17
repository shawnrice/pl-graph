import { isElement } from '@pl-graph/core/src/core/Element';

import { Predicate } from '../../Predicates';
import { Traversal } from '../../Traversal';
import { Traverser } from '../../Traverser';

/* eslint-disable @typescript-eslint/no-magic-numbers */
/* eslint-disable func-style */
/* eslint-disable functional/immutable-data */
/* eslint-disable no-restricted-syntax */
import type { UnaryFn } from '@pl-graph/fp/src';

type TraversalFn<S, E> = UnaryFn<Traversal<S, E>, Traversal<S, E>>;

export function has<S, E = S>(key: string): Traversal<S, E>;
export function has<S, E = S>(key: string, predicate: Predicate<S>): TraversalFn<S, E>;
export function has<S, E = S>(key: string, value: any): TraversalFn<S, E>;
export function has<S, E = S>(label: string, key: string, value: S): TraversalFn<S, E>;
export function has<S, E = S>(
  label: string,
  key: string,
  predicate: Predicate<S>,
): TraversalFn<S, E>;
export function has<S, E = S>(...args: (string | Predicate<S> | any)[]): TraversalFn<S, E> {
  return traversal => {
    const params = { args, genus: 'filter', species: 'has' };

    if (args.length === 3 && typeof args[2] === 'function') {
      const [label, key, predicate] = args as [label: string, key: string, predicate: Predicate<S>];
      const callback = function* callback(prev: Iterable<Traverser<S>>) {
        for (const t of prev) {
          const val = t.get();
          if (isElement(val) && val.hasLabel(label) && predicate(val.properties[key])) {
            yield t;
          }
        }
      };

      return traversal.addStep({ ...params, callback });
    }

    if (args.length === 3) {
      const [label, key, value] = args as [label: string, key: string, value: S];
      const callback = function* callback(prev: Iterable<Traverser<S>>) {
        for (const t of prev) {
          const val = t.get();
          if (isElement(val) && val.hasLabel(label) && val.properties[key] === value) {
            yield t;
          }
        }
      };

      return traversal.addStep({ ...params, callback });
    }

    if (args.length === 2 && typeof args[1] === 'function') {
      const [key, predicate] = args as [key: string, predicate: Predicate<S>];
      const callback = function* callback(prev: Iterable<Traverser<S>>) {
        for (const t of prev) {
          const val = t.get();
          if (isElement(val)) {
            if (key in val.properties && predicate(val.properties[key])) {
              yield t;
            }
          } else if (typeof val === 'object' && val && key in val && predicate(val[key])) {
            yield t;
          }
        }
      };

      return traversal.addStep({ ...params, callback });
    }

    if (args.length === 2) {
      const [key, value] = args as [key: string, value: any];
      const callback = function* callback(prev: Iterable<Traverser<S>>) {
        for (const t of prev) {
          const val = t.get();
          if (isElement(val)) {
            if (key in val.properties && val.properties[key] === value) {
              yield t;
            }
          } else if (typeof val === 'object' && val && key in val && val[key] === value) {
            yield t;
          }
        }
      };

      return traversal.addStep({ ...params, callback });
    }

    if (args.length === 1) {
      const [key] = args as [key: string];
      const callback = function* callback(prev: Iterable<Traverser<S>>) {
        for (const t of prev) {
          const val = t.get();
          if (isElement(val)) {
            if (key in val.properties) {
              yield t;
            }
          } else if (val && key in val) {
            yield t;
          }
        }
      };

      return traversal.addStep({ ...params, callback });
    }

    // Getting here is basically an error

    return traversal;
  };
}
