/* eslint-disable functional/no-let */
/* eslint-disable no-restricted-syntax */
/* eslint-disable functional/immutable-data */
/* eslint-disable func-style */

import { isElement } from '@pl-graph/core/src/core/Element';

import { Traversal } from '../../Traversal';
import { Traverser } from '../../Traverser';
import { Step } from '../types';
import { pathElementPropertyProjection } from './pathElementPropertyProjection';

import type { UnaryFn } from '@pl-graph/fp/src';

const findLastRealStep = (traversal: Traversal): Step<any, any> | null => {
  for (let i = traversal.steps.length - 1; i >= 0; i--) {
    const step = traversal.steps[i];
    if (step && step.genus !== 'modulator') {
      return step;
    }
  }

  return null;
};

const getModulatingSteps = (traversal: Traversal): Step<any, any>[] => {
  const modulatingSteps: Step<any, any>[] = [];
  for (let i = traversal.steps.length - 1; i >= 0; i--) {
    const step = traversal.steps[i];
    if (step && step.genus !== 'modulator') {
      break;
    }

    modulatingSteps.push(step);
  }

  return modulatingSteps.reverse();
};

export function by(
  traversal: Traversal<any, any>,
): UnaryFn<Traversal<any, any>, Traversal<any, any>>;
export function by(key: string): UnaryFn<Traversal<any, any>, Traversal<any, any>>;
export function by(
  arg: Traversal<any, any> | string,
): UnaryFn<Traversal<any, any>, Traversal<any, any>> {
  return traversal => {
    const lastStep = findLastRealStep(traversal);
    if (!lastStep) {
      // THIS SHOULD BE A NOOP
      return traversal;
    }

    const params = { args: [arg], genus: 'modulator', species: 'by' };

    if (lastStep.species === 'path' && typeof arg === 'string') {
      // WE MIGHT NOT BE ABLE TO QUEUE THIS ENTIRELY OFF STRINGS
      return traversal.addStep({ ...params, callback: pathElementPropertyProjection(arg) });
    }

    if (lastStep.species === 'select' && typeof arg === 'string') {
      // WE MIGHT NOT BE ABLE TO QUEUE THIS ENTIRELY OFF STRINGS
      const callback = function* (prev: Iterable<Traverser<any>>) {
        for (const t of prev) {
          const element = t.get();
          if (!Object.values(element).every(x => isElement(x) && arg in x.properties)) {
            // eslint-disable-next-line no-continue
            continue;
          }

          const value = Object.fromEntries(
            Object.entries(element).map(([key, val]) => [key, val.properties[arg]]),
          );

          yield t.clone({ value });
        }
      };

      return traversal.addStep({ ...params, callback });
    }

    // handle the string
    if (typeof arg === 'string') {
      const elementPropertyProjection = function* (prev: Iterable<Traverser<any>>) {
        for (const t of prev) {
          const val = t.get();

          // This is one bit of modulation
          if (isElement(val) && arg in val.properties) {
            yield t.clone({ value: val.properties[arg] });
          }
          // Note, if it's not an element or if the key is not in the properties, then
          // we'll essentially filter it out.
        }
      };

      // WE MIGHT NOT BE ABLE TO QUEUE THIS ENTIRELY OFF STRINGS
      return traversal.addStep({ ...params, callback: elementPropertyProjection });
    }

    return traversal;
  };
}
