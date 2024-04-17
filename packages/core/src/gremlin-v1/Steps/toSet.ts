import { Traversal } from '../Traversal';
import { toArray } from './toArray';

/**
 * This is a terminal step... in that sense, it's not really a step but a function that runs everything
 */
export const toSet = <S, E>(traversal: Traversal<S, E>): Set<E> => new Set(toArray(traversal));
