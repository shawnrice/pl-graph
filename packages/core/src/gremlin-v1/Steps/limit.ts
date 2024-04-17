import type { UnaryFn } from '@pl-graph/fp/src';

import { Traversal } from '../Traversal';
import { range } from './range';

export const limit = (x: number): UnaryFn<Traversal<any, any>> => range(0, x);
