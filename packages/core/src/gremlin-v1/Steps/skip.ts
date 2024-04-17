import type { UnaryFn } from '@pl-graph/fp/src';

import { Traversal } from '../Traversal';
import { range } from './range';

export const skip = (x: number): UnaryFn<Traversal<any, any>> => range(x, -1);
