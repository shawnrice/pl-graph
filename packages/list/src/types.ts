import { UnaryFn } from '../../fp/src';
import { List } from './List';

export type ListFn<T, R = T> = UnaryFn<List<T>, List<R>>;
