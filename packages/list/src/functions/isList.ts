import { List } from '../List';

export const isList = <T>(x: unknown): x is List<T> => x instanceof List;
