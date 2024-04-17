export type Predicate<T> = (x: T) => boolean;

export type MapFn<T, R> = (value: T, index: number) => R;

export type SortFn<T> = (a: T, b: T) => number;

export type NullaryFn<R = void> = () => R;

export type UnaryFn<T, R = T> = (x0: T) => R;

export type BinaryFn<T1, T2 = T1, R = T1> = (x0: T1, x1: T2) => R;
