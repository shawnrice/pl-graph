import {
  after,
  before,
  type BinaryFn,
  equals,
  every,
  filter,
  map,
  type MapFn,
  type NullaryFn,
  type Predicate,
  reject,
  sideEffect,
  skip,
  skipWhile,
  some,
  sort,
  type SortFn,
  take,
  takeWhile,
  type UnaryFn,
  uniq,
} from '@lenke/fp';

import { empty, from, isList, of } from './functions/index.js';

/**
 * A list is list an array or a Set but better for iterators
 *
 * It also has a length property. If the length is unknown,
 * then it will be `Infinity`.
 */
export class List<T> {
  length: number;

  /** Alias of {@link length}, for Set/Map-style `.size` intuition (`Infinity` if unknown). */
  get size(): number {
    return this.length;
  }

  iter: NullaryFn<Generator<T, void>>;

  constructor(generator: NullaryFn<Generator<T, void>>, length = Infinity) {
    this.iter = generator;
    this.length = length;
  }

  static empty = empty;

  static from = from;

  static of = of;

  static isList = isList;

  *[Symbol.iterator](): Generator<T, void> {
    yield* this.iter();
  }

  /**
   * Lazy iteration.
   */
  after(predicate: Predicate<T>): List<T> {
    return List.from(after(predicate, this));
  }

  /**
   * Lazy iteration.
   */
  before(predicate: Predicate<T>): List<T> {
    return List.from(before(predicate, this));
  }

  distinct(): List<T> {
    return List.from(uniq(this));
  }

  /**
   * Lazy iteration.
   */
  equals(list: List<T>, comparator?: BinaryFn<T, T, boolean>): boolean {
    return equals(this, list, comparator);
  }

  every(predicate: Predicate<T>): boolean {
    return every(predicate, this);
  }

  /**
   * Lazy iteration.
   */
  filter(predicate: Predicate<T>): List<T> {
    return List.from(filter(predicate, this));
  }

  /**
   * Lazy iteration.
   */
  map<R = T>(mapper: MapFn<T, R>): List<R> {
    return List.from(map(mapper, this));
  }

  reject(predicate: Predicate<T>): List<T> {
    return List.from(reject(predicate, this));
  }

  sideEffect(effect: UnaryFn<T, any>): List<T> {
    return List.from(sideEffect(effect, this));
  }

  /**
   * Lazy iteration.
   */
  skip(x: number): List<T> {
    return List.from(skip(x, this));
  }

  skipWhile(predicate: Predicate<T>): List<T> {
    return List.from(skipWhile(predicate, this));
  }

  some(predicate: Predicate<T>): boolean {
    return some(predicate, this);
  }

  /**
   * Sort creates an intermediary array
   */
  sort(sorter: SortFn<T>): List<T> {
    return List.from(sort(sorter, this));
  }

  /**
   * Lazy iteration.
   */
  take(x: number): List<T> {
    return List.from(take(x, this));
  }

  head(): T | undefined {
    const [val] = Array.from(this.take(1));

    return val;
  }

  /**
   * Takes the last value
   */
  last(): T | undefined {
    let val = this.head();

    for (const v of this) {
      val = v;
    }

    return val;
  }

  /**
   * All but the first element
   */
  tail(): List<T> {
    return this.skip(1);
  }

  takeWhile(predicate: Predicate<T>): List<T> {
    return List.from(takeWhile(predicate, this));
  }

  /**
   * Turns this into an array
   */
  toArray(): T[] {
    return Array.from(this);
  }

  /**
   * Clones the list
   */
  toList(): List<T> {
    return from(this);
  }

  toString(): string {
    return `List { ${JSON.stringify(Array.from(this.take(3)))} }`;
  }
}
