/* eslint-disable consistent-this, no-param-reassign, no-restricted-syntax */
/* eslint-disable @typescript-eslint/no-this-alias */

import { equals } from '../../fp/src';
import { identity } from '../../identity';
import { List } from '../../list/src';
import { rando } from '../../rando';
import { deserialize } from './deserialize';
import { serialize } from './serialize';

type UnaryFn<T extends any = any, R extends any = T> = (x0: T) => R;
type BinaryFn<A extends any = any, B = A, R = A> = (x0: A, x1: B) => R;

export type TreeNodeJSON = {
  id: string;
  value: any;
  parentId: string | null;
  children: TreeNodeJSON[];
};

export type SerializedTreeNode<T> = {
  id: string;
  value: T; // well a stringified version of this
  parentId: string | null;
};

/**
 * A simple tree class with a few bells
 *
 * A TreeNode has a value. All relevant information is stored as private properties and is
 * accessed via getters. The getters return copies of everything except the TreeNodes themselves
 * in order to discourage direct mutation.
 *
 * Controlling the tree is managed via the class methods.
 *
 * This tree supports a couple of expected operations:
 *
 *   - Add a tree node from a value
 *   - Add a pre-existing tree node to be a child of another tree node
 *   - Easy root tracking
 *   - Casting depth-first
 *   - Casting breadth-first
 *   - Remove a TreeNode, making its children
 *
 *
 * Do we add map / filter / reduce ?
 */

export class TreeNode<T> {
  #id: string;

  #parent: TreeNode<T> | null;

  #value: T;

  #children: Set<TreeNode<T>>;

  #childrenById: Map<string, TreeNode<T>>;

  /**
   * Deserializes a serialized TreeNode<T>
   */
  static deserialize<T>(
    serialized: SerializedTreeNode<T>[],
    deserializeValue: (x: any) => T = identity,
  ): TreeNode<T> {
    return deserialize(serialized, deserializeValue);
  }

  /**
   * Serializes the Tree from the `TreeNode` down using the serializeValue function
   */
  static serialize<T>(node: TreeNode<T>, serializeValue = identity): SerializedTreeNode<T>[] {
    return serialize(node, serializeValue);
  }

  /**
   * Checks if two Trees are equal in value and id
   *
   * Note: this uses `Object.is` equality by default to check values
   */
  static equals<T>(
    a: TreeNode<T>,
    b: TreeNode<T>,
    comparator: BinaryFn<T, T, boolean> = Object.is,
  ): boolean {
    const inner = (x: TreeNode<T>, y: TreeNode<T>): boolean =>
      x.id === y.id && comparator(x.value, y.value);

    return equals(a, b, inner);
  }

  /**
   * Creates a TreeNode with no parent from a value and maybe an ID
   */
  static from<T>(value: T, id?: string | null): TreeNode<T> {
    return new TreeNode<T>(null, value, id);
  }

  /**
   * Removes a TreeNode from a Tree
   */
  static removeTreeNode<T>(node: TreeNode<T>): TreeNode<T> | null {
    if (node.isRoot() && node.childCount === 0) {
      // If we have a Tree made of a single TreeNode and we remove it, then we return null
      return null;
    }

    if (node.isRoot() && node.childCount === 1) {
      // So, now we are removing the root node, and there is one child
      const [child] = node.#children;

      if (!child) {
        return null;
      }

      child.removeParent();
      node.delistChild(child);

      // And the child is the new root, so return that
      return child;
    }

    if (node.isRoot() && node.childCount > 1) {
      // Otherwise, it we still try to remove the root which would result in multiple trees, then
      // we'll throw an error
      throw new Error(
        'Cannot remove the root node from a tree when the root has multiple children',
      );
    }

    const nextParent = node.parent;
    if (!nextParent) {
      // We should actually not get here with any well-formed tree
      throw new Error('Cannot remove a node that has no parent');
    }

    // Here, we have a more normal use case of removing a leaf node or a branch
    node.setParent(null);

    node.children.forEach(child => {
      // This should re-home the child appropriately
      nextParent.addChild(child);
    });

    node.#parent = null;

    return nextParent.root;
  }

  constructor(parent: TreeNode<T> | null, value: T, id: string | null = null) {
    this.#value = value;
    this.#parent = parent ?? null;
    this.#id = id ?? rando();
    this.#children = new Set();
    this.#childrenById = new Map();
  }

  get id(): string {
    return this.#id;
  }

  get parent(): TreeNode<T> | null {
    return this.#parent ?? null;
  }

  get root(): TreeNode<T> {
    return this.#parent?.root ?? this;
  }

  get children(): TreeNode<T>[] {
    return Array.from(this.#children);
  }

  get childCount(): number {
    return this.#children.size;
  }

  get ancestors(): TreeNode<T>[] {
    return this.getAncestors();
  }

  get depth(): number {
    return this.ancestors.length;
  }

  get value(): T {
    return this.#value;
  }

  /**
   * Adds an Extant TreeNode<T> as a child of this TreeNode<T>
   */
  addChild(node: TreeNode<T>): TreeNode<T> {
    // Always test from the top of the tree to ensure that this tree doesn't cycle
    if (node.castDepthFirst().some(n => this.root.contains(n) || this.root === n)) {
      throw new Error('Operation would create a cycle');
    }

    this.#children.add(node);
    this.#childrenById.set(node.id, node);
    node.setParent(this);

    return this;
  }

  /**
   * Removes a child. Its children will become this node's children
   *
   * @returns The current tree node
   */
  removeChild(node: TreeNode<T>): TreeNode<T> {
    this.delistChild(node);
    node.removeParent();

    return this;
  }

  /**
   * Adds a child
   *
   * @returns The new child TreeNode
   */
  createChild(value: T, id: string | null = null): TreeNode<T> {
    const node = new TreeNode(this, value, id);
    this.#children.add(node);
    this.#childrenById.set(node.id, node);
    return node;
  }

  /**
   * Checks if a TreeNode has the passed TreeNode as a child
   */
  hasChild(node: TreeNode<T>): boolean {
    return this.#childrenById.has(node.id);
  }

  /**
   * Detaches this node and its children to be a SubTree
   */
  detach(): TreeNode<T> {
    if (this.#parent) {
      this.#parent?.removeChild(this);
      this.#parent = null;
    }

    return this;
  }

  /**
   * Removes this TreeNode from the Tree
   *
   * Returns the root of the Tree
   */
  remove(): TreeNode<T> | null {
    return TreeNode.removeTreeNode(this);
  }

  /**
   * Checks if a `TreeNode<T>` is a root
   */
  isRoot(): boolean {
    return this.root === this;
  }

  /**
   * Checks if a TreeNode contains another TreeNode as a descendant
   */
  contains(needle: TreeNode<T>): boolean {
    const queue: TreeNode<T>[] = Array.from(this.#children);

    while (queue.length) {
      const node = queue.shift();

      if (node === needle) {
        return true;
      }

      node?.children.forEach(child => {
        queue.push(child);
      });
    }

    return false;
  }

  /**
   * Depth first iterator for TreeNodes
   */
  *depthFirst(): Generator<TreeNode<T>> {
    yield this;

    for (const child of this.#children) {
      yield* child.depthFirst();
    }
  }

  /**
   * Filter Breadth First, returns a generator of TreeNodes that match the predicate
   */
  *filterBreadthFirst(predicate: UnaryFn<TreeNode<T>, boolean>): Generator<TreeNode<T>> {
    const queue = [this];

    while (queue.length) {
      const node = queue.shift();

      if (!node) {
        continue; // eslint-disable-line
      }

      if (predicate(node)) {
        yield node;
      }

      for (const child of node.children) {
        // @ts-expect-error: the child might be instantiated with a different type
        queue.push(child);
      }
    }
  }

  /**
   * The iterator for a TreeNode is depth-first
   */
  *[Symbol.iterator](): Generator<TreeNode<T>, void> {
    yield* this.depthFirst();
  }

  /**
   * Casts the tree depth-first into an array of TreeNode<T>
   */
  castDepthFirst(): TreeNode<T>[] {
    return Array.from(this);
  }

  /**
   * Filter Depth First, returns a generator of TreeNodes that match the predicate
   */
  *filterDepthFirst(predicate: UnaryFn<TreeNode<T>, boolean>): Generator<TreeNode<T>> {
    if (predicate(this)) {
      yield this;
    }

    for (const child of this.#children) {
      yield* child.filterDepthFirst(predicate);
    }
  }

  /**
   * Casts the tree depth-first into an array of T, stripping away
   * the TreeNode parts
   */
  castDepthFirstValue(): T[] {
    return this.castDepthFirst().map(x => x.#value);
  }

  forEach(callback: UnaryFn<TreeNode<T>, any>): void {
    for (const node of this) {
      callback(node);
    }
  }

  /**
   * A breadth-first iterator for tree nodes
   */
  *breadthFirst(): Generator<TreeNode<T>> {
    const queue = [this];

    while (queue.length) {
      const node = queue.shift();

      if (!node) {
        continue; // eslint-disable-line
      }

      for (const child of node.children) {
        // @ts-expect-error: the child might be instantiated with a different type
        queue.push(child);
      }

      yield node;
    }
  }

  /**
   * Casts the tree breadth-first into an array of TreeNode<T>
   */
  castBreadthFirst(): TreeNode<T>[] {
    return Array.from(this.breadthFirst());
  }

  /**
   * Casts the tree breadth-first into an array of T, stripping away
   * the TreeNode parts
   */
  castBreadthFirstValue(): T[] {
    return this.castBreadthFirst().map(x => x.#value);
  }

  /**
   * Searches depth-first for a TreeNode via an ID
   *
   * If no node is found, we return `null`
   */
  getDescendantById(id: string): TreeNode<T> | null {
    if (this.#childrenById.has(id)) {
      return this.#childrenById.get(id) as TreeNode<T>;
    }

    for (const child of this.#children) {
      const node = child.getDescendantById(id);

      if (node) {
        return node;
      }
    }

    return null;
  }

  /**
   * Sets the value of the TreeNode to another T
   *
   * Returns the `TreeNode`
   */
  setValue(callback: UnaryFn<T>): TreeNode<T> {
    this.#value = callback(this.#value);

    return this;
  }

  private delistChild(child: TreeNode<T>): void {
    this.#children.has(child) && this.#children.delete(child);
    this.#childrenById.has(child?.id) && this.#childrenById.delete(child?.id);
  }

  private removeParent() {
    if (this.#parent) {
      this.#parent = null;
    }
  }

  /**
   * Clones a SubTree starting at this node
   */
  clone(): TreeNode<T> {
    // Some algo to create a new TreeNode for each tree node from here down.
    const nodeMap = new Map<string, TreeNode<T>>();

    for (const original of this.breadthFirst()) {
      // if we have a parent, and if we have already cloned the parent, then
      // we'll grab the cloned parent, and create this node as its child
      if (original.parent && nodeMap.has(original.parent.id)) {
        const parent = nodeMap.get(original.parent.id)!;
        const next = parent.createChild(original.value, original.id);
        nodeMap.set(next.id, next);
      } else {
        // We don't have a parent that has been cloned, which likely means that
        // we're at the root, so we'll just create a new node with no parent.
        const next = new TreeNode(null, original.value, original.id);
        nodeMap.set(next.id, next);
      }
    }

    return nodeMap.get(this.id)!;
  }

  /**
   * Clones the entire tree, starting at the root, regardless of the node
   * that you call this from
   */
  cloneEntireTree(): TreeNode<T> {
    return this.root.clone();
  }

  /**
   * Internal method for setting parents
   *
   * If you want to alter trees, use detach, addChild, createChild, or removeNode
   */
  private setParent(node: TreeNode<T> | null) {
    if (this.#parent === node) {
      // no-op
      return this;
    }

    this.#parent?.removeChild(this);
    this.#parent = node;
    return this;
  }

  /**
   * Gets all the ancestors of a node in order
   */
  getAncestors(): TreeNode<T>[] {
    let node: TreeNode<T> | null = this;
    const queue: TreeNode<T>[] = [];

    while (node) {
      node = node.parent;
      node && queue.unshift(node);
    }

    return queue;
  }

  toArray(): TreeNode<T>[] {
    return this.castDepthFirst();
  }

  /**
   * Casts the Tree depth-first to a List.
   *
   * If you want to cast breadth-first, just call:
   *
   * ```
   * new List(() => TreeNode.breadthFirst());
   * ```
   *
   * Note: you'll need to wrap it in an extra function, otherwise, you'll lose the `this` context.
   */
  toList(): List<TreeNode<T>> {
    return List.from(this);
  }

  /**
   * This isn't super helpful, but it prints this as a string
   */
  toString(): string {
    return `TreeNode<${JSON.stringify(this.value)}>`;
  }

  toJSON(): TreeNodeJSON {
    return {
      children: this.children.map(child => child.toJSON()),
      id: this.#id,
      parentId: this.#parent?.id ?? null,
      value: this.#value,
    };
  }

  /**
   * Serializes a Tree into an array for easier storage / transport
   */
  serialize(serializeValue = identity): SerializedTreeNode<T>[] {
    return serialize(this, serializeValue);
  }

  /**
   * Checks if two Trees are the same for value and for id
   */
  equals(node: TreeNode<T>, comparator?: BinaryFn<T, T, boolean>): boolean {
    return TreeNode.equals<T>(this, node, comparator);
  }
}
