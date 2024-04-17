import { identity } from '../../identity';
import { TreeNode } from './TreeNode';
import { SerializedTreeNode } from './types';

export const serialize = <T>(
  node: TreeNode<T>,
  serializeValue = identity,
): SerializedTreeNode<T>[] =>
  node.castBreadthFirst().map(x => ({
    id: x.id,
    parentId: x.parent?.id ?? null,
    value: serializeValue(x.value),
  }));
