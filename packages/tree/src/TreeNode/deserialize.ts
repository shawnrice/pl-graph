import { identity } from '../../identity';
import { TreeNode } from './TreeNode';
import { SerializedTreeNode } from './types';

export const deserialize = <T>(
  serialized: SerializedTreeNode<T>[],
  deserializeValue: (x: any) => T = identity,
): TreeNode<T> => {
  const nodeMap = new Map<string, TreeNode<T>>();

  let root: TreeNode<T> | null = null;

  serialized.forEach(x => {
    if (x.parentId && nodeMap.has(x.parentId)) {
      const parent = nodeMap.get(x.parentId)!;
      const child = parent.createChild(deserializeValue(x.value), x.id);
      nodeMap.set(child.id, child);
    } else {
      if (root) {
        throw new Error('More than one root');
      }

      root = new TreeNode(null, deserializeValue(x.value), x.id);
      nodeMap.set(root.id, root);
    }
  });

  // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
  if (!root) {
    throw new Error('Failed to find a root node');
  }

  return root;
};
