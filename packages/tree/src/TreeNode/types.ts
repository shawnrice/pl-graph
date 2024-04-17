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
