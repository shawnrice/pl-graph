import { useState, useCallback } from 'react';

export const useForceUpdate = (): (() => void) => {
  const [, update] = useState(Object.create(null));

  return useCallback(() => update(Object.create(null)), [update]);
};
