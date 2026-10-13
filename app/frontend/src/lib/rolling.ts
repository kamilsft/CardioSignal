/** A new array holding the latest `max` samples of `prev` followed by `next`. Inputs are not modified. */
export function appendRolling(prev: readonly number[], next: readonly number[], max: number): readonly number[] {
  if (next.length === 0) return prev;
  if (next.length >= max) return next.slice(next.length - max);
  const keep = Math.max(0, max - next.length);
  return [...prev.slice(Math.max(0, prev.length - keep)), ...next];
}
