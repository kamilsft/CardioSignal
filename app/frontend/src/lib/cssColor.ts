/** Canvas needs concrete colors: resolve "var(--token)" against the element's computed style. */
export function resolveColor(element: HTMLElement, color: string): string {
  const match = /^var\((--[\w-]+)\)$/.exec(color);
  if (!match) return color;
  return getComputedStyle(element).getPropertyValue(match[1]).trim() || "#888";
}
