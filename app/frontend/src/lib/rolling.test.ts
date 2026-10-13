import { describe, expect, it } from "vitest";
import { appendRolling } from "./rolling";

describe("appendRolling", () => {
  it("appends new samples", () => {
    expect(appendRolling([1, 2], [3], 5)).toEqual([1, 2, 3]);
  });

  it("keeps only the latest `max` samples", () => {
    expect(appendRolling([1, 2, 3, 4], [5, 6, 7], 5)).toEqual([3, 4, 5, 6, 7]);
  });

  it("handles a chunk longer than the window", () => {
    expect(appendRolling([1], [2, 3, 4, 5], 2)).toEqual([4, 5]);
  });

  it("does not modify its inputs", () => {
    const prev = [1, 2];
    const next = [3];
    appendRolling(prev, next, 2);
    expect(prev).toEqual([1, 2]);
    expect(next).toEqual([3]);
  });

  it("returns the same array when there is nothing new", () => {
    const prev = [1, 2];
    expect(appendRolling(prev, [], 5)).toBe(prev);
  });
});
