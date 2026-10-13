import { useCallback, useRef, useState } from "react";

export type TaskState<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: T }
  | { status: "error"; error: string };

/** Run one async task at a time; results from superseded runs are ignored. */
export function useAsyncTask<T>() {
  const [state, setState] = useState<TaskState<T>>({ status: "idle" });
  const latest = useRef(0);

  const run = useCallback(async (task: () => Promise<T>) => {
    const id = ++latest.current;
    setState({ status: "loading" });
    try {
      const data = await task();
      if (id === latest.current) setState({ status: "success", data });
    } catch (error) {
      if (id === latest.current) {
        setState({ status: "error", error: error instanceof Error ? error.message : "Something went wrong." });
      }
    }
  }, []);

  const reset = useCallback(() => {
    latest.current++;
    setState({ status: "idle" });
  }, []);

  return { state, run, reset };
}
