import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { UploadForm, validateUpload } from "./UploadForm";

describe("validateUpload", () => {
  const csv = new File(["1"], "a.csv");
  const cases: { name: string; file: File | null; rate: string; message: string }[] = [
    { name: "no file", file: null, rate: "500", message: "Choose a file" },
    { name: "missing rate", file: csv, rate: "", message: "sample rate" },
    { name: "fractional rate", file: csv, rate: "12.5", message: "sample rate" },
    { name: "rate too low", file: csv, rate: "100", message: "sample rate" },
  ];
  it.each(cases)("rejects $name", ({ file, rate, message }) => {
    expect(validateUpload(file, rate).error).toContain(message);
  });

  it("accepts CSV with a rate and WAV without one", () => {
    expect(validateUpload(csv, "500")).toEqual({ error: null, sampleRate: 500 });
    expect(validateUpload(new File(["x"], "a.WAV"), "")).toEqual({ error: null, sampleRate: null });
  });

  it("rejects files over 5 MB", () => {
    const big = new File([new Uint8Array(5 * 1024 * 1024 + 1)], "a.csv");
    expect(validateUpload(big, "500").error).toContain("too large");
  });
});

describe("UploadForm", () => {
  it("shows an error instead of submitting invalid input", async () => {
    const onSubmit = vi.fn();
    render(<UploadForm signalName="ECG" onSubmit={onSubmit} />);
    await userEvent.click(screen.getByRole("button", { name: "Translate file" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Choose a file");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("submits a valid CSV with its sample rate", async () => {
    const onSubmit = vi.fn();
    const { container } = render(<UploadForm signalName="ECG" onSubmit={onSubmit} />);
    const file = new File(["1\n2"], "ecg.csv");
    await userEvent.upload(container.querySelector("input[type=file]") as HTMLInputElement, file);
    await userEvent.type(screen.getByPlaceholderText("e.g. 500"), "500");
    await userEvent.click(screen.getByRole("button", { name: "Translate file" }));
    expect(onSubmit).toHaveBeenCalledWith(file, 500);
  });
});
