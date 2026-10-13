import { expect, test } from "@playwright/test";

const SHOTS = "e2e/screenshots";

function syntheticEcgCsv(seconds: number, rate: number, bpm = 70): string {
  const period = Math.round((rate * 60) / bpm);
  const rows = Array.from({ length: seconds * rate }, (_, i) => {
    const d = i % period;
    const qrs = Math.exp(-((d - 20) ** 2) / 8);              // sharp R-peak
    const tWave = 0.3 * Math.exp(-((d - 140) ** 2) / 600);   // slower T-wave
    return (qrs + tWave).toFixed(5);
  });
  return ["ecg", ...rows].join("\n");
}

test.describe("CardioSignal", () => {
  test("translates a demo recording in both directions", async ({ page }, info) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "CardioSignal" })).toBeVisible();

    await page.getByRole("button", { name: /Healthy patient 1/ }).click();
    await expect(page.getByText("Generated heart sound", { exact: true })).toBeVisible();
    await expect(page.getByText("Heart rate (input)")).toBeVisible();
    await expect(page.getByRole("button", { name: /Play generated heart sound/ })).toBeVisible();
    await page.screenshot({ path: `${SHOTS}/${info.project.name}-ecg-to-pcg.png`, fullPage: true });

    await page.getByRole("radio", { name: "Heart sound → ECG" }).click();
    await expect(page.getByText("Generated ECG", { exact: true })).toBeVisible();
    await expect(page.getByText("Heart-sound quality")).toBeVisible();
    await page.screenshot({ path: `${SHOTS}/${info.project.name}-pcg-to-ecg.png`, fullPage: true });
  });

  test("plays the generated heart sound", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /Healthy patient 2/ }).click();
    const play = page.getByRole("button", { name: /Play generated heart sound/ });
    await play.click();
    await expect(page.getByRole("button", { name: /Stop/ })).toBeVisible();
    await expect(page.getByRole("button", { name: /Play generated heart sound/ })).toBeVisible({ timeout: 10_000 });
  });

  test("uploads a CSV file and validates bad input", async ({ page }, info) => {
    await page.goto("/");
    await page.getByRole("tab", { name: /Upload your ECG/ }).click();

    const fileInput = page.locator("input[type=file]");
    await fileInput.setInputFiles({ name: "short.csv", mimeType: "text/csv", buffer: Buffer.from(syntheticEcgCsv(1, 500)) });
    await page.getByPlaceholder("e.g. 500").fill("500");
    await page.getByRole("button", { name: "Translate file" }).click();
    await expect(page.getByRole("alert")).toContainText("at least 3 s");

    await fileInput.setInputFiles({ name: "ecg.csv", mimeType: "text/csv", buffer: Buffer.from(syntheticEcgCsv(10, 500)) });
    await page.getByRole("button", { name: "Translate file" }).click();
    await expect(page.getByText("Generated heart sound", { exact: true })).toBeVisible();
    await expect(page.getByText("70 bpm").first()).toBeVisible();
    await page.screenshot({ path: `${SHOTS}/${info.project.name}-upload.png`, fullPage: true });
  });
});
