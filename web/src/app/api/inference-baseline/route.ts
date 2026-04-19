import { NextRequest, NextResponse } from "next/server";
import { writeFile, unlink, mkdir } from "fs/promises";
import { join } from "path";
import { spawn } from "child_process";
import { existsSync } from "fs";

const PROJECT_ROOT = join(process.cwd(), "..");
const UPLOAD_DIR = join(process.cwd(), "tmp");
const INFERENCE_SCRIPT = join(PROJECT_ROOT, "scripts", "inference_baseline.py");
const VENV_PYTHON = join(PROJECT_ROOT, ".venv", "bin", "python");

export async function POST(request: NextRequest) {
  const startTime = Date.now();
  console.log("[baseline] Request received");

  try {
    const formData = await request.formData();
    const file = formData.get("image") as File | null;

    if (!file) {
      console.log("[baseline] No image in request");
      return NextResponse.json({ error: "No image provided" }, { status: 400 });
    }

    console.log(`[baseline] Image: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);

    if (!existsSync(UPLOAD_DIR)) {
      await mkdir(UPLOAD_DIR, { recursive: true });
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const filename = `upload_baseline_${Date.now()}.jpg`;
    const filepath = join(UPLOAD_DIR, filename);
    await writeFile(filepath, buffer);
    console.log(`[baseline] Saved to ${filepath}`);

    console.log("[baseline] Starting HOG+SVM inference...");
    const result = await runInference(filepath);
    console.log(`[baseline] Done! ${(result as { num_detections?: number }).num_detections} detections in ${Date.now() - startTime}ms total`);

    await unlink(filepath).catch(() => {});

    return NextResponse.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Inference failed";
    console.error(`[baseline] ERROR after ${Date.now() - startTime}ms: ${message}`);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

function runInference(imagePath: string): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const python = existsSync(VENV_PYTHON) ? VENV_PYTHON : "python3";
    console.log(`[baseline] Python: ${python}`);

    const proc = spawn(python, [INFERENCE_SCRIPT, imagePath], {
      cwd: PROJECT_ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    console.log(`[baseline] Process spawned (PID: ${proc.pid})`);

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    proc.stderr.on("data", (data) => {
      const chunk = data.toString();
      stderr += chunk;
      console.log(`[baseline:python] ${chunk.trimEnd()}`);
    });

    proc.on("close", (code) => {
      console.log(`[baseline] Process exited with code ${code}`);
      if (code !== 0) {
        reject(new Error(stderr || `Process exited with code ${code}`));
        return;
      }
      try {
        const result = JSON.parse(stdout);
        resolve(result);
      } catch {
        console.error(`[baseline] Failed to parse stdout: ${stdout.slice(0, 200)}`);
        reject(new Error("Failed to parse inference output"));
      }
    });

    proc.on("error", (err) => {
      console.error(`[baseline] Failed to spawn process: ${err.message}`);
      reject(new Error(`Failed to start Python: ${err.message}`));
    });

    // Timeout after 60 seconds
    setTimeout(() => {
      console.error("[baseline] TIMEOUT - killing process after 60s");
      proc.kill();
      reject(new Error("Inference timed out (60s)"));
    }, 60000);
  });
}
