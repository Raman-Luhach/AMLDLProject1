import { NextRequest, NextResponse } from "next/server";
import { writeFile, unlink, mkdir } from "fs/promises";
import { join } from "path";
import { spawn } from "child_process";
import { existsSync } from "fs";

const PROJECT_ROOT = join(process.cwd(), "..");
const UPLOAD_DIR = join(process.cwd(), "tmp");
const INFERENCE_SCRIPT = join(PROJECT_ROOT, "scripts", "inference_hybrid.py");
const VENV_PYTHON = join(PROJECT_ROOT, ".venv", "bin", "python");

export async function POST(request: NextRequest) {
  const startTime = Date.now();
  console.log("[hybrid] Request received");

  try {
    const formData = await request.formData();
    const file = formData.get("image") as File | null;

    if (!file) {
      console.log("[hybrid] No image in request");
      return NextResponse.json({ error: "No image provided" }, { status: 400 });
    }

    console.log(`[hybrid] Image: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);

    if (!existsSync(UPLOAD_DIR)) {
      await mkdir(UPLOAD_DIR, { recursive: true });
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const filename = `upload_hybrid_${Date.now()}.jpg`;
    const filepath = join(UPLOAD_DIR, filename);
    await writeFile(filepath, buffer);
    console.log(`[hybrid] Saved to ${filepath}`);

    console.log("[hybrid] Starting Hybrid inference...");
    const result = await runInference(filepath);
    console.log(`[hybrid] Done! ${(result as { num_detections?: number }).num_detections} detections in ${Date.now() - startTime}ms total`);

    await unlink(filepath).catch(() => {});

    return NextResponse.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Inference failed";
    console.error(`[hybrid] ERROR after ${Date.now() - startTime}ms: ${message}`);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

function runInference(imagePath: string): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const python = existsSync(VENV_PYTHON) ? VENV_PYTHON : "python3";
    console.log(`[hybrid] Python: ${python}`);
    console.log(`[hybrid] Script: ${INFERENCE_SCRIPT}`);

    const proc = spawn(python, [INFERENCE_SCRIPT, imagePath], {
      cwd: PROJECT_ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    console.log(`[hybrid] Process spawned (PID: ${proc.pid})`);

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    proc.stderr.on("data", (data) => {
      const chunk = data.toString();
      stderr += chunk;
      console.log(`[hybrid:python] ${chunk.trimEnd()}`);
    });

    proc.on("close", (code) => {
      console.log(`[hybrid] Process exited with code ${code}`);
      if (code !== 0) {
        reject(new Error(stderr || `Process exited with code ${code}`));
        return;
      }
      try {
        const result = JSON.parse(stdout);
        resolve(result);
      } catch {
        console.error(`[hybrid] Failed to parse stdout: ${stdout.slice(0, 200)}`);
        reject(new Error("Failed to parse inference output"));
      }
    });

    proc.on("error", (err) => {
      console.error(`[hybrid] Failed to spawn process: ${err.message}`);
      reject(new Error(`Failed to start Python: ${err.message}`));
    });

    // Timeout after 180 seconds (hybrid model loads spatial engine + YOLACT)
    setTimeout(() => {
      console.error("[hybrid] TIMEOUT - killing process after 180s");
      proc.kill();
      reject(new Error("Inference timed out (180s)"));
    }, 180000);
  });
}
