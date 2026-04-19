import { NextRequest, NextResponse } from "next/server";
import { writeFile, unlink, mkdir } from "fs/promises";
import { join } from "path";
import { spawn } from "child_process";
import { existsSync } from "fs";

const PROJECT_ROOT = join(process.cwd(), "..");
const UPLOAD_DIR = join(process.cwd(), "tmp");
const INFERENCE_SCRIPT = join(PROJECT_ROOT, "scripts", "inference_api.py");
const VENV_PYTHON = join(PROJECT_ROOT, ".venv", "bin", "python");

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("image") as File | null;

    if (!file) {
      return NextResponse.json({ error: "No image provided" }, { status: 400 });
    }

    // Ensure upload dir exists
    if (!existsSync(UPLOAD_DIR)) {
      await mkdir(UPLOAD_DIR, { recursive: true });
    }

    // Save uploaded file
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const filename = `upload_${Date.now()}.jpg`;
    const filepath = join(UPLOAD_DIR, filename);
    await writeFile(filepath, buffer);

    // Run inference via Python script
    const result = await runInference(filepath);

    // Clean up uploaded file
    await unlink(filepath).catch(() => {});

    return NextResponse.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Inference failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

function runInference(imagePath: string): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const python = existsSync(VENV_PYTHON) ? VENV_PYTHON : "python3";

    const proc = spawn(python, [INFERENCE_SCRIPT, imagePath], {
      cwd: PROJECT_ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    proc.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `Process exited with code ${code}`));
        return;
      }
      try {
        const result = JSON.parse(stdout);
        resolve(result);
      } catch {
        reject(new Error("Failed to parse inference output"));
      }
    });

    // Timeout after 60 seconds
    setTimeout(() => {
      proc.kill();
      reject(new Error("Inference timed out (60s)"));
    }, 60000);
  });
}
