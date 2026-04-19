/* eslint-disable @next/next/no-img-element */
"use client";

import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import {
  Upload,
  ArrowLeft,
  Loader2,
  ImageIcon,
  Box,
  X,
} from "lucide-react";

interface Detection {
  box: [number, number, number, number];
  score: number;
  label: number;
}

interface InferenceResult {
  detections: Detection[];
  num_detections: number;
  inference_time_ms: number;
  image_width: number;
  image_height: number;
}

export default function DemoPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Please upload an image file");
      return;
    }
    setFile(f);
    setResult(null);
    setResultImage(null);
    setError(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const runInference = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setResultImage(null);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const res = await fetch("/api/inference", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Inference failed (${res.status})`);
      }

      const data: InferenceResult = await res.json();
      setResult(data);
      drawDetections(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed");
    } finally {
      setLoading(false);
    }
  };

  const drawDetections = (data: InferenceResult) => {
    if (!preview) return;

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      data.detections.forEach((det) => {
        const [x1, y1, x2, y2] = det.box;
        const w = x2 - x1;
        const h = y2 - y1;

        const alpha = Math.max(0.4, det.score);
        ctx.strokeStyle = `rgba(37, 99, 235, ${alpha})`;
        ctx.lineWidth = Math.max(1, Math.min(3, img.width / 300));
        ctx.strokeRect(x1, y1, w, h);

        if (det.score > 0.3) {
          const label = `${(det.score * 100).toFixed(0)}%`;
          ctx.font = `${Math.max(10, img.width / 60)}px sans-serif`;
          const metrics = ctx.measureText(label);
          ctx.fillStyle = "rgba(37, 99, 235, 0.85)";
          ctx.fillRect(x1, y1 - 16, metrics.width + 6, 16);
          ctx.fillStyle = "white";
          ctx.fillText(label, x1 + 3, y1 - 3);
        }
      });

      // Convert canvas to data URL and display as <img> to avoid CSS sizing issues
      setResultImage(canvas.toDataURL("image/png"));
    };
    img.src = preview;
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setResultImage(null);
    setError(null);
  };

  return (
    <main className="min-h-screen bg-white">
      {/* Header */}
      <header className="border-b border-zinc-200 px-6 py-4 bg-white">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-sm text-zinc-500 hover:text-zinc-900 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </Link>
          <h1 className="font-serif text-lg text-zinc-900">
            Dense<span className="italic">Vision</span>
          </h1>
          <div className="w-16" />
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <h2 className="font-serif text-3xl md:text-4xl mb-3 text-zinc-900">
            Live Inference Demo
          </h2>
          <p className="text-zinc-500">
            Upload a retail shelf image to detect and segment products
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left: Upload & Image Preview */}
          <div className="space-y-4">
            <AnimatePresence mode="wait">
              {!preview ? (
                <motion.div
                  key="dropzone"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`
                    aspect-[4/3] rounded-2xl border-2 border-dashed cursor-pointer
                    flex flex-col items-center justify-center gap-4 transition-colors
                    ${
                      dragOver
                        ? "border-blue-500 bg-blue-50"
                        : "border-zinc-300 hover:border-blue-400 bg-zinc-50"
                    }
                  `}
                >
                  <div className="w-14 h-14 rounded-2xl bg-zinc-200 flex items-center justify-center">
                    <Upload className="w-6 h-6 text-zinc-500" />
                  </div>
                  <div className="text-center">
                    <p className="text-sm font-medium text-zinc-700">
                      Drop an image here or click to upload
                    </p>
                    <p className="text-xs text-zinc-400 mt-1">
                      JPG, PNG up to 10MB
                    </p>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) handleFile(f);
                    }}
                  />
                </motion.div>
              ) : (
                <motion.div
                  key="preview"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="relative"
                >
                  <div className="rounded-2xl overflow-hidden bg-zinc-100 border border-zinc-200 relative">
                    <img
                      src={resultImage || preview}
                      alt={resultImage ? "Detection results" : "Upload preview"}
                      className="w-full h-auto max-h-[70vh] object-contain"
                    />
                  </div>
                  <button
                    onClick={reset}
                    className="absolute top-3 right-3 w-8 h-8 rounded-full bg-white/80 backdrop-blur border border-zinc-200 flex items-center justify-center hover:bg-white transition-colors shadow-sm"
                  >
                    <X className="w-4 h-4 text-zinc-600" />
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Action buttons */}
            {preview && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex gap-3"
              >
                <button
                  onClick={runInference}
                  disabled={loading}
                  className="flex-1 inline-flex items-center justify-center gap-2 bg-zinc-900 text-white px-6 py-3 rounded-xl font-medium text-sm hover:bg-zinc-800 transition-colors disabled:opacity-50 shadow-sm"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Box className="w-4 h-4" />
                      Run Detection
                    </>
                  )}
                </button>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-4 py-3 rounded-xl border border-zinc-300 text-sm text-zinc-500 hover:text-zinc-900 hover:border-zinc-400 transition-colors"
                >
                  <ImageIcon className="w-4 h-4" />
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleFile(f);
                  }}
                />
              </motion.div>
            )}
          </div>

          {/* Right: Results */}
          <div className="space-y-4">
            <AnimatePresence mode="wait">
              {error && (
                <motion.div
                  key="error"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm"
                >
                  {error}
                </motion.div>
              )}

              {loading && (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="aspect-[4/3] rounded-2xl border border-zinc-200 bg-zinc-50 flex flex-col items-center justify-center gap-4"
                >
                  <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
                  <div className="text-center">
                    <p className="text-sm font-medium text-zinc-700">Running YOLACT inference...</p>
                    <p className="text-xs text-zinc-400 mt-1">
                      This may take 10-30 seconds
                    </p>
                  </div>
                </motion.div>
              )}

              {result && !loading && (
                <motion.div
                  key="results"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-4"
                >
                  {/* Summary */}
                  <div className="border border-zinc-200 rounded-2xl p-6 bg-zinc-50">
                    <h3 className="font-medium text-zinc-900 mb-4">
                      Detection Results
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center p-3 rounded-xl bg-white border border-zinc-200 shadow-sm">
                        <p className="text-2xl font-serif text-blue-600">
                          {result.num_detections}
                        </p>
                        <p className="text-xs text-zinc-400 mt-1">
                          Objects Detected
                        </p>
                      </div>
                      <div className="text-center p-3 rounded-xl bg-white border border-zinc-200 shadow-sm">
                        <p className="text-2xl font-serif text-blue-600">
                          {result.inference_time_ms.toFixed(0)}ms
                        </p>
                        <p className="text-xs text-zinc-400 mt-1">
                          Inference Time
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Top detections */}
                  {result.detections.length > 0 && (
                    <div className="border border-zinc-200 rounded-2xl p-6 bg-zinc-50">
                      <h3 className="font-medium text-zinc-900 mb-3">
                        Top Detections
                      </h3>
                      <div className="space-y-2 max-h-64 overflow-y-auto">
                        {result.detections.slice(0, 20).map((det, i) => (
                          <div
                            key={i}
                            className="flex items-center justify-between py-2 border-b border-zinc-100 last:border-0"
                          >
                            <span className="text-xs text-zinc-500 tabular-nums">
                              #{i + 1} &nbsp;[{det.box.map((v) => v.toFixed(0)).join(", ")}]
                            </span>
                            <span
                              className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                                det.score > 0.5
                                  ? "bg-green-50 text-green-600 border border-green-200"
                                  : det.score > 0.2
                                  ? "bg-amber-50 text-amber-600 border border-amber-200"
                                  : "bg-zinc-100 text-zinc-500 border border-zinc-200"
                              }`}
                            >
                              {(det.score * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Image dimensions */}
                  <div className="text-xs text-zinc-400 text-center">
                    Input: {result.image_width}x{result.image_height} &rarr;
                    resized to 550x550 for inference
                  </div>
                </motion.div>
              )}

              {!result && !loading && !error && (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="aspect-[4/3] rounded-2xl border border-zinc-200 bg-zinc-50 flex flex-col items-center justify-center gap-4 text-center px-8"
                >
                  <Box className="w-8 h-8 text-zinc-300" />
                  <div>
                    <p className="text-sm text-zinc-500">
                      Upload an image and click &quot;Run Detection&quot; to see results
                    </p>
                    <p className="text-xs text-zinc-400 mt-2">
                      YOLACT + MobileNetV3 + CBAM + Soft-NMS
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </main>
  );
}
