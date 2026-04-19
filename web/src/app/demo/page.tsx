/* eslint-disable @next/next/no-img-element */
"use client";

import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import {
  Upload,
  ArrowLeft,
  Loader2,
  ImageIcon,
  X,
  Maximize2,
  Download,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Brain,
  ScanLine,
  SlidersHorizontal,
} from "lucide-react";

type ModelType = "yolact" | "hogsvm";

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
  model?: string;
}

const MODEL_INFO: Record<ModelType, { name: string; tag: string; desc: string; endpoint: string }> = {
  yolact: {
    name: "YOLACT",
    tag: "Deep Learning",
    desc: "MobileNetV3 + FPN + CBAM + Soft-NMS",
    endpoint: "/api/inference",
  },
  hogsvm: {
    name: "HOG + SVM",
    tag: "Classical ML",
    desc: "HOG features + Linear SVM + Sliding Window",
    endpoint: "/api/inference-baseline",
  },
};

const SAMPLE_IMAGES = [
  { src: "/samples/shelf_dense.png", label: "Dense Shelf" },
  { src: "/samples/shelf_medium.png", label: "Medium Shelf" },
  { src: "/samples/shelf_sparse.png", label: "Sparse Shelf" },
];

export default function DemoPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelType>("yolact");
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.1);
  const [fullscreen, setFullscreen] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0 });
  const panOffset = useRef({ x: 0, y: 0 });
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Filter detections by confidence threshold (client-side, no re-inference)
  const filteredDetections = useMemo(() => {
    if (!result) return [];
    return result.detections.filter((d) => d.score >= confidenceThreshold);
  }, [result, confidenceThreshold]);

  const drawDetections = useCallback((detections: Detection[]) => {
    if (!preview) return;

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      detections.forEach((det) => {
        const [x1, y1, x2, y2] = det.box;
        const w = x2 - x1;
        const h = y2 - y1;

        let r: number, g: number, b: number;
        if (det.score > 0.5) {
          const t = (det.score - 0.5) / 0.5;
          r = Math.round(255 * (1 - t));
          g = 220;
          b = 0;
        } else {
          const t = det.score / 0.5;
          r = 255;
          g = Math.round(180 * t);
          b = 0;
        }

        const lineW = Math.max(2, Math.min(4, img.width / 200));

        ctx.shadowColor = `rgba(0, 0, 0, 0.5)`;
        ctx.shadowBlur = 3;
        ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.lineWidth = lineW;
        ctx.strokeRect(x1, y1, w, h);
        ctx.shadowBlur = 0;

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.08)`;
        ctx.fillRect(x1, y1, w, h);

        if (det.score > 0.3) {
          const label = `${(det.score * 100).toFixed(0)}%`;
          const fontSize = Math.max(11, img.width / 50);
          ctx.font = `bold ${fontSize}px sans-serif`;
          const metrics = ctx.measureText(label);
          const labelH = fontSize + 4;
          const labelW = metrics.width + 8;

          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
          ctx.fillRect(x1, y1 - labelH, labelW, labelH);

          ctx.fillStyle = det.score > 0.5 ? "#000" : "#fff";
          ctx.fillText(label, x1 + 4, y1 - 4);
        }
      });

      setResultImage(canvas.toDataURL("image/png"));
    };
    img.src = preview;
  }, [preview]);

  // Redraw detections when threshold changes
  useEffect(() => {
    if (result && preview) {
      drawDetections(filteredDetections);
    }
  }, [confidenceThreshold, result, preview, drawDetections, filteredDetections]);

  const closeFullscreen = useCallback(() => {
    setFullscreen(false);
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && fullscreen) closeFullscreen();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [fullscreen, closeFullscreen]);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Please upload an image file");
      return;
    }
    setFile(f);
    setResult(null);
    setResultImage(null);
    setError(null);
    setConfidenceThreshold(0.1);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const handleSampleImage = async (src: string) => {
    setResult(null);
    setResultImage(null);
    setError(null);
    setConfidenceThreshold(0.1);
    setPreview(src);

    // Convert to File for the API
    const res = await fetch(src);
    const blob = await res.blob();
    const f = new File([blob], src.split("/").pop() || "sample.png", { type: blob.type });
    setFile(f);
  };

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

    const modelConfig = MODEL_INFO[selectedModel];

    try {
      const formData = new FormData();
      formData.append("image", file);

      const res = await fetch(modelConfig.endpoint, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Inference failed (${res.status})`);
      }

      const data: InferenceResult = await res.json();
      data.model = modelConfig.name;
      setResult(data);
      const filtered = data.detections.filter((d) => d.score >= confidenceThreshold);
      drawDetections(filtered);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setResultImage(null);
    setError(null);
    setConfidenceThreshold(0.1);
  };

  const openFullscreen = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setFullscreen(true);
  };

  const handleZoomIn = () => setZoom((z) => Math.min(z * 1.3, 8));
  const handleZoomOut = () => setZoom((z) => Math.max(z / 1.3, 0.5));
  const handleZoomReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      setZoom((z) => Math.min(z * 1.1, 8));
    } else {
      setZoom((z) => Math.max(z / 1.1, 0.5));
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom <= 1) return;
    setIsPanning(true);
    panStart.current = { x: e.clientX, y: e.clientY };
    panOffset.current = { x: pan.x, y: pan.y };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isPanning) return;
    setPan({
      x: panOffset.current.x + (e.clientX - panStart.current.x),
      y: panOffset.current.y + (e.clientY - panStart.current.y),
    });
  };

  const handleMouseUp = () => setIsPanning(false);

  const downloadImage = () => {
    if (!resultImage) return;
    const link = document.createElement("a");
    link.download = `detection_${selectedModel}_${Date.now()}.png`;
    link.href = resultImage;
    link.click();
  };

  const modelInfo = MODEL_INFO[selectedModel];

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
        <div className="text-center mb-8">
          <h2 className="font-serif text-3xl md:text-4xl mb-3 text-zinc-900">
            Live Inference Demo
          </h2>
          <p className="text-zinc-500">
            Upload a retail shelf image to detect and segment products
          </p>
        </div>

        {/* Model selector */}
        <div className="flex justify-center mb-10">
          <div className="inline-flex bg-zinc-100 rounded-2xl p-1.5 gap-1.5">
            <button
              onClick={() => { setSelectedModel("yolact"); setResult(null); setResultImage(null); }}
              className={`
                flex items-center gap-2.5 px-5 py-3 rounded-xl text-sm font-medium transition-all
                ${selectedModel === "yolact"
                  ? "bg-white text-zinc-900 shadow-sm border border-zinc-200"
                  : "text-zinc-500 hover:text-zinc-700"
                }
              `}
            >
              <Brain className="w-4 h-4" />
              <div className="text-left">
                <div className="font-medium">YOLACT</div>
                <div className={`text-[10px] ${selectedModel === "yolact" ? "text-blue-600" : "text-zinc-400"}`}>
                  Deep Learning
                </div>
              </div>
            </button>
            <button
              onClick={() => { setSelectedModel("hogsvm"); setResult(null); setResultImage(null); }}
              className={`
                flex items-center gap-2.5 px-5 py-3 rounded-xl text-sm font-medium transition-all
                ${selectedModel === "hogsvm"
                  ? "bg-white text-zinc-900 shadow-sm border border-zinc-200"
                  : "text-zinc-500 hover:text-zinc-700"
                }
              `}
            >
              <ScanLine className="w-4 h-4" />
              <div className="text-left">
                <div className="font-medium">HOG + SVM</div>
                <div className={`text-[10px] ${selectedModel === "hogsvm" ? "text-amber-600" : "text-zinc-400"}`}>
                  Classical ML
                </div>
              </div>
            </button>
          </div>
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
                  className="space-y-4"
                >
                  <div
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    className={`
                      aspect-[4/3] rounded-2xl border-2 border-dashed cursor-pointer
                      flex flex-col items-center justify-center gap-4 transition-colors
                      ${dragOver
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
                  </div>

                  {/* Sample images */}
                  <div>
                    <p className="text-xs text-zinc-400 mb-2.5 text-center">Or try a sample image</p>
                    <div className="grid grid-cols-3 gap-2.5">
                      {SAMPLE_IMAGES.map((sample) => (
                        <button
                          key={sample.src}
                          onClick={() => handleSampleImage(sample.src)}
                          className="group relative rounded-xl overflow-hidden border border-zinc-200 hover:border-zinc-400 transition-all hover:shadow-sm"
                        >
                          <img
                            src={sample.src}
                            alt={sample.label}
                            className="w-full aspect-square object-cover"
                          />
                          <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/60 to-transparent px-2 py-1.5">
                            <p className="text-[10px] text-white font-medium">{sample.label}</p>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
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
                    {resultImage && result && (
                      <div className="absolute top-3 left-3">
                        <span className={`
                          text-xs font-medium px-2.5 py-1 rounded-lg backdrop-blur
                          ${result.model === "YOLACT"
                            ? "bg-blue-600/80 text-white"
                            : "bg-amber-500/80 text-white"
                          }
                        `}>
                          {result.model}
                        </span>
                      </div>
                    )}
                    {resultImage && (
                      <div className="absolute bottom-3 right-3 flex gap-2">
                        <button
                          onClick={openFullscreen}
                          title="View fullscreen"
                          className="w-9 h-9 rounded-lg bg-black/60 backdrop-blur text-white flex items-center justify-center hover:bg-black/80 transition-colors"
                        >
                          <Maximize2 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={downloadImage}
                          title="Download image"
                          className="w-9 h-9 rounded-lg bg-black/60 backdrop-blur text-white flex items-center justify-center hover:bg-black/80 transition-colors"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                      </div>
                    )}
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
                  className={`
                    flex-1 inline-flex items-center justify-center gap-2 text-white px-6 py-3 rounded-xl font-medium text-sm transition-colors disabled:opacity-50 shadow-sm
                    ${selectedModel === "yolact"
                      ? "bg-zinc-900 hover:bg-zinc-800"
                      : "bg-amber-600 hover:bg-amber-700"
                    }
                  `}
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Running {modelInfo.name}...
                    </>
                  ) : (
                    <>
                      {selectedModel === "yolact" ? <Brain className="w-4 h-4" /> : <ScanLine className="w-4 h-4" />}
                      Run {modelInfo.name}
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
                  <Loader2 className={`w-8 h-8 animate-spin ${selectedModel === "yolact" ? "text-blue-600" : "text-amber-600"}`} />
                  <div className="text-center">
                    <p className="text-sm font-medium text-zinc-700">
                      Running {modelInfo.name} inference...
                    </p>
                    <p className="text-xs text-zinc-400 mt-1">
                      {selectedModel === "yolact"
                        ? "This may take 10-30 seconds"
                        : "This may take 5-10 seconds"
                      }
                    </p>
                    <p className="text-xs text-zinc-300 mt-2">
                      {modelInfo.desc}
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
                  {/* Model badge */}
                  <div className={`
                    rounded-xl px-4 py-2.5 flex items-center justify-between
                    ${result.model === "YOLACT"
                      ? "bg-blue-50 border border-blue-200"
                      : "bg-amber-50 border border-amber-200"
                    }
                  `}>
                    <div className="flex items-center gap-2">
                      {result.model === "YOLACT" ? (
                        <Brain className="w-4 h-4 text-blue-600" />
                      ) : (
                        <ScanLine className="w-4 h-4 text-amber-600" />
                      )}
                      <span className={`text-sm font-medium ${result.model === "YOLACT" ? "text-blue-700" : "text-amber-700"}`}>
                        {result.model}
                      </span>
                    </div>
                    <span className={`text-xs ${result.model === "YOLACT" ? "text-blue-500" : "text-amber-500"}`}>
                      {result.model === "YOLACT" ? "Deep Learning" : "Classical ML"}
                    </span>
                  </div>

                  {/* Summary */}
                  <div className="border border-zinc-200 rounded-2xl p-6 bg-zinc-50">
                    <h3 className="font-medium text-zinc-900 mb-4">
                      Detection Results
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center p-3 rounded-xl bg-white border border-zinc-200 shadow-sm">
                        <p className={`text-2xl font-serif ${result.model === "YOLACT" ? "text-blue-600" : "text-amber-600"}`}>
                          {filteredDetections.length}
                        </p>
                        <p className="text-xs text-zinc-400 mt-1">
                          Objects Shown
                        </p>
                        {filteredDetections.length !== result.num_detections && (
                          <p className="text-[10px] text-zinc-300 mt-0.5">
                            {result.num_detections} total
                          </p>
                        )}
                      </div>
                      <div className="text-center p-3 rounded-xl bg-white border border-zinc-200 shadow-sm">
                        <p className={`text-2xl font-serif ${result.model === "YOLACT" ? "text-blue-600" : "text-amber-600"}`}>
                          {result.inference_time_ms.toFixed(0)}ms
                        </p>
                        <p className="text-xs text-zinc-400 mt-1">
                          Inference Time
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Confidence threshold slider */}
                  <div className="border border-zinc-200 rounded-2xl p-5 bg-zinc-50">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <SlidersHorizontal className="w-3.5 h-3.5 text-zinc-500" />
                        <span className="text-sm font-medium text-zinc-700">Confidence Threshold</span>
                      </div>
                      <span className="text-sm font-mono font-medium text-zinc-900 bg-white px-2 py-0.5 rounded border border-zinc-200">
                        {(confidenceThreshold * 100).toFixed(0)}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.05"
                      max="0.95"
                      step="0.05"
                      value={confidenceThreshold}
                      onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-zinc-900"
                    />
                    <div className="flex justify-between mt-1.5">
                      <span className="text-[10px] text-zinc-400">5% (show all)</span>
                      <span className="text-[10px] text-zinc-400">95% (high confidence only)</span>
                    </div>
                  </div>

                  {/* No detections message */}
                  {filteredDetections.length === 0 && (
                    <div className="border border-amber-200 rounded-xl p-4 bg-amber-50 text-center">
                      <p className="text-sm text-amber-700 font-medium">No objects at this threshold</p>
                      <p className="text-xs text-amber-500 mt-1">
                        {confidenceThreshold > 0.3
                          ? "Try lowering the confidence threshold to see more detections"
                          : result.model === "HOG+SVM"
                          ? "HOG+SVM struggles with real-world images \u2014 try YOLACT for better results"
                          : "Try a different image"
                        }
                      </p>
                    </div>
                  )}

                  {/* Top detections */}
                  {filteredDetections.length > 0 && (
                    <div className="border border-zinc-200 rounded-2xl p-6 bg-zinc-50">
                      <h3 className="font-medium text-zinc-900 mb-3">
                        Top Detections
                      </h3>
                      <div className="space-y-2 max-h-48 overflow-y-auto">
                        {filteredDetections.slice(0, 20).map((det, i) => (
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
                    Input: {result.image_width}x{result.image_height}
                    {result.model === "YOLACT"
                      ? " \u2192 resized to 550x550 for inference"
                      : " \u2192 resized to max 600px for sliding window"
                    }
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
                  {selectedModel === "yolact" ? (
                    <Brain className="w-8 h-8 text-zinc-300" />
                  ) : (
                    <ScanLine className="w-8 h-8 text-zinc-300" />
                  )}
                  <div>
                    <p className="text-sm text-zinc-500">
                      Upload an image and click &quot;Run {modelInfo.name}&quot; to see results
                    </p>
                    <p className="text-xs text-zinc-400 mt-2">
                      {modelInfo.desc}
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      {/* Fullscreen overlay */}
      <AnimatePresence>
        {fullscreen && resultImage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black flex flex-col"
          >
            <div className="flex items-center justify-between px-4 py-3 bg-black/80 border-b border-white/10">
              <div className="flex items-center gap-3">
                <span className={`
                  text-xs font-medium px-2 py-0.5 rounded
                  ${result?.model === "YOLACT" ? "bg-blue-600 text-white" : "bg-amber-500 text-white"}
                `}>
                  {result?.model}
                </span>
                <span className="text-sm text-white/70">
                  {filteredDetections.length} detections &middot; {result?.inference_time_ms.toFixed(0)}ms
                </span>
                <span className="text-xs text-white/40">
                  {Math.round(zoom * 100)}%
                </span>
              </div>
              <div className="flex items-center gap-1">
                <button onClick={handleZoomOut} title="Zoom out" className="w-9 h-9 rounded-lg text-white/70 hover:text-white hover:bg-white/10 flex items-center justify-center transition-colors">
                  <ZoomOut className="w-4 h-4" />
                </button>
                <button onClick={handleZoomReset} title="Reset zoom" className="w-9 h-9 rounded-lg text-white/70 hover:text-white hover:bg-white/10 flex items-center justify-center transition-colors">
                  <RotateCcw className="w-4 h-4" />
                </button>
                <button onClick={handleZoomIn} title="Zoom in" className="w-9 h-9 rounded-lg text-white/70 hover:text-white hover:bg-white/10 flex items-center justify-center transition-colors">
                  <ZoomIn className="w-4 h-4" />
                </button>
                <div className="w-px h-5 bg-white/20 mx-1" />
                <button onClick={downloadImage} title="Download image" className="w-9 h-9 rounded-lg text-white/70 hover:text-white hover:bg-white/10 flex items-center justify-center transition-colors">
                  <Download className="w-4 h-4" />
                </button>
                <button onClick={closeFullscreen} title="Close fullscreen" className="w-9 h-9 rounded-lg text-white/70 hover:text-white hover:bg-white/10 flex items-center justify-center transition-colors">
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            <div
              className="flex-1 overflow-hidden cursor-grab active:cursor-grabbing select-none"
              onWheel={handleWheel}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            >
              <div className="w-full h-full flex items-center justify-center">
                <img
                  src={resultImage}
                  alt="Detection results - fullscreen"
                  draggable={false}
                  style={{
                    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                    transition: isPanning ? "none" : "transform 0.2s ease-out",
                    maxWidth: "100%",
                    maxHeight: "100%",
                    objectFit: "contain",
                  }}
                />
              </div>
            </div>

            <div className="px-4 py-2 bg-black/80 border-t border-white/10 text-center">
              <p className="text-xs text-white/30">
                Scroll to zoom &middot; Drag to pan &middot; Esc to close
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </main>
  );
}
