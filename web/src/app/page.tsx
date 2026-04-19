/* eslint-disable @next/next/no-img-element */
"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { ArrowRight, ChevronDown, ExternalLink } from "lucide-react";

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" as const } },
};
const stagger = { visible: { transition: { staggerChildren: 0.1 } } };

export default function Home() {
  return (
    <main className="bg-white">
      {/* ── SLIDE 1: HERO ── */}
      <section className="slide-section relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(37,99,235,0.05),transparent_70%)]" />
        <div className="max-w-5xl mx-auto px-6 text-center relative z-10">
          <motion.div initial="hidden" animate="visible" variants={stagger}>
            <motion.p variants={fadeUp} className="text-sm tracking-widest uppercase text-zinc-400 mb-6">
              B.Tech Final Year Project &middot; Applied ML &amp; Deep Learning
            </motion.p>
            <motion.h1 variants={fadeUp} className="font-serif text-5xl md:text-7xl lg:text-8xl leading-[1.05] text-zinc-900">
              High-Density Object<br />
              <span className="italic text-blue-600">Segmentation</span>
            </motion.h1>
            <motion.p variants={fadeUp} className="mt-6 text-lg text-zinc-500 max-w-2xl mx-auto leading-relaxed">
              YOLACT + MobileNetV3 + CBAM Attention + Soft-NMS on SKU-110K
            </motion.p>
            <motion.p variants={fadeUp} className="mt-2 text-sm text-zinc-400">
              Raman Luhach (230107) &middot; Rachit Kumar (230128)
            </motion.p>
            <motion.div variants={fadeUp} className="mt-10 flex items-center justify-center gap-4">
              <Link href="/demo" className="inline-flex items-center gap-2 bg-zinc-900 text-white px-6 py-3 rounded-full font-medium text-sm hover:bg-zinc-800 transition-colors shadow-md">
                Try Live Demo <ArrowRight className="w-4 h-4" />
              </Link>
              <a href="#problem" className="inline-flex items-center gap-2 border border-zinc-300 px-6 py-3 rounded-full text-sm text-zinc-600 hover:border-zinc-400 transition-colors">
                View Presentation
              </a>
            </motion.div>
            <motion.div variants={fadeUp} className="mt-16 grid grid-cols-4 gap-8 max-w-lg mx-auto">
              {[
                ["~10M", "Parameters"],
                ["147", "Avg Obj/Image"],
                ["8.3", "FPS (ONNX)"],
                ["0.6MB", "Model Size"],
              ].map(([v, l]) => (
                <div key={l} className="text-center">
                  <p className="text-2xl md:text-3xl font-serif text-zinc-900">{v}</p>
                  <p className="text-xs text-zinc-400 mt-1">{l}</p>
                </div>
              ))}
            </motion.div>
          </motion.div>
        </div>
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.5 }} className="absolute bottom-8 left-1/2 -translate-x-1/2">
          <ChevronDown className="w-5 h-5 text-zinc-400 animate-bounce" />
        </motion.div>
      </section>

      {/* ── SLIDE 2: PROBLEM ── */}
      <section id="problem" className="slide-section bg-zinc-50/60">
        <div className="max-w-5xl mx-auto px-6">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={stagger}>
            <motion.div variants={fadeUp}>
              <p className="text-sm tracking-widest uppercase text-blue-600 mb-3 font-medium">The Challenge</p>
              <h2 className="font-serif text-4xl md:text-5xl text-zinc-900 mb-4">Why Dense Detection is Hard</h2>
              <p className="text-zinc-500 max-w-2xl mb-10">
                SKU-110K contains retail shelf images with 147 products per frame on average.
                Standard NMS aggressively removes overlapping detections, killing recall.
              </p>
            </motion.div>
            <motion.div variants={fadeUp} className="grid md:grid-cols-2 gap-6">
              <div className="border border-zinc-200 rounded-2xl p-6 bg-white shadow-sm space-y-3">
                <h3 className="font-semibold text-zinc-900">Dataset at a Glance</h3>
                {[
                  ["Total images", "11,762"],
                  ["Avg objects/image", "147.4"],
                  ["Total annotations", "1.73M"],
                  ["Train / Val / Test", "8,233 / 588 / 2,941"],
                ].map(([l, v]) => (
                  <div key={l} className="flex justify-between py-2 border-b border-zinc-100 last:border-0 text-sm">
                    <span className="text-zinc-500">{l}</span>
                    <span className="font-medium text-zinc-900 tabular-nums">{v}</span>
                  </div>
                ))}
              </div>
              <div className="border border-zinc-200 rounded-2xl overflow-hidden bg-white shadow-sm">
                <img src="/results/objects_per_image_histogram.png" alt="Object count distribution" className="w-full" />
                <p className="px-4 py-2 text-xs text-zinc-400 border-t border-zinc-100">Object count distribution &mdash; long tail shows images with 400+ products</p>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ── SLIDE 3: ARCHITECTURE ── */}
      <section className="slide-section">
        <div className="max-w-5xl mx-auto px-6">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={stagger}>
            <motion.div variants={fadeUp}>
              <p className="text-sm tracking-widest uppercase text-blue-600 mb-3 font-medium">Our Approach</p>
              <h2 className="font-serif text-4xl md:text-5xl text-zinc-900 mb-4">Architecture</h2>
              <p className="text-zinc-500 max-w-2xl mb-10">
                Lightweight YOLACT with attention-enhanced features and density-aware post-processing.
              </p>
            </motion.div>

            {/* Pipeline flow */}
            <motion.div variants={fadeUp} className="border border-zinc-200 rounded-2xl p-6 bg-zinc-50 shadow-sm mb-8">
              <div className="flex flex-wrap items-center justify-center gap-2 text-sm">
                {[
                  ["Input", "550x550"],
                  ["MobileNetV3", "5.4M params"],
                  ["FPN + CBAM", "Attention"],
                  ["ProtoNet", "32 masks"],
                  ["Head", "cls+box+coeff"],
                  ["Soft-NMS", "\u03C3=0.5"],
                  ["Output", "Detections"],
                ].map(([title, sub], i, arr) => (
                  <div key={title} className="flex items-center gap-2">
                    <div className="bg-white border border-zinc-200 rounded-xl px-4 py-2.5 text-center shadow-sm">
                      <p className="font-medium text-zinc-900 text-xs">{title}</p>
                      <p className="text-[10px] text-zinc-400">{sub}</p>
                    </div>
                    {i < arr.length - 1 && <ArrowRight className="w-3 h-3 text-blue-400 shrink-0" />}
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Key components */}
            <motion.div variants={fadeUp} className="grid md:grid-cols-2 gap-5">
              {[
                ["MobileNetV3-Large", "NAS-designed backbone with SE blocks and h-swish. 88% fewer params than ResNet-101. Features extracted at strides 8, 16, 32."],
                ["CBAM Attention", "Channel attention (what to focus on) + spatial attention (where to focus) applied on FPN levels P3-P5. Adds only ~10K params."],
                ["Focal Loss + Label Smoothing", "Focal loss (\u03B3=2) handles extreme class imbalance. Label smoothing (\u03B5=0.1) prevents overconfident predictions."],
                ["Soft-NMS + MixUp", "Gaussian score decay preserves valid overlapping detections. MixUp (\u03B1=0.2) creates virtual training examples for regularization."],
              ].map(([title, desc]) => (
                <div key={title} className="border border-zinc-200 rounded-2xl p-5 bg-white shadow-sm">
                  <h3 className="font-semibold text-zinc-900 mb-2 text-sm">{title}</h3>
                  <p className="text-sm text-zinc-500 leading-relaxed">{desc}</p>
                </div>
              ))}
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ── SLIDE 4: RESULTS ── */}
      <section className="slide-section bg-zinc-50/60">
        <div className="max-w-5xl mx-auto px-6">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={stagger}>
            <motion.div variants={fadeUp}>
              <p className="text-sm tracking-widest uppercase text-blue-600 mb-3 font-medium">Results</p>
              <h2 className="font-serif text-4xl md:text-5xl text-zinc-900 mb-4">Training &amp; Evaluation</h2>
              <p className="text-zinc-500 max-w-2xl mb-10">
                Trained on 3,000 images for 8 epochs. Loss reduced 49% with consistent convergence.
              </p>
            </motion.div>

            <motion.div variants={fadeUp} className="grid md:grid-cols-2 gap-6 mb-8">
              {/* Loss bars */}
              <div className="border border-zinc-200 rounded-2xl p-6 bg-white shadow-sm">
                <h3 className="font-semibold text-zinc-900 mb-1 text-sm">Loss Convergence</h3>
                <p className="text-xs text-zinc-400 mb-4">Train loss per epoch</p>
                <div className="space-y-2">
                  {[8.594,8.049,6.449,5.548,5.097,4.701,4.568,4.355].map((v, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <span className="text-[10px] text-zinc-400 w-5">E{i+1}</span>
                      <div className="flex-1 h-5 bg-zinc-100 rounded-full overflow-hidden">
                        <div className="h-full bg-blue-500 rounded-full flex items-center justify-end pr-1.5" style={{ width: `${(v/8.594)*100}%` }}>
                          <span className="text-[9px] text-white font-medium">{v.toFixed(1)}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-3 pt-3 border-t border-zinc-100 flex justify-between text-xs">
                  <span className="text-zinc-400">Val loss (final)</span>
                  <span className="text-blue-600 font-semibold">3.603</span>
                </div>
              </div>

              {/* Benchmarks */}
              <div className="border border-zinc-200 rounded-2xl p-6 bg-white shadow-sm">
                <h3 className="font-semibold text-zinc-900 mb-1 text-sm">Deployment Benchmarks</h3>
                <p className="text-xs text-zinc-400 mb-4">550x550 input, single image</p>
                {[
                  ["PyTorch FP32 (MPS)", "318ms / 3.1 FPS"],
                  ["ONNX FP32 (CPU)", "120ms / 8.3 FPS"],
                  ["ONNX FP32 size", "0.6 MB"],
                  ["ONNX INT8 size", "10.0 MB"],
                  ["Total parameters", "~10.0M"],
                ].map(([l, v]) => (
                  <div key={l} className="flex justify-between py-2.5 border-b border-zinc-100 last:border-0 text-sm">
                    <span className="text-zinc-500">{l}</span>
                    <span className="font-medium text-zinc-900 tabular-nums">{v}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Detection samples + PR curve */}
            <motion.div variants={fadeUp} className="grid md:grid-cols-2 gap-6">
              <div className="border border-zinc-200 rounded-2xl overflow-hidden bg-white shadow-sm">
                <img src="/results/detection_samples.png" alt="Detection samples" className="w-full" />
                <p className="px-4 py-2 text-xs text-zinc-400 border-t border-zinc-100">Sample detections on validation images</p>
              </div>
              <div className="border border-zinc-200 rounded-2xl overflow-hidden bg-white shadow-sm">
                <img src="/results/precision_recall.png" alt="PR curves" className="w-full" />
                <p className="px-4 py-2 text-xs text-zinc-400 border-t border-zinc-100">Precision-Recall curves at multiple IoU thresholds</p>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ── SLIDE 5: INTERPRETABILITY ── */}
      <section className="slide-section">
        <div className="max-w-5xl mx-auto px-6">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={stagger}>
            <motion.div variants={fadeUp}>
              <p className="text-sm tracking-widest uppercase text-blue-600 mb-3 font-medium">Validation</p>
              <h2 className="font-serif text-4xl md:text-5xl text-zinc-900 mb-4">Grad-CAM &amp; Robustness</h2>
              <p className="text-zinc-500 max-w-2xl mb-10">
                Interpretability analysis confirms the model learns product-relevant features.
                Robustness testing evaluates resilience under input corruptions.
              </p>
            </motion.div>

            <motion.div variants={fadeUp} className="border border-zinc-200 rounded-2xl overflow-hidden bg-white shadow-sm mb-8">
              <img src="/results/gradcam_grid.png" alt="Grad-CAM heatmaps" className="w-full" />
              <p className="px-4 py-3 text-xs text-zinc-400 border-t border-zinc-100">
                Grad-CAM heatmaps &mdash; warm colors show where the model focuses. It correctly attends to product regions, not shelf structure.
              </p>
            </motion.div>

            <motion.div variants={fadeUp} className="grid md:grid-cols-2 gap-6">
              <div className="border border-zinc-200 rounded-2xl overflow-hidden bg-white shadow-sm">
                <img src="/results/robustness_analysis.png" alt="Robustness analysis" className="w-full" />
                <p className="px-4 py-2 text-xs text-zinc-400 border-t border-zinc-100">AP under noise, blur, and brightness corruptions</p>
              </div>
              <div className="border border-zinc-200 rounded-2xl overflow-hidden bg-white shadow-sm">
                <img src="/results/density_analysis.png" alt="Density analysis" className="w-full" />
                <p className="px-4 py-2 text-xs text-zinc-400 border-t border-zinc-100">Performance by object density per image</p>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ── SLIDE 6: CTA ── */}
      <section className="slide-section bg-zinc-50/60">
        <div className="max-w-3xl mx-auto px-6 text-center">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={stagger}>
            <motion.h2 variants={fadeUp} className="font-serif text-4xl md:text-6xl text-zinc-900 mb-6">
              See it in <span className="italic text-blue-600">action</span>
            </motion.h2>
            <motion.p variants={fadeUp} className="text-zinc-500 text-lg mb-10">
              Upload a retail shelf image and watch the model detect every product.
            </motion.p>
            <motion.div variants={fadeUp}>
              <Link href="/demo" className="inline-flex items-center gap-3 bg-zinc-900 text-white px-8 py-4 rounded-full font-medium hover:bg-zinc-800 transition-colors text-lg shadow-lg">
                Launch Demo <ArrowRight className="w-5 h-5" />
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-zinc-200 py-6 bg-white">
        <div className="max-w-5xl mx-auto px-6 flex items-center justify-between text-sm text-zinc-400">
          <p>Raman Luhach (230107) &middot; Rachit Kumar (230128)</p>
          <a href="https://github.com/Raman-Luhach/AMLDLProject1" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1.5 hover:text-zinc-700 transition-colors">
            <ExternalLink className="w-3.5 h-3.5" /> GitHub
          </a>
        </div>
      </footer>
    </main>
  );
}
