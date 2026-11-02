import { useState, useRef, useEffect } from 'react'
import {
  detectAudio,
  getModels,
  MODEL_LABELS,
  type DetectionResult,
  type Explanation,
  type ModelInfo,
} from '../services/detectionService'
import { generateReport } from '../services/forensicsService'
import { addHistory } from '../services/history'
import { ApiError } from '../config/apiConfig'

const EMPTY_AUDIO_SEGMENTS: { start_s: number; end_s: number; importance: number }[] = []

function ConfidenceGauge({ confidence, label }: { confidence: number; label: string }) {
  const pct = Math.round(confidence * 100)
  const color = label === 'fake' ? 'text-red-400' : 'text-green-400'
  const ring = label === 'fake' ? 'border-red-500' : 'border-green-500'
  return (
    <div className={`flex flex-col items-center gap-2 p-6 rounded-xl border-2 ${ring} bg-gray-800`}>
      <span className={`text-5xl font-bold font-mono ${color}`}>{pct}%</span>
      <span className={`text-lg font-semibold uppercase tracking-widest ${color}`}>{label}</span>
      <span className="text-xs text-gray-500">confidence</span>
    </div>
  )
}

function AttributionView({
  explanation,
  label,
}: {
  explanation: Explanation
  label: string
}) {
  const { attribution_frames: frames, top_segments: segments, method, frame_duration_ms } = explanation
  const isFake = label === 'fake'
  const barColor = isFake ? 'bg-rose-500' : 'bg-emerald-500'
  const accentText = isFake ? 'text-rose-300' : 'text-emerald-300'

  // Total clip duration (frames × frame length) → for the time axis.
  const totalS = (frames.length * (frame_duration_ms || 10)) / 1000
  const methodLabel =
    method === 'integrated_gradients' ? 'Integrated Gradients' : method === 'occlusion' ? 'Occlusion' : method

  // Downsample to ~140 bars for a clean timeline of per-frame importance.
  const step = Math.max(1, Math.ceil(frames.length / 140))
  const bars: number[] = []
  for (let i = 0; i < frames.length; i += step) bars.push(Math.max(...frames.slice(i, i + step)))

  const top = [...segments].sort((a, b) => b.importance - a.importance).slice(0, 5)
  const strongest = top[0]

  return (
    <div className="bg-gray-800 rounded-xl p-5 border border-gray-700 space-y-4">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <p className="text-sm font-medium text-gray-200">Why this verdict — moment-by-moment</p>
        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-gray-900 border border-gray-700 text-gray-400">
          {methodLabel} · {frame_duration_ms || 10}ms frames
        </span>
      </div>

      {/* LLM forensic narrative */}
      {explanation.narrative && (
        <div className="rounded-lg border border-indigo-500/30 bg-indigo-500/5 p-3">
          <p className="text-[11px] uppercase tracking-wider text-indigo-300/80 mb-1 flex items-center gap-1.5">
            <span aria-hidden>✨</span> AI analysis
          </p>
          <p className="text-sm text-gray-200 leading-relaxed">{explanation.narrative}</p>
        </div>
      )}

      {/* plain-language interpretation */}
      {strongest && (
        <p className="text-sm text-gray-400">
          The detector weighed the whole clip and leaned{' '}
          <span className={`font-semibold ${accentText}`}>{label.toUpperCase()}</span> most strongly on the audio
          around{' '}
          <span className="font-mono text-gray-200">
            {strongest.start_s.toFixed(1)}s–{strongest.end_s.toFixed(1)}s
          </span>
          {top.length > 1 && (
            <>
              , with further support at{' '}
              <span className="font-mono text-gray-200">
                {top[1].start_s.toFixed(1)}s–{top[1].end_s.toFixed(1)}s
              </span>
            </>
          )}
          .
        </p>
      )}

      {/* timeline */}
      <div>
        <div className="flex items-end gap-px h-20">
          {bars.map((v, i) => (
            <div
              key={i}
              className={`flex-1 rounded-sm ${barColor}`}
              style={{ height: `${Math.max(3, v * 100)}%`, opacity: 0.35 + v * 0.65 }}
              title={`importance ${v.toFixed(2)}`}
            />
          ))}
        </div>
        <div className="flex justify-between text-[10px] font-mono text-gray-600 mt-1">
          <span>0.0s</span>
          <span>{(totalS / 2).toFixed(1)}s</span>
          <span>{totalS.toFixed(1)}s</span>
        </div>
      </div>

      {/* ranked suspicious moments with importance bars */}
      {top.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-[11px] uppercase tracking-wider text-gray-500">
            Most influential moments
          </p>
          {top.map((s, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="text-xs font-mono text-gray-500 w-4 text-right">{i + 1}</span>
              <span className="text-xs font-mono text-gray-300 w-24 shrink-0">
                {s.start_s.toFixed(1)}–{s.end_s.toFixed(1)}s
              </span>
              <div className="flex-1 h-2 rounded-full bg-gray-900 overflow-hidden">
                <div
                  className={`h-full ${barColor}`}
                  style={{ width: `${Math.max(4, s.importance * 100)}%` }}
                />
              </div>
              <span className={`text-xs font-mono ${accentText} w-10 text-right`}>
                {Math.round(s.importance * 100)}%
              </span>
            </div>
          ))}
        </div>
      )}

      <p className="text-[11px] text-gray-600">
        {methodLabel === 'Integrated Gradients'
          ? 'Integrated Gradients traces the verdict back through the model to each 10ms frame — taller/brighter and higher-ranked = more influence.'
          : 'Occlusion silences each time window in turn and measures how much the verdict shifts — bigger shift = more influence.'}
      </p>
    </div>
  )
}

function AudioWaveform({
  file,
  segments = EMPTY_AUDIO_SEGMENTS,
}: {
  file: File
  segments?: { start_s: number; end_s: number; importance: number }[]
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let cancelled = false
    let audioContext: AudioContext | null = null

    const drawWaveform = async () => {
      try {
        audioContext = new AudioContext()
        const buffer = await file.arrayBuffer()
        const audio = await audioContext.decodeAudioData(buffer)

        if (cancelled || !canvasRef.current) return

        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        const data = audio.getChannelData(0)
        const width = canvas.width
        const height = canvas.height
        const step = Math.ceil(data.length / width)

        ctx.clearRect(0, 0, width, height)
        // Highlight suspicious audio time segments
        ctx.fillStyle = 'rgba(239, 68, 68, 0.25)'

        for (const segment of segments) {
          const startX = Math.max(
            0,
            (segment.start_s / audio.duration) * width
          )

          const endX = Math.min(
            width,
            (segment.end_s / audio.duration) * width
          )

          if (endX > startX) {
            ctx.fillRect(startX, 0, endX - startX, height)
          }
        }
        ctx.fillStyle = '#22d3ee'

        for (let x = 0; x < width; x++) {
          let peak = 0

          for (let i = 0; i < step; i++) {
            peak = Math.max(peak, Math.abs(data[x * step + i] ?? 0))
          }

          const barHeight = Math.max(2, peak * height)
          ctx.fillRect(x, (height - barHeight) / 2, 1, barHeight)
        }
      } catch (error) {
        console.error('Waveform generation failed:', error)
      } finally {
        if (audioContext && audioContext.state !== 'closed') {
          await audioContext.close()
        }
      }
    }

    drawWaveform()

    return () => {
      cancelled = true
      if (audioContext && audioContext.state !== 'closed') {
        void audioContext.close()
      }
    }
  }, [file, segments])

  return (
    <div className="bg-gray-900 border border-cyan-500/20 rounded-xl p-4">
      <p className="text-sm text-cyan-300 mb-3 font-medium">
        Audio Waveform
      </p>
      <canvas
        ref={canvasRef}
        width={1000}
        height={160}
        className="w-full h-32"
      />
    </div>
  )
}
function AudioSpectrogram({ file }: { file: File }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let cancelled = false
    let audioContext: AudioContext | null = null

    const draw = async () => {
      try {
        audioContext = new AudioContext()
        const buffer = await file.arrayBuffer()
        const audio = await audioContext.decodeAudioData(buffer)

        if (cancelled || !canvasRef.current) return

        const data = audio.getChannelData(0)
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        const width = canvas.width
        const height = canvas.height
        const size = 1024
        const hop = 512
        const half = size / 2

        const real = new Float64Array(size)
        const imag = new Float64Array(size)
        const hann = new Float64Array(size)

        for (let i = 0; i < size; i++) {
          hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)))
        }

        const image = ctx.createImageData(width, height)
        const frames = Math.max(1, Math.floor((data.length - size) / hop) + 1)

        for (let x = 0; x < width; x++) {
          if (cancelled) return

          const frame = Math.min(frames - 1, Math.floor((x / width) * frames))
          const start = frame * hop

          real.fill(0)
          imag.fill(0)

          for (let i = 0; i < size; i++) {
            real[i] = (data[start + i] ?? 0) * hann[i]
          }

          // Bit-reversal permutation
          let j = 0
          for (let i = 1; i < size; i++) {
            let bit = size >> 1
            while (j & bit) {
              j ^= bit
              bit >>= 1
            }
            j ^= bit

            if (i < j) {
              ;[real[i], real[j]] = [real[j], real[i]]
            }
          }

          // Radix-2 FFT
          for (let len = 2; len <= size; len <<= 1) {
            const angle = (-2 * Math.PI) / len
            const wr = Math.cos(angle)
            const wi = Math.sin(angle)

            for (let i = 0; i < size; i += len) {
              let ur = 1
              let ui = 0

              for (let k = 0; k < len / 2; k++) {
                const even = i + k
                const odd = even + len / 2

                const tr = ur * real[odd] - ui * imag[odd]
                const ti = ur * imag[odd] + ui * real[odd]

                real[odd] = real[even] - tr
                imag[odd] = imag[even] - ti
                real[even] += tr
                imag[even] += ti

                const nextUr = ur * wr - ui * wi
                ui = ur * wi + ui * wr
                ur = nextUr
              }
            }
          }

          for (let y = 0; y < height; y++) {
            const bin = Math.min(half - 1, Math.max(1,
              Math.floor((1 - y / height) * half)
            ))

            const magnitude = Math.sqrt(
              real[bin] ** 2 + imag[bin] ** 2
            ) / size

            const intensity = Math.max(
              0,
              Math.min(255, (20 * Math.log10(magnitude + 1e-7) + 100) * 3)
            )

            const p = (y * width + x) * 4
            image.data[p] = Math.min(255, intensity * 0.7)
            image.data[p + 1] = Math.min(255, intensity * 0.35)
            image.data[p + 2] = Math.min(255, 80 + intensity * 0.8)
            image.data[p + 3] = 255
          }
        }

        if (!cancelled) ctx.putImageData(image, 0, 0)
      } catch (error) {
        console.error('Spectrogram generation failed:', error)
      } finally {
        if (audioContext && audioContext.state !== 'closed') {
          await audioContext.close()
        }
      }
    }

    void draw()

    return () => {
      cancelled = true
      if (audioContext && audioContext.state !== 'closed') {
        void audioContext.close()
      }
    }
  }, [file])

  return (
    <div className="bg-gray-900 border border-purple-500/20 rounded-xl p-4">
      <p className="text-sm text-purple-300 mb-3 font-medium">
        Audio Spectrogram
      </p>
      <canvas
        ref={canvasRef}
        width={1000}
        height={240}
        className="w-full h-48"
      />
      <p className="text-xs text-gray-500 mt-2">
        Frequency intensity over time
      </p>
    </div>
  )
}
export default function DetectTab() {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [explain, setExplain] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [reportBusy, setReportBusy] = useState(false)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [model, setModel] = useState('classical')
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    getModels()
      .then((ms) => {
        // Show the flagship models first; available ones bubble up.
        const order = Object.keys(MODEL_LABELS)
        ms.sort(
          (a, b) =>
            Number(b.available) - Number(a.available) ||
            order.indexOf(a.key) - order.indexOf(b.key),
        )
        setModels(ms)
        const first = ms.find((m) => m.available)
        if (first) setModel(first.key)
      })
      .catch(() => setModels([]))
  }, [])

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await detectAudio(file, explain, model)
      setResult(data)
      addHistory(file.name, data)
    } catch (e) {
      if (e instanceof ApiError) {
        setError(
          e.status === 401
            ? 'Not authenticated — use the Log in button (top-right) first.'
            : e.message,
        )
      } else {
        setError('Could not reach the detection API. Please try again in a moment.')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleReport = async () => {
    if (!result) return
    setReportBusy(true)
    try {
      const report = await generateReport(result)
      window.open(report.report_url, '_blank')
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not generate the report.')
    } finally {
      setReportBusy(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-white mb-1">Upload Audio for Analysis</h2>
        <p className="text-sm text-gray-400">
          Supported formats: WAV, MP3, FLAC · Max 100MB · Audio auto-deleted after 60s (PDPL)
        </p>
      </div>

      {/* Upload area */}
      <div
        className="border-2 border-dashed border-gray-700 rounded-xl p-10 text-center cursor-pointer hover:border-indigo-500 transition-colors"
        onClick={() => fileRef.current?.click()}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".wav,.mp3,.flac,.ogg,audio/*"
          className="hidden"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        {file ? (
          <div className="space-y-1">
            <p className="text-white font-medium">{file.name}</p>
            <p className="text-gray-400 text-sm">{(file.size / 1024).toFixed(1)} KB</p>
          </div>
        ) : (
          <div className="space-y-2">
            <div className="text-4xl">🎙️</div>
            <p className="text-gray-300">Click to select audio file</p>
            <p className="text-gray-500 text-sm">or drag and drop · use a few seconds of speech</p>
          </div>
        )}
      </div>

      {file && (
  <AudioWaveform
    file={file}
    segments={result?.explanation?.top_segments ?? []}
  />
)}
      {file && <AudioSpectrogram file={file} />}
      {/* Model picker */}
      <div className="space-y-1">
        <label className="text-sm text-gray-400">Detector model</label>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="w-full bg-gray-800 border border-gray-700 text-gray-200 rounded-lg px-3 py-2 text-sm"
        >
          {models.map((m) => (
            <option key={m.key} value={m.key} disabled={!m.available}>
              {(MODEL_LABELS[m.key] ?? m.key) + (m.available ? '' : ' — checkpoint unavailable')}
            </option>
          ))}
        </select>
      </div>

      {/* Options + submit */}
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2 text-sm text-gray-400 select-none cursor-pointer">
          <input
            type="checkbox"
            checked={explain}
            onChange={(e) => setExplain(e.target.checked)}
            className="accent-indigo-500"
          />
          Explain decision
        </label>
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="flex-1 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-lg transition-colors"
        >
          {loading ? 'Analysing…' : 'Detect Deepfake'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded-lg p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-4">
          <ConfidenceGauge confidence={result.confidence} label={result.label} />
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {[
              { k: 'Model', v: result.model },
              { k: 'Latency', v: `${result.latency_ms.toFixed(1)}ms` },
              { k: 'Audio Hash', v: result.audio_hash.slice(0, 8) + '…' },
            ].map(({ k, v }) => (
              <div key={k} className="bg-gray-800 rounded-lg p-3">
                <p className="text-xs text-gray-500">{k}</p>
                <p className="text-sm text-white font-mono truncate">{v}</p>
              </div>
            ))}
          </div>

          {/* Verdict explainer */}
          <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
            <p className="text-sm font-medium text-gray-300 mb-1">
              {result.label === 'fake'
                ? 'This audio shows synthetic-speech artifacts.'
                : 'This audio is consistent with genuine human speech.'}
            </p>
            <p className="text-xs text-gray-500">
              Scored by the {result.model} detector (XLS-R self-supervised front-end + AASIST
              graph-attention back-end). Uploaded audio is deleted within 60 seconds (PDPL).
            </p>
          </div>

          {/* Explainability */}
          {result.explanation && (
            <AttributionView
              explanation={result.explanation}
              label={result.label}
            />
          )}

          {/* Forensic report */}
          <button
            onClick={handleReport}
            disabled={reportBusy}
            className="w-full border border-gray-700 bg-gray-800 hover:bg-gray-700 disabled:opacity-50 text-gray-200 font-medium py-2.5 px-6 rounded-lg transition-colors text-sm"
          >
            {reportBusy ? 'Generating…' : '⬇ Download forensic PDF report'}
          </button>
        </div>
      )}
    </div>
  )
}
