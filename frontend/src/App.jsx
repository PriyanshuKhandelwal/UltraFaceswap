import { useState, useEffect } from 'react'

const API_BASE = '/api'

const STAGES = [
  { id: 'extracting', label: 'Extracting frames', icon: '📂' },
  { id: 'swapping', label: 'Swapping faces', icon: '🔄' },
  { id: 'interpolating', label: 'Motion smoothing', icon: '✨' },
  { id: 'merging', label: 'Merging video', icon: '🎬' },
  { id: 'done', label: 'Complete', icon: '✓' },
]

export default function App() {
  const [source, setSource] = useState(null)
  const [target, setTarget] = useState(null)
  const [sourcePreview, setSourcePreview] = useState(null)
  const [targetPreview, setTargetPreview] = useState(null)
  const [enhance, setEnhance] = useState(false)
  const [hairMatch, setHairMatch] = useState(true)
  const [swapModel, setSwapModel] = useState('inswapper')
  const [detSize, setDetSize] = useState(640)
  const [upscale, setUpscale] = useState(1)
  const [interpolate, setInterpolate] = useState(1)
  const [suggestion, setSuggestion] = useState(null)
  const [suggestLoading, setSuggestLoading] = useState(false)
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)
  const [polling, setPolling] = useState(false)

  useEffect(() => {
    if (source) {
      const url = URL.createObjectURL(source)
      setSourcePreview(url)
      return () => URL.revokeObjectURL(url)
    }
    setSourcePreview(null)
  }, [source])

  useEffect(() => {
    if (target) {
      const url = URL.createObjectURL(target)
      setTargetPreview(url)
      return () => URL.revokeObjectURL(url)
    }
    setTargetPreview(null)
  }, [target])

  const fetchSuggestions = async () => {
    if (!source || !target) return
    setSuggestLoading(true)
    setSuggestion(null)
    try {
      const form = new FormData()
      form.append('source', source)
      form.append('target', target)
      const res = await fetch(`${API_BASE}/suggest`, {
        method: 'POST',
        body: form,
      })
      const data = await res.json()
      if (res.ok) setSuggestion(data)
    } catch {
      setSuggestion(null)
    } finally {
      setSuggestLoading(false)
    }
  }

  const applySuggestion = () => {
    if (!suggestion) return
    setSwapModel(suggestion.swap_model)
    setDetSize(suggestion.det_size)
    setUpscale(suggestion.upscale)
    setEnhance(suggestion.enhance)
    setInterpolate(suggestion.interpolate || 1)
  }

  const reset = () => {
    setSource(null)
    setTarget(null)
    setSourcePreview(null)
    setTargetPreview(null)
    setSuggestion(null)
    setJobId(null)
    setStatus(null)
    setError(null)
    setPolling(false)
  }

  const pollStatus = async (id) => {
    setPolling(true)
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/status/${id}`)
        const data = await res.json()
        setStatus(data)
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(interval)
          setPolling(false)
        }
      } catch (e) {
        clearInterval(interval)
        setPolling(false)
        setError('Failed to poll status')
      }
    }, 1000)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    if (!source || !target) {
      setError('Please upload both source photo and target video')
      return
    }

    const form = new FormData()
    form.append('source', source)
    form.append('target', target)
    form.append('enhance', enhance)
    form.append('hair_match', hairMatch)
    form.append('swap_model', swapModel)
    form.append('det_size', String(detSize))
    form.append('upscale', String(upscale))
    form.append('interpolate', String(interpolate))

    try {
      const res = await fetch(`${API_BASE}/swap`, {
        method: 'POST',
        body: form,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      setJobId(data.job_id)
      pollStatus(data.job_id)
    } catch (e) {
      setError(e.message)
    }
  }

  const downloadResult = () => {
    if (jobId) {
      const s = status?.settings || {}
      const suffix = s.swap_model && s.det_size != null
        ? `${s.swap_model}_d${s.det_size}_u${s.upscale || 1}_i${s.interpolate || 1}_enh${s.enhance ? 1 : 0}_hair${s.hair_match ? 1 : 0}`
        : jobId
      const a = document.createElement('a')
      a.href = `${API_BASE}/result/${jobId}`
      a.download = `ultrafaceswap_${suffix}.mp4`
      a.click()
    }
  }

  const getCurrentStageIndex = () => {
    const s = status?.stage || ''
    const i = STAGES.findIndex((st) => st.id === s)
    return i >= 0 ? i : (status?.status === 'completed' ? STAGES.length - 1 : 0)
  }

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>UltraFaceswap</h1>
        <p style={styles.subtitle}>Swap a face from a photo onto a video</p>
      </header>

      <main style={styles.main}>
        {!jobId ? (
          <form onSubmit={handleSubmit} style={styles.form}>
            <div style={styles.previewRow}>
              <div style={styles.uploadGroup}>
                <label style={styles.label}>Source face (photo)</label>
                <div style={styles.dropZone}>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => setSource(e.target.files[0])}
                    style={styles.hiddenInput}
                  />
                  {sourcePreview ? (
                    <img src={sourcePreview} alt="Source" style={styles.previewImg} />
                  ) : (
                    <span style={styles.placeholder}>+ Add photo</span>
                  )}
                </div>
                {source && <span style={styles.fileName}>{source.name}</span>}
              </div>

              <div style={styles.uploadGroup}>
                <label style={styles.label}>Target video</label>
                <div style={styles.dropZone}>
                  <input
                    type="file"
                    accept="video/*"
                    onChange={(e) => setTarget(e.target.files[0])}
                    style={styles.hiddenInput}
                  />
                  {targetPreview ? (
                    <video
                      src={targetPreview}
                      style={styles.previewVideo}
                      muted
                      loop
                      playsInline
                      onLoadedMetadata={(e) => e.target.play().catch(() => {})}
                    />
                  ) : (
                    <span style={styles.placeholder}>+ Add video</span>
                  )}
                </div>
                {target && <span style={styles.fileName}>{target.name}</span>}
              </div>
            </div>

            {source && target && (
              <div style={styles.suggestionBox}>
                <button
                  type="button"
                  onClick={fetchSuggestions}
                  disabled={suggestLoading}
                  style={styles.suggestBtn}
                >
                  {suggestLoading ? 'Analyzing...' : 'Get suggested settings'}
                </button>
                {suggestion && (
                  <div style={styles.suggestionContent}>
                    <p style={styles.suggestionMeta}>
                      Video: {suggestion.meta?.video_width || '?'}×{suggestion.meta?.video_height || '?'} @ {suggestion.meta?.video_fps || '?'} fps
                      {suggestion.meta?.video_frames ? ` · ${suggestion.meta.video_frames} frames` : ''}
                    </p>
                    <p style={styles.suggestionMeta}>
                      Photo: {suggestion.meta?.image_width || '?'}×{suggestion.meta?.image_height || '?'}
                    </p>
                    <p style={styles.suggestionRec}>
                      Suggested: {suggestion.swap_model === 'simswap' ? 'SimSwap' : 'InSwapper'} · det {suggestion.det_size} · {suggestion.upscale}× upscale · {suggestion.enhance ? 'GFPGAN on' : 'off'} · {suggestion.interpolate > 1 ? `${suggestion.interpolate}× smoother` : 'no interpolation'}
                    </p>
                    <button type="button" onClick={applySuggestion} style={styles.applyBtn}>
                      Apply suggested
                    </button>
                  </div>
                )}
              </div>
            )}

            <div style={styles.qualitySection}>
              <h3 style={styles.qualityTitle}>Quality options</h3>

              <div style={styles.option}>
                <label style={styles.optionLabel}>Face swap model</label>
                <select
                  value={swapModel}
                  onChange={(e) => setSwapModel(e.target.value)}
                  style={styles.select}
                >
                  <option value="inswapper">InSwapper 128 (faster)</option>
                  <option value="simswap">SimSwap 256 (sharper faces)</option>
                </select>
                <p style={styles.optionDesc}>
                  InSwapper: quicker, good for most cases. SimSwap: sharper facial details, better for HD video.
                </p>
              </div>

              <div style={styles.option}>
                <label style={styles.optionLabel}>Face detection precision</label>
                <select
                  value={detSize}
                  onChange={(e) => setDetSize(Number(e.target.value))}
                  style={styles.select}
                >
                  <option value={320}>320 (faster)</option>
                  <option value={640}>640 (better for HD)</option>
                </select>
                <p style={styles.optionDesc}>
                  Higher values help with HD video and small faces. Use 640 for 1080p and above.
                </p>
              </div>

              <div style={styles.option}>
                <label style={styles.optionLabel}>Output resolution</label>
                <select
                  value={upscale}
                  onChange={(e) => setUpscale(Number(e.target.value))}
                  style={styles.select}
                >
                  <option value={1}>1× (original)</option>
                  <option value={2}>2× (double size)</option>
                  <option value={4}>4× (four times sharper)</option>
                </select>
                <p style={styles.optionDesc}>
                  AI upscaling makes the video sharper. 2× or 4× improve face clarity but take longer.
                </p>
              </div>

              <div style={styles.option}>
                <label style={styles.optionLabel}>Motion smoothing</label>
                <select
                  value={interpolate}
                  onChange={(e) => setInterpolate(Number(e.target.value))}
                  style={styles.select}
                >
                  <option value={1}>1× (original)</option>
                  <option value={2}>2× (smoother)</option>
                  <option value={4}>4× (very smooth)</option>
                </select>
                <p style={styles.optionDesc}>
                  Inserts extra frames between existing ones for smoother playback and less flicker.
                </p>
              </div>

              <label style={styles.checkbox}>
                <input
                  type="checkbox"
                  checked={hairMatch}
                  onChange={(e) => setHairMatch(e.target.checked)}
                />
                Hair color matching
              </label>
              <p style={styles.optionDesc}>
                Transfer hair color from source to swapped face for more consistent look.
              </p>

              <label style={styles.checkbox}>
                <input
                  type="checkbox"
                  checked={enhance}
                  onChange={(e) => setEnhance(e.target.checked)}
                />
                Face restoration (GFPGAN)
              </label>
              <p style={styles.optionDesc}>
                Cleans up blur and improves skin texture on swapped faces.
              </p>
            </div>

            {error && <p style={styles.error}>{error}</p>}

            <button type="submit" style={styles.button} disabled={!source || !target}>
              Start face swap
            </button>
          </form>
        ) : (
          <div style={styles.statusCard}>
            <h2 style={styles.statusTitle}>
              {status?.status === 'completed' && 'Done!'}
              {status?.status === 'failed' && 'Failed'}
              {(status?.status === 'pending' || status?.status === 'processing') &&
                (status?.stage === 'extracting'
                  ? 'Extracting frames...'
                  : status?.stage === 'swapping'
                  ? `Swapping faces (${status?.processed_frames || 0}/${status?.total_frames || 0})`
                  : status?.stage === 'interpolating'
                  ? 'Motion smoothing...'
                  : status?.stage === 'merging'
                  ? 'Merging video...'
                  : 'Processing...')}
            </h2>

            {(status?.status === 'processing' || status?.status === 'pending') && (
              <div style={styles.steps}>
                {STAGES.map((stage, i) => {
                  const current = getCurrentStageIndex()
                  const done = i < current || (status?.status === 'completed' && i <= current)
                  const active = i === current && status?.status !== 'completed'
                  return (
                    <div
                      key={stage.id}
                      style={{
                        ...styles.step,
                        ...(active ? styles.stepActive : {}),
                        ...(done ? styles.stepDone : {}),
                      }}
                    >
                      <span style={styles.stepIcon}>{stage.icon}</span>
                      <span style={styles.stepLabel}>{stage.label}</span>
                      {active && i === 1 && status?.total_frames > 0 && (
                        <span style={styles.stepDetail}>
                          {status.processed_frames || 0} / {status.total_frames} frames
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>
            )}

            {(status?.status === 'processing' || status?.status === 'pending') && (
              <div style={styles.progress}>
                <div style={styles.progressBar}>
                  <div
                    style={{
                      ...styles.progressFill,
                      width: `${status?.progress || 0}%`,
                    }}
                  />
                </div>
                <span style={styles.progressText}>{status?.progress || 0}%</span>
              </div>
            )}

            {status?.error && <p style={styles.error}>{status.error}</p>}

            {status?.status === 'completed' && (
              <>
                {status?.settings && (
                  <p style={styles.settingsUsed}>
                    Settings: {status.settings.swap_model} · det {status.settings.det_size} · {status.settings.upscale}× upscale · {status.settings.interpolate}× smoother · enhance {status.settings.enhance ? 'on' : 'off'} · hair {status.settings.hair_match ? 'on' : 'off'}
                  </p>
                )}
                <div style={styles.resultPreview}>
                  <video
                    src={`${API_BASE}/result/${jobId}`}
                    controls
                    style={styles.resultVideo}
                    poster=""
                  />
                  <p style={styles.resultHint}>Preview your result below</p>
                </div>
                <button onClick={downloadResult} style={styles.button}>
                  Download result
                </button>
              </>
            )}

            <button onClick={reset} style={styles.secondaryButton}>
              Start over
            </button>
          </div>
        )}
      </main>

      <footer style={styles.footer}>
        <p>
          Powered by InsightFace InSwapper ·{' '}
          <a href="/docs" style={styles.link}>
            API docs
          </a>
        </p>
      </footer>
    </div>
  )
}

const styles = {
  container: {
    maxWidth: 640,
    margin: '0 auto',
    padding: '2rem 1rem',
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    textAlign: 'center',
    marginBottom: '2rem',
  },
  title: {
    fontSize: '1.875rem',
    fontWeight: 700,
    margin: 0,
    background: 'linear-gradient(135deg, #22d3ee, #06b6d4)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  },
  subtitle: {
    color: 'var(--text-muted)',
    marginTop: '0.5rem',
  },
  main: {
    flex: 1,
  },
  form: {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 12,
    padding: '1.75rem',
  },
  previewRow: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '1rem',
    marginBottom: '1.25rem',
  },
  uploadGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.35rem',
  },
  label: {
    fontSize: '0.9rem',
    color: 'var(--text-muted)',
  },
  dropZone: {
    position: 'relative',
    aspectRatio: '1',
    background: 'var(--bg)',
    border: '2px dashed var(--border)',
    borderRadius: 8,
    overflow: 'hidden',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  hiddenInput: {
    position: 'absolute',
    inset: 0,
    opacity: 0,
    cursor: 'pointer',
    width: '100%',
  },
  previewImg: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
  previewVideo: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
  placeholder: {
    color: 'var(--text-muted)',
    fontSize: '0.9rem',
  },
  fileName: {
    fontSize: '0.8rem',
    color: 'var(--accent)',
  },
  suggestionBox: {
    marginBottom: '1rem',
    padding: '1rem',
    background: 'linear-gradient(135deg, rgba(34, 211, 238, 0.08), rgba(6, 182, 212, 0.05))',
    borderRadius: 8,
    border: '1px solid var(--border)',
  },
  suggestBtn: {
    padding: '0.5rem 1rem',
    background: 'transparent',
    color: 'var(--accent)',
    border: '1px solid var(--accent)',
    borderRadius: 6,
    cursor: 'pointer',
    fontSize: '0.9rem',
  },
  suggestionContent: {
    marginTop: '0.75rem',
  },
  suggestionMeta: {
    margin: '0.25rem 0',
    fontSize: '0.8rem',
    color: 'var(--text-muted)',
  },
  suggestionRec: {
    margin: '0.5rem 0',
    fontSize: '0.85rem',
    color: 'var(--text)',
  },
  applyBtn: {
    marginTop: '0.5rem',
    padding: '0.4rem 0.75rem',
    background: 'var(--accent)',
    color: 'var(--bg)',
    border: 'none',
    borderRadius: 6,
    cursor: 'pointer',
    fontSize: '0.85rem',
  },
  qualitySection: {
    marginBottom: '1rem',
    padding: '1rem',
    background: 'var(--bg)',
    borderRadius: 8,
    border: '1px solid var(--border)',
  },
  qualityTitle: {
    margin: '0 0 1rem 0',
    fontSize: '0.95rem',
    color: 'var(--text)',
  },
  option: {
    marginBottom: '1rem',
  },
  optionLabel: {
    display: 'block',
    fontSize: '0.9rem',
    marginBottom: '0.35rem',
    color: 'var(--text-muted)',
  },
  select: {
    width: '100%',
    padding: '0.5rem',
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 6,
    color: 'var(--text)',
    fontSize: '0.9rem',
  },
  optionDesc: {
    margin: '0.35rem 0 0 0',
    fontSize: '0.8rem',
    color: 'var(--text-muted)',
    lineHeight: 1.4,
  },
  checkbox: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    marginBottom: '0.35rem',
    marginTop: '0.5rem',
    cursor: 'pointer',
    fontSize: '0.9rem',
  },
  error: {
    color: 'var(--error)',
    marginBottom: '1rem',
  },
  button: {
    width: '100%',
    padding: '0.875rem',
    background: 'var(--accent)',
    color: 'var(--bg)',
    border: 'none',
    borderRadius: 8,
    fontSize: '1rem',
    fontWeight: 600,
    cursor: 'pointer',
  },
  secondaryButton: {
    width: '100%',
    padding: '0.75rem',
    marginTop: '0.75rem',
    background: 'transparent',
    color: 'var(--text-muted)',
    border: '1px solid var(--border)',
    borderRadius: 8,
    cursor: 'pointer',
  },
  statusCard: {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 12,
    padding: '1.75rem',
  },
  statusTitle: {
    marginTop: 0,
    marginBottom: '1.25rem',
    fontSize: '1.15rem',
  },
  steps: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
    marginBottom: '1.25rem',
  },
  step: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.6rem',
    padding: '0.5rem 0.75rem',
    borderRadius: 8,
    background: 'var(--bg)',
    opacity: 0.6,
    transition: 'all 0.2s',
  },
  stepActive: {
    opacity: 1,
    borderLeft: '3px solid var(--accent)',
  },
  stepDone: {
    opacity: 0.9,
  },
  stepIcon: {
    fontSize: '1.1rem',
  },
  stepLabel: {
    flex: 1,
    fontSize: '0.9rem',
  },
  stepDetail: {
    fontSize: '0.8rem',
    color: 'var(--text-muted)',
  },
  progress: {
    marginBottom: '1rem',
  },
  progressBar: {
    height: 8,
    background: 'var(--border)',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    background: 'linear-gradient(90deg, var(--accent), #06b6d4)',
    transition: 'width 0.3s ease',
  },
  progressText: {
    display: 'block',
    marginTop: '0.5rem',
    fontSize: '0.85rem',
    color: 'var(--text-muted)',
  },
  resultPreview: {
    marginBottom: '1rem',
  },
  resultVideo: {
    width: '100%',
    borderRadius: 8,
    background: '#000',
  },
  settingsUsed: {
    fontSize: '0.85rem',
    color: 'var(--text-muted)',
    marginBottom: '0.75rem',
    padding: '0.5rem 0.75rem',
    background: 'var(--bg)',
    borderRadius: 6,
  },
  resultHint: {
    fontSize: '0.8rem',
    color: 'var(--text-muted)',
    marginTop: '0.5rem',
  },
  footer: {
    marginTop: '2rem',
    textAlign: 'center',
    fontSize: '0.85rem',
    color: 'var(--text-muted)',
  },
  link: {
    color: 'var(--accent)',
  },
}
