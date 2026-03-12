import { useState, useEffect } from 'react'

const API_BASE = '/api'

const STAGES = [
  { id: 'extracting', label: 'Extracting frames', icon: '📂' },
  { id: 'swapping', label: 'Swapping faces', icon: '🔄' },
  { id: 'merging', label: 'Merging video', icon: '🎬' },
  { id: 'done', label: 'Complete', icon: '✓' },
]

export default function App() {
  const [source, setSource] = useState(null)
  const [target, setTarget] = useState(null)
  const [sourcePreview, setSourcePreview] = useState(null)
  const [targetPreview, setTargetPreview] = useState(null)
  const [enhance, setEnhance] = useState(false)
  const [swapModel, setSwapModel] = useState('inswapper')
  const [detSize, setDetSize] = useState(640)
  const [upscale, setUpscale] = useState(1)
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

  const reset = () => {
    setSource(null)
    setTarget(null)
    setSourcePreview(null)
    setTargetPreview(null)
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
    form.append('swap_model', swapModel)
    form.append('det_size', String(detSize))
    form.append('upscale', String(upscale))

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
      const a = document.createElement('a')
      a.href = `${API_BASE}/result/${jobId}`
      a.download = `ultrafaceswap_${jobId}.mp4`
      a.click()
    }
  }

  const getCurrentStageIndex = () => {
    const s = status?.stage || ''
    const i = STAGES.findIndex((st) => st.id === s)
    return i >= 0 ? i : (status?.status === 'completed' ? 3 : 0)
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
