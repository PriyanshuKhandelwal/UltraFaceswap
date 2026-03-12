import { useState } from 'react'

const API_BASE = '/api'

export default function App() {
  const [source, setSource] = useState(null)
  const [target, setTarget] = useState(null)
  const [enhance, setEnhance] = useState(false)
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)
  const [polling, setPolling] = useState(false)

  const reset = () => {
    setSource(null)
    setTarget(null)
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
    }, 1500)
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

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>UltraFaceswap</h1>
        <p style={styles.subtitle}>Swap a face from a photo onto a video</p>
      </header>

      <main style={styles.main}>
        {!jobId ? (
          <form onSubmit={handleSubmit} style={styles.form}>
            <div style={styles.uploadGroup}>
              <label style={styles.label}>Source face (photo)</label>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setSource(e.target.files[0])}
                style={styles.input}
              />
              {source && <span style={styles.fileName}>{source.name}</span>}
            </div>

            <div style={styles.uploadGroup}>
              <label style={styles.label}>Target video</label>
              <input
                type="file"
                accept="video/*"
                onChange={(e) => setTarget(e.target.files[0])}
                style={styles.input}
              />
              {target && <span style={styles.fileName}>{target.name}</span>}
            </div>

            <label style={styles.checkbox}>
              <input
                type="checkbox"
                checked={enhance}
                onChange={(e) => setEnhance(e.target.checked)}
              />
              Face restoration (GFPGAN)
            </label>

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
              {(status?.status === 'pending' || status?.status === 'processing') && 'Processing...'}
            </h2>

            {status?.status === 'processing' && (
              <div style={styles.progress}>
                <div style={styles.progressBar}>
                  <div
                    style={{
                      ...styles.progressFill,
                      width: `${status.progress || 0}%`,
                    }}
                  />
                </div>
                <span style={styles.progressText}>
                  {status.progress}% {status.total_frames > 0 && `(${status.total_frames} frames)`}
                </span>
              </div>
            )}

            {status?.error && <p style={styles.error}>{status.error}</p>}

            {status?.status === 'completed' && (
              <button onClick={downloadResult} style={styles.button}>
                Download result
              </button>
            )}

            <button onClick={reset} style={styles.secondaryButton}>
              Start over
            </button>
          </div>
        )}
      </main>

      <footer style={styles.footer}>
        <p>Powered by InsightFace InSwapper · <a href="/docs" style={styles.link}>API docs</a></p>
      </footer>
    </div>
  )
}

const styles = {
  container: {
    maxWidth: 520,
    margin: '0 auto',
    padding: '2rem 1rem',
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    textAlign: 'center',
    marginBottom: '2.5rem',
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
  uploadGroup: {
    marginBottom: '1.25rem',
  },
  label: {
    display: 'block',
    marginBottom: '0.5rem',
    fontSize: '0.9rem',
    color: 'var(--text-muted)',
  },
  input: {
    width: '100%',
    padding: '0.75rem',
    background: 'var(--bg)',
    border: '1px solid var(--border)',
    borderRadius: 8,
    color: 'var(--text)',
    cursor: 'pointer',
  },
  fileName: {
    display: 'block',
    marginTop: '0.35rem',
    fontSize: '0.85rem',
    color: 'var(--accent)',
  },
  checkbox: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    marginBottom: '1rem',
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
    marginBottom: '1rem',
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
    background: 'var(--accent)',
    transition: 'width 0.3s',
  },
  progressText: {
    display: 'block',
    marginTop: '0.5rem',
    fontSize: '0.85rem',
    color: 'var(--text-muted)',
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
