import { useState, useEffect, useRef } from 'react'

const API_BASE = '/api'

const PRESETS = {
  quick: {
    label: 'Quick',
    desc: 'Fast preview — no face enhancement',
    detail: 'HyperSwap + 256px + RetinaFace',
    time: '~1 min',
  },
  best: {
    label: 'Best',
    desc: 'Recommended — sharp, realistic faces',
    detail: 'HyperSwap + 256px + enhancement (two-pass)',
    time: '~2-3 min',
  },
  max: {
    label: 'Max',
    desc: 'Maximum quality — slower',
    detail: 'HyperSwap + 512px + enhancement (two-pass)',
    time: '~5-8 min',
  },
}

const FF_STAGES = [
  { id: 'swapping', label: 'Swapping faces', icon: '🔄' },
  { id: 'enhancing', label: 'Enhancing faces', icon: '✨' },
  { id: 'validating', label: 'Checking frame quality', icon: '🔍' },
  { id: 'repairing', label: 'Repairing flickering frames', icon: '🔧' },
  { id: 'done', label: 'Complete', icon: '✓' },
]

const CLASSIC_STAGES = [
  { id: 'extracting', label: 'Extracting frames', icon: '📂' },
  { id: 'swapping', label: 'Swapping faces', icon: '🔄' },
  { id: 'interpolating', label: 'Motion smoothing', icon: '✨' },
  { id: 'merging', label: 'Merging video', icon: '🎬' },
  { id: 'done', label: 'Complete', icon: '✓' },
]

export default function App() {
  const [capabilities, setCapabilities] = useState({ classic: true, facefusion: false })
  const [mainTab, setMainTab] = useState('swap')
  const [source, setSource] = useState(null)
  const [target, setTarget] = useState(null)
  const [sourcePreview, setSourcePreview] = useState(null)
  const [targetPreview, setTargetPreview] = useState(null)
  const [preset, setPreset] = useState('best')
  const [lipSync, setLipSync] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Advanced overrides
  const [advModel, setAdvModel] = useState('hyperswap_1a_256')
  const [advPixelBoost, setAdvPixelBoost] = useState('256')
  const [advDetector, setAdvDetector] = useState('retinaface')
  const [advDetectorScore, setAdvDetectorScore] = useState(0.35)
  const [advSelectorMode, setAdvSelectorMode] = useState('reference')
  const [advMaskBlur, setAdvMaskBlur] = useState(0.3)
  const [advEnhancer, setAdvEnhancer] = useState(true)
  const [advEnhancerBlend, setAdvEnhancerBlend] = useState(0.5)
  const [advTwoPass, setAdvTwoPass] = useState(true)

  // Multi-angle
  const [multiSources, setMultiSources] = useState([])
  const [multiTarget, setMultiTarget] = useState(null)
  const [multiTargetPreview, setMultiTargetPreview] = useState(null)
  const [multiEnhancer, setMultiEnhancer] = useState(true)

  // Job state
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)
  const [polling, setPolling] = useState(false)
  const autoDownloadedRef = useRef(null)

  useEffect(() => {
    fetch(`${API_BASE}/capabilities`)
      .then((r) => r.json())
      .then((data) => {
        setCapabilities(data)
        if (data.facefusion) setMainTab('swap')
      })
      .catch(() => setCapabilities({ classic: true, facefusion: false }))
  }, [])

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

  useEffect(() => {
    if (multiTarget) {
      const url = URL.createObjectURL(multiTarget)
      setMultiTargetPreview(url)
      return () => URL.revokeObjectURL(url)
    }
    setMultiTargetPreview(null)
  }, [multiTarget])

  const reset = () => {
    setSource(null)
    setTarget(null)
    setSourcePreview(null)
    setTargetPreview(null)
    setMultiSources([])
    setMultiTarget(null)
    setMultiTargetPreview(null)
    setJobId(null)
    setStatus(null)
    setError(null)
    setPolling(false)
    autoDownloadedRef.current = null
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
      } catch {
        clearInterval(interval)
        setPolling(false)
        setError('Failed to poll status')
      }
    }, 1000)
  }

  // ---- Submit handlers ----

  const handleSubmitPreset = async (e) => {
    e.preventDefault()
    setError(null)
    if (!source || !target) {
      setError('Please upload both a source photo and a target video')
      return
    }
    const form = new FormData()
    form.append('source', source)
    form.append('target', target)
    form.append('preset', preset)
    form.append('lip_sync', lipSync)
    try {
      const res = await fetch(`${API_BASE}/swap-preset`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      setJobId(data.job_id)
      pollStatus(data.job_id)
    } catch (e) {
      setError(e.message)
    }
  }

  const handleSubmitAdvanced = async (e) => {
    e.preventDefault()
    setError(null)
    if (!source || !target) {
      setError('Please upload both a source photo and a target video')
      return
    }
    const form = new FormData()
    form.append('source', source)
    form.append('target', target)
    form.append('facefusion_model', advModel)
    form.append('facefusion_pixel_boost', advPixelBoost)
    form.append('facefusion_face_enhancer', advEnhancer)
    form.append('facefusion_face_enhancer_blend', String(advEnhancerBlend))
    form.append('facefusion_lip_sync', lipSync)
    form.append('face_detector_model', advDetector)
    form.append('face_detector_score', String(advDetectorScore))
    form.append('face_selector_mode', advSelectorMode)
    form.append('face_mask_blur', String(advMaskBlur))
    form.append('two_pass', advTwoPass)
    try {
      const res = await fetch(`${API_BASE}/swap-pro`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      setJobId(data.job_id)
      pollStatus(data.job_id)
    } catch (e) {
      setError(e.message)
    }
  }

  const handleSubmitMulti = async (e) => {
    e.preventDefault()
    setError(null)
    if (multiSources.length < 1 || multiSources.length > 5) {
      setError('Upload 1 to 5 source images')
      return
    }
    if (!multiTarget) {
      setError('Upload target video')
      return
    }
    const form = new FormData()
    multiSources.forEach((file) => form.append('sources', file))
    form.append('target', multiTarget)
    form.append('face_enhancer', multiEnhancer)
    try {
      const res = await fetch(`${API_BASE}/swap-multi`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) {
        const msg = Array.isArray(data.detail)
          ? data.detail.map((d) => d.msg || d.loc?.join('.')).join(' ')
          : data.detail || 'Upload failed'
        throw new Error(typeof msg === 'string' ? msg : 'Upload failed')
      }
      setJobId(data.job_id)
      pollStatus(data.job_id)
    } catch (e) {
      setError(e.message)
    }
  }

  const downloadResult = () => {
    if (!jobId) return
    const s = status?.settings || {}
    let suffix = jobId
    const p = s.preset
    if (p) {
      const enh = s.facefusion_face_enhancer ? 'enh1' : 'enh0'
      suffix = `${p}_${s.facefusion_model || 'hyperswap'}_p${s.facefusion_pixel_boost || '256'}_${enh}`
    } else if (s.multi_angle) {
      suffix = 'multi_hyperswap_1a_256_p256'
    } else if (s.pro_mode) {
      suffix = `pro_${s.facefusion_model || 'hyperswap'}_p${s.facefusion_pixel_boost || '256'}`
    }
    const a = document.createElement('a')
    a.href = `${API_BASE}/result/${jobId}`
    a.download = `ultrafaceswap_${suffix}.mp4`
    a.click()
  }

  useEffect(() => {
    if (status?.status === 'completed' && jobId && autoDownloadedRef.current !== jobId) {
      autoDownloadedRef.current = jobId
      downloadResult()
    }
  }, [status?.status, jobId])

  const isFaceFusion = status?.settings?.engine === 'facefusion'
  const stages = isFaceFusion ? FF_STAGES : CLASSIC_STAGES
  const curStage = status?.stage || ''
  const stageIdx = stages.findIndex((s) => s.id === curStage)
  const currentStageIndex = stageIdx >= 0 ? stageIdx : status?.status === 'completed' ? stages.length - 1 : 0

  const ffUnavailable = !capabilities.facefusion

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>UltraFaceswap</h1>
        <p style={styles.subtitle}>Professional face swap for video</p>
      </header>

      <main style={styles.main}>
        {!jobId ? (
          <>
            {/* Tab bar */}
            <div style={styles.mainTabs}>
              <button
                type="button"
                style={{ ...styles.mainTab, ...(mainTab === 'swap' ? styles.mainTabActive : {}), ...(ffUnavailable ? styles.tabDisabled : {}) }}
                onClick={() => !ffUnavailable && setMainTab('swap')}
                title={ffUnavailable ? 'FaceFusion required' : ''}
              >
                Face Swap{ffUnavailable ? ' (unavailable)' : ''}
              </button>
              <button
                type="button"
                style={{ ...styles.mainTab, ...(mainTab === 'multi' ? styles.mainTabActive : {}), ...(ffUnavailable ? styles.tabDisabled : {}) }}
                onClick={() => !ffUnavailable && setMainTab('multi')}
                title={ffUnavailable ? 'FaceFusion required' : ''}
              >
                Multi-angle{ffUnavailable ? ' (unavailable)' : ''}
              </button>
            </div>

            {/* ===== FACE SWAP TAB ===== */}
            {mainTab === 'swap' && (
              <form onSubmit={showAdvanced ? handleSubmitAdvanced : handleSubmitPreset} style={styles.form}>
                {/* Upload row */}
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
                          muted loop playsInline
                          onLoadedMetadata={(e) => e.target.play().catch(() => {})}
                        />
                      ) : (
                        <span style={styles.placeholder}>+ Add video</span>
                      )}
                    </div>
                    {target && <span style={styles.fileName}>{target.name}</span>}
                  </div>
                </div>

                {/* Preset picker */}
                {!showAdvanced && (
                  <div style={styles.presetSection}>
                    <h3 style={styles.sectionTitle}>Quality</h3>
                    <div style={styles.presetGrid}>
                      {Object.entries(PRESETS).map(([key, p]) => (
                        <button
                          key={key}
                          type="button"
                          style={{
                            ...styles.presetCard,
                            ...(preset === key ? styles.presetCardActive : {}),
                          }}
                          onClick={() => setPreset(key)}
                        >
                          <span style={styles.presetLabel}>{p.label}</span>
                          <span style={styles.presetDesc}>{p.desc}</span>
                          <span style={styles.presetDetail}>{p.detail}</span>
                          <span style={styles.presetTime}>{p.time}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Advanced panel */}
                {showAdvanced && (
                  <div style={styles.advancedSection}>
                    <h3 style={styles.sectionTitle}>Advanced settings</h3>
                    <div style={styles.option}>
                      <label style={styles.optionLabel}>Swap model</label>
                      <select value={advModel} onChange={(e) => setAdvModel(e.target.value)} style={styles.select}>
                        <option value="hyperswap_1a_256">HyperSwap 1A (best identity)</option>
                        <option value="inswapper_128_fp16">InSwapper 128 (fast)</option>
                        <option value="simswap_256">SimSwap 256</option>
                        <option value="blendswap_256">BlendSwap 256</option>
                      </select>
                    </div>
                    <div style={styles.option}>
                      <label style={styles.optionLabel}>Pixel boost</label>
                      <select value={advPixelBoost} onChange={(e) => setAdvPixelBoost(e.target.value)} style={styles.select}>
                        <option value="256">256 (fast, good quality)</option>
                        <option value="512">512 (best balance)</option>
                        <option value="768">768 (max quality, slow)</option>
                      </select>
                    </div>
                    <div style={styles.option}>
                      <label style={styles.optionLabel}>Face detector</label>
                      <select value={advDetector} onChange={(e) => setAdvDetector(e.target.value)} style={styles.select}>
                        <option value="retinaface">RetinaFace (best for angles)</option>
                        <option value="scrfd">SCRFD (lightweight)</option>
                        <option value="yoloface">YOLOFace (default)</option>
                      </select>
                    </div>
                    <div style={styles.option}>
                      <label style={styles.optionLabel}>Face selector mode</label>
                      <select value={advSelectorMode} onChange={(e) => setAdvSelectorMode(e.target.value)} style={styles.select}>
                        <option value="reference">Reference (consistent)</option>
                        <option value="many">Many</option>
                        <option value="one">One</option>
                      </select>
                    </div>
                    <div style={styles.option}>
                      <label style={styles.optionLabel}>Detector confidence ({advDetectorScore})</label>
                      <input
                        type="range" min={0.1} max={0.9} step={0.05}
                        value={advDetectorScore}
                        onChange={(e) => setAdvDetectorScore(parseFloat(e.target.value))}
                        style={{ width: '100%' }}
                      />
                    </div>
                    <div style={styles.option}>
                      <label style={styles.optionLabel}>Face mask blur (0-1)</label>
                      <input
                        type="number" min={0} max={1} step={0.05}
                        value={advMaskBlur}
                        onChange={(e) => setAdvMaskBlur(parseFloat(e.target.value) || 0.3)}
                        style={styles.select}
                      />
                    </div>
                    <label style={styles.checkbox}>
                      <input type="checkbox" checked={advEnhancer} onChange={(e) => setAdvEnhancer(e.target.checked)} />
                      Face enhancer (GFPGAN)
                    </label>
                    {advEnhancer && (
                      <div style={{ ...styles.option, marginTop: 8 }}>
                        <label style={styles.optionLabel}>Enhancer blend (0-1)</label>
                        <input
                          type="number" min={0} max={1} step={0.1}
                          value={advEnhancerBlend}
                          onChange={(e) => setAdvEnhancerBlend(parseFloat(e.target.value) || 0.5)}
                          style={styles.select}
                        />
                      </div>
                    )}
                    <label style={styles.checkbox}>
                      <input type="checkbox" checked={advTwoPass} onChange={(e) => setAdvTwoPass(e.target.checked)} />
                      Two-pass processing (lower memory, recommended)
                    </label>
                  </div>
                )}

                {/* Lip sync toggle */}
                <label style={{ ...styles.checkbox, marginTop: 12 }}>
                  <input type="checkbox" checked={lipSync} onChange={(e) => setLipSync(e.target.checked)} />
                  Lip sync (requires audio in video)
                </label>

                {/* Advanced toggle */}
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  style={styles.advancedToggle}
                >
                  {showAdvanced ? 'Use presets' : 'Advanced settings'}
                </button>

                {error && <p style={styles.error}>{error}</p>}

                <button type="submit" style={styles.button} disabled={!source || !target}>
                  {showAdvanced
                    ? 'Start face swap (advanced)'
                    : `Start face swap — ${PRESETS[preset]?.label || 'Best'}`}
                </button>
              </form>
            )}

            {/* ===== MULTI-ANGLE TAB ===== */}
            {mainTab === 'multi' && (
              <form onSubmit={handleSubmitMulti} style={styles.form}>
                <p style={styles.optionDesc}>
                  Upload 1-5 photos of the same face from different angles for better identity matching. Uses HyperSwap + RetinaFace with two-pass enhancement.
                </p>
                <div style={styles.previewRow}>
                  <div style={styles.uploadGroup}>
                    <label style={styles.label}>Source face(s) — 1 to 5 photos</label>
                    <div style={styles.dropZone}>
                      <input
                        type="file" accept="image/*" multiple
                        onChange={(e) => setMultiSources(e.target.files?.length ? Array.from(e.target.files) : [])}
                        style={styles.hiddenInput}
                      />
                      {multiSources.length > 0 ? (
                        <span style={styles.placeholder}>{multiSources.length} image{multiSources.length !== 1 ? 's' : ''} selected</span>
                      ) : (
                        <span style={styles.placeholder}>+ Add 1 or more photos</span>
                      )}
                    </div>
                    {multiSources.length > 0 && (
                      <span style={styles.fileName}>
                        {multiSources.length > 5 ? 'Max 5 images' : `${multiSources.length} selected`}
                      </span>
                    )}
                  </div>
                  <div style={styles.uploadGroup}>
                    <label style={styles.label}>Target video</label>
                    <div style={styles.dropZone}>
                      <input
                        type="file" accept="video/*"
                        onChange={(e) => setMultiTarget(e.target.files?.[0] || null)}
                        style={styles.hiddenInput}
                      />
                      {multiTargetPreview ? (
                        <video
                          src={multiTargetPreview}
                          style={styles.previewVideo}
                          muted loop playsInline
                          onLoadedMetadata={(ev) => ev.target.play().catch(() => {})}
                        />
                      ) : (
                        <span style={styles.placeholder}>+ Add video</span>
                      )}
                    </div>
                    {multiTarget && <span style={styles.fileName}>{multiTarget.name}</span>}
                  </div>
                </div>
                <label style={styles.checkbox}>
                  <input type="checkbox" checked={multiEnhancer} onChange={(e) => setMultiEnhancer(e.target.checked)} />
                  Face enhancer (two-pass, recommended)
                </label>
                <p style={styles.optionDesc}>
                  HyperSwap + RetinaFace + 256px{multiEnhancer ? ' + enhancement (two-pass)' : ''}
                </p>
                {error && <p style={styles.error}>{error}</p>}
                <button
                  type="submit" style={styles.button}
                  disabled={multiSources.length < 1 || multiSources.length > 5 || !multiTarget}
                >
                  Start multi-angle swap
                </button>
              </form>
            )}
          </>
        ) : (
          /* ===== STATUS VIEW ===== */
          <div style={styles.statusCard}>
            <h2 style={styles.statusTitle}>
              {status?.status === 'completed' && 'Done!'}
              {status?.status === 'failed' && 'Failed'}
              {(status?.status === 'pending' || status?.status === 'processing') && (
                curStage === 'swapping'
                  ? (status?.total_frames > 0
                    ? `Swapping faces (${status?.processed_frames || 0}/${status?.total_frames})`
                    : 'Running FaceFusion...')
                  : curStage === 'enhancing'
                  ? (status?.total_frames > 0
                    ? `Enhancing faces (${status?.processed_frames || 0}/${status?.total_frames})`
                    : 'Enhancing faces...')
                  : curStage === 'extracting'
                  ? 'Extracting frames...'
                  : curStage === 'interpolating'
                  ? 'Motion smoothing...'
                  : curStage === 'merging'
                  ? 'Merging video...'
                  : curStage === 'cloth'
                  ? 'Applying cloth color...'
                  : 'Processing...'
              )}
            </h2>

            {(status?.status === 'processing' || status?.status === 'pending') && (
              <div style={styles.steps}>
                {stages.map((stage, i) => {
                  const done = i < currentStageIndex || (status?.status === 'completed' && i <= currentStageIndex)
                  const active = i === currentStageIndex && status?.status !== 'completed'
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
                      {active && status?.total_frames > 0 && (
                        <span style={styles.stepDetail}>
                          {status.processed_frames || 0} / {status.total_frames} frames
                          {(status.processed_frames || 0) === 0 && ' (loading models...)'}
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
                  <div style={{ ...styles.progressFill, width: `${status?.progress || 0}%` }} />
                </div>
                <span style={styles.progressText}>{status?.progress || 0}%</span>
              </div>
            )}

            {status?.error && <p style={styles.error}>{status.error}</p>}

            {/* Warning banner (OOM fallback / no face detected) */}
            {status?.settings?.warning && status?.status === 'completed' && (
              <div style={styles.warningBanner}>
                {status.settings.warning}
              </div>
            )}

            {/* Frame repair report */}
            {status?.status === 'completed' && status?.settings?.validation && (
              <div style={
                status.settings.validation.repaired_frames > 0 || status.settings.validation.failed_frames > 0
                  ? styles.repairBanner
                  : styles.successBanner
              }>
                <strong>
                  {status.settings.validation.repaired_frames > 0
                    ? `Frame quality: ${status.settings.validation.good_frames}/${status.settings.validation.total_frames} frames perfect, ${status.settings.validation.repaired_frames} repaired`
                    : status.settings.validation.failed_frames > 0
                    ? `Frame quality: ${status.settings.validation.good_frames}/${status.settings.validation.total_frames} frames OK, ${status.settings.validation.failed_frames} could not be repaired`
                    : `Frame quality: all ${status.settings.validation.total_frames} frames verified`}
                </strong>
                {status.settings.repair_details && (
                  <div style={{ marginTop: '0.25rem', fontSize: '0.85rem', opacity: 0.85 }}>
                    {status.settings.repair_details}
                  </div>
                )}
              </div>
            )}

            {(status?.status === 'processing' || status?.status === 'pending' || status?.status === 'completed') && status?.settings && (
              <p style={styles.settingsUsed}>
                {status.settings.preset
                  ? `Preset: ${PRESETS[status.settings.preset]?.label || status.settings.preset} · ${status.settings.facefusion_model} · p${status.settings.facefusion_pixel_boost} · enhance ${status.settings.facefusion_face_enhancer ? 'on' : 'off'}${status.settings.two_pass ? ' (two-pass)' : ''}`
                  : status.settings.multi_angle
                  ? `Multi-angle · HyperSwap · p256 · enhance ${status.settings.facefusion_face_enhancer ? 'on' : 'off'}${status.settings.two_pass ? ' (two-pass)' : ''}`
                  : status.settings.pro_mode
                  ? `Pro · ${status.settings.facefusion_model} · p${status.settings.facefusion_pixel_boost} · det=${status.settings.face_detector_model || 'default'} · enhance ${status.settings.facefusion_face_enhancer ? 'on' : 'off'}${status.settings.two_pass ? ' (two-pass)' : ''}`
                  : status.settings.engine === 'facefusion'
                  ? `FaceFusion · ${status.settings.facefusion_model} · p${status.settings.facefusion_pixel_boost}`
                  : `Classic · ${status.settings.swap_model} · det ${status.settings.det_size} · ${status.settings.upscale}x upscale`}
              </p>
            )}

            {status?.status === 'completed' && (
              <>
                <div style={styles.resultPreview}>
                  <video src={`${API_BASE}/result/${jobId}`} controls style={styles.resultVideo} poster="" />
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
        <p>Powered by FaceFusion &amp; InsightFace</p>
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
    minWidth: 0,
  },
  uploadGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.35rem',
    minWidth: 0,
  },
  label: {
    fontSize: '0.9rem',
    color: 'var(--text-muted)',
  },
  dropZone: {
    position: 'relative',
    aspectRatio: '1',
    minHeight: 0,
    minWidth: 0,
    maxWidth: '100%',
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
    height: '100%',
  },
  previewImg: {
    position: 'absolute',
    inset: 0,
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
  previewVideo: {
    position: 'absolute',
    top: 0,
    left: 0,
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

  // Tabs
  mainTabs: {
    display: 'flex',
    gap: 0,
    marginBottom: '1.25rem',
    border: '1px solid var(--border)',
    borderRadius: 8,
    overflow: 'hidden',
  },
  mainTab: {
    flex: 1,
    padding: '0.75rem 1rem',
    background: 'var(--bg)',
    border: 'none',
    cursor: 'pointer',
    fontSize: '1rem',
    color: 'var(--text-muted)',
    transition: 'all 0.15s',
  },
  mainTabActive: {
    background: 'var(--accent)',
    color: 'var(--bg)',
  },
  tabDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },

  // Preset cards
  presetSection: {
    marginBottom: '1rem',
  },
  sectionTitle: {
    margin: '0 0 0.75rem 0',
    fontSize: '0.95rem',
    color: 'var(--text)',
  },
  presetGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '0.75rem',
  },
  presetCard: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.25rem',
    padding: '0.85rem 0.75rem',
    background: 'var(--bg)',
    border: '2px solid var(--border)',
    borderRadius: 10,
    cursor: 'pointer',
    textAlign: 'left',
    transition: 'all 0.15s',
  },
  presetCardActive: {
    borderColor: 'var(--accent)',
    background: 'rgba(34, 211, 238, 0.06)',
  },
  presetLabel: {
    fontWeight: 600,
    fontSize: '0.95rem',
    color: 'var(--text)',
  },
  presetDesc: {
    fontSize: '0.78rem',
    color: 'var(--text-muted)',
    lineHeight: 1.3,
  },
  presetDetail: {
    fontSize: '0.72rem',
    color: 'var(--text-muted)',
    opacity: 0.7,
    marginTop: 2,
  },
  presetTime: {
    fontSize: '0.72rem',
    color: 'var(--accent)',
    marginTop: 2,
  },

  // Advanced
  advancedSection: {
    marginBottom: '1rem',
    padding: '1rem',
    background: 'var(--bg)',
    borderRadius: 8,
    border: '1px solid var(--border)',
  },
  advancedToggle: {
    display: 'block',
    margin: '0.75rem auto',
    padding: '0.4rem 1rem',
    background: 'transparent',
    color: 'var(--text-muted)',
    border: '1px solid var(--border)',
    borderRadius: 6,
    cursor: 'pointer',
    fontSize: '0.85rem',
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

  // Buttons
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

  // Status
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
  warningBanner: {
    padding: '0.75rem 1rem',
    marginBottom: '1rem',
    background: 'rgba(250, 204, 21, 0.1)',
    border: '1px solid rgba(250, 204, 21, 0.3)',
    borderRadius: 8,
    color: '#ca8a04',
    fontSize: '0.85rem',
    lineHeight: 1.4,
  },
  repairBanner: {
    padding: '0.75rem 1rem',
    marginBottom: '1rem',
    background: 'rgba(59, 130, 246, 0.08)',
    border: '1px solid rgba(59, 130, 246, 0.25)',
    borderRadius: 8,
    color: '#2563eb',
    fontSize: '0.85rem',
    lineHeight: 1.4,
  },
  successBanner: {
    padding: '0.75rem 1rem',
    marginBottom: '1rem',
    background: 'rgba(34, 197, 94, 0.08)',
    border: '1px solid rgba(34, 197, 94, 0.25)',
    borderRadius: 8,
    color: '#16a34a',
    fontSize: '0.85rem',
    lineHeight: 1.4,
  },
  settingsUsed: {
    fontSize: '0.85rem',
    color: 'var(--text-muted)',
    marginBottom: '0.75rem',
    padding: '0.5rem 0.75rem',
    background: 'var(--bg)',
    borderRadius: 6,
  },
  resultPreview: {
    marginBottom: '1rem',
  },
  resultVideo: {
    width: '100%',
    borderRadius: 8,
    background: '#000',
  },
  footer: {
    marginTop: '2rem',
    textAlign: 'center',
    fontSize: '0.85rem',
    color: 'var(--text-muted)',
  },
}
