import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { AlertTriangle, ArrowUp, ChevronDown, Copy, ExternalLink, Globe, Menu, PanelLeft, Plus, Shield, Square, Trash2, X } from 'lucide-react'
import { chatService, createEmptySession, requestFingerprint } from './services/chatService'
import { deleteSession, listSessions, saveSession } from './db/conversations'
import { mockCitations, sampleQuestions } from './mocks/fixtures'
import type { ChatMessage, ChatSession, Citation, Evidence, SystemMetadata, TraceStep } from './types/chat'
import './App.css'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function stripCitationDump(content: string, hasEvidences: boolean): string {
  if (!hasEvidences) return content
  return content
    .replace(/\n{0,2}Căn cứ pháp lý\s*[:：][^\n]*/gi, '')
    .trimEnd()
}

function preprocessMarkdown(content: string): string {
  return content.replace(/^(Ý\s+\d+\.)/gm, '**$1**')
}

function extractEffectiveDateFromExcerpt(excerpt: string): { cleanExcerpt: string; effectiveDateBadge: string | null } {
  const match = excerpt.match(/\(Hiệu lực văn bản từ\s+[\d\-/]+\)\s*$/i)
  if (!match) return { cleanExcerpt: excerpt, effectiveDateBadge: null }
  return {
    cleanExcerpt: excerpt.slice(0, match.index).trimEnd(),
    effectiveDateBadge: match[0].replace(/^\(|\)$/g, '').trim(),
  }
}

function cleanContextCitation(raw: string): string {
  let clean = raw.replace(/^-\s*(AMENDS|GUIDES|MODIFIES|SUPPLEMENTS)\s*:\s*/i, '').trim()
  const half = Math.ceil(clean.length / 2)
  if (clean.slice(0, half).trim() === clean.slice(half).trim()) {
    clean = clean.slice(0, half).trim()
  }
  return clean
}

function hasRealAnswer(message: ChatMessage): boolean {
  if (message.clarification?.status === 'pending') return false
  const stripped = message.content.trim()
  if (stripped.length < 60 && !message.evidences?.length && !message.webContext?.length) return false
  return true
}

function isInstructionalOption(label: string, requiresInput?: boolean): boolean {
  if (requiresInput) return true
  return /^(nêu rõ|chọn số hiệu|nhập\s+|bổ sung\s+)/i.test(label.trim())
}

// ---------------------------------------------------------------------------
// Popover
// ---------------------------------------------------------------------------

const DEFAULT_INTERNAL_DOCS = [
  'Luật Doanh nghiệp 59/2020/QH14',
  'Nghị định 168/2025/NĐ-CP',
  'Nghị định 296/2026/NĐ-CP',
  'Luật Quản lý thuế 108/2025/QH15',
  'Luật Thuế TNDN 67/2025/QH15',
  'Luật 116/2025/QH15',
]

function MetadataPopover({
  label,
  isMock,
  documents,
}: {
  label: string
  isMock: boolean
  documents?: string[]
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const docList = documents && documents.length > 0 ? documents : DEFAULT_INTERNAL_DOCS

  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  return (
    <div className="metadata-popover-wrap" ref={ref}>
      <button
        type="button"
        className="metadata-pill"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="true"
      >
        <span>{label}</span>
        {isMock && <span className="mock-label"> · Mock</span>}
      </button>
      {open && (
        <div className="metadata-popover" role="tooltip">
          <div className="metadata-popover-title">Phạm vi văn bản pháp lý ({docList.length})</div>
          <ul className="metadata-popover-list">
            {docList.map((doc) => (
              <li key={doc}>{doc}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Per-message Trace Bar (isolated state per message)
// ---------------------------------------------------------------------------

function MessageTraceBar({ trace }: { trace: TraceStep[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="trace-bar">
      <button className="trace-toggle" type="button" onClick={() => setOpen((o) => !o)}>
        <ChevronDown size={14} className={open ? 'rotated' : ''} />
        <strong>Trace xử lý Agent</strong>
        <span>{trace.filter((s) => s.status === 'done').length}/{trace.length} bước hoàn thành</span>
      </button>
      {open && (
        <div className="trace-details">
          {trace.map((step) => (
            <p key={step.id}>● {step.label}{step.detail ? ` · ${step.detail}` : ''}</p>
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

function Sidebar({
  sessions,
  activeId,
  onNew,
  onSelect,
  onDelete,
  open,
  onClose,
  onToggleDesktop,
}: {
  sessions: ChatSession[]
  activeId: string | null
  onNew: () => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
  open: boolean
  onClose: () => void
  onToggleDesktop: () => void
}) {
  return (
    <aside className={`sidebar ${open ? 'open' : ''}`}>
      <button className="sidebar-close" type="button" onClick={onClose} aria-label="Đóng sidebar"><X size={18} /></button>
      <button className="sidebar-toggle-desktop" type="button" onClick={onToggleDesktop} aria-label="Thu gọn sidebar">
        <PanelLeft size={18} />
      </button>
      <div className="brand">
        <div className="brand-mark"><Shield size={16} /></div>
        <span>AgentLaw</span>
      </div>
      <button className="new-chat" type="button" onClick={onNew}>
        <Plus size={17} />
        <span>Cuộc hội thoại mới</span>
      </button>
      <div className="history-label">LỊCH SỬ GẦN ĐÂY</div>
      <div className="history-list">
        {sessions.map((session) => (
          <div
            className={`history-item ${session.id === activeId ? 'active' : ''}`}
            key={session.id}
            title={session.title}
          >
            <button type="button" className="history-select" onClick={() => onSelect(session.id)}>
              <span>▢</span>
              <span>{session.title}</span>
            </button>
            <button
              type="button"
              className="delete-session"
              aria-label={`Xóa ${session.title}`}
              onClick={() => onDelete(session.id)}
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>
      <div className="sidebar-storage-info">
        Lưu trữ cục bộ trên trình duyệt · AgentLaw
      </div>
    </aside>
  )
}

// ---------------------------------------------------------------------------
// Composer
// ---------------------------------------------------------------------------

function Composer({
  value,
  onChange,
  onSend,
  onAbort,
  compact = false,
  loading = false,
}: {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onAbort: () => void
  compact?: boolean
  loading?: boolean
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      if (!loading && value.trim()) onSend()
    }
  }

  const handleInput = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(event.target.value)
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`
    }
  }

  return (
    <div className={compact ? 'composer compact' : 'composer'}>
      <textarea
        ref={textareaRef}
        value={value}
        onChange={handleInput}
        onKeyDown={onKeyDown}
        disabled={loading}
        rows={compact ? 1 : 2}
        placeholder={
          loading
            ? 'AgentLaw đang tra cứu, vui lòng chờ trong giây lát...'
            : compact
              ? 'Nhập câu hỏi tiếp theo (AgentLaw tự gửi ngữ cảnh cũ)...'
              : 'Nhập câu hỏi pháp luật doanh nghiệp hoặc thuế của bạn...'
        }
      />
      {!compact && <div className="composer-hint">Enter để gửi · Shift + Enter để xuống dòng</div>}

      {loading ? (
        <button
          className="send-button abort-button"
          type="button"
          onClick={onAbort}
          aria-label="Dừng truy vấn"
          title="Dừng truy vấn"
        >
          <Square size={16} />
        </button>
      ) : (
        <button
          className="send-button"
          type="button"
          onClick={onSend}
          disabled={!value.trim()}
          aria-label="Gửi câu hỏi"
        >
          <ArrowUp size={18} />
        </button>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Clarification Wizard
// ---------------------------------------------------------------------------

function ClarificationWizard({
  message,
  onConfirm,
  onSkip,
  disabled,
}: {
  message: ChatMessage
  onConfirm: (msg: ChatMessage, userAnswerPayload: string, displaySummary: string) => void
  onSkip: (msg: ChatMessage) => void
  disabled: boolean
}) {
  const clarification = message.clarification!
  const [index, setIndex] = useState(0)
  const [selectedOptionIds, setSelectedOptionIds] = useState<Record<string, string>>({})
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const [customTexts, setCustomTexts] = useState<Record<string, string>>({})
  const [placeholders, setPlaceholders] = useState<Record<string, string>>({})
  const customInputRef = useRef<HTMLTextAreaElement>(null)

  const clarifyKey = `${message.id}::${clarification.questions.map((q) => q.id).join(',')}`
  const prevKeyRef = useRef(clarifyKey)
  if (prevKeyRef.current !== clarifyKey) {
    prevKeyRef.current = clarifyKey
    setIndex(0)
    setSelectedOptionIds({})
    setAnswers({})
    setCustomTexts({})
    setPlaceholders({})
  }

  const totalQ = clarification.questions.length
  const question = clarification.questions[index]
  if (!question) return null

  const handleOptionClick = (option: {
    id: string
    label: string
    allowCustomText?: boolean
    requiresInput?: boolean
    placeholder?: string
  }) => {
    const needsDetailInput = Boolean(option.allowCustomText) || isInstructionalOption(option.label, option.requiresInput)
    setSelectedOptionIds((prev) => ({ ...prev, [question.id]: option.id }))
    if (needsDetailInput) {
      setAnswers((prev) => ({ ...prev, [question.id]: '' }))
      setPlaceholders((prev) => ({
        ...prev,
        [question.id]: option.placeholder || `Nhập chi tiết cho "${option.label}" (ví dụ: Thuế GTGT, doanh nghiệp; hoặc Luật 59/2020/QH14)...`,
      }))
      window.setTimeout(() => customInputRef.current?.focus(), 0)
    } else {
      setAnswers((prev) => ({ ...prev, [question.id]: option.label }))
      setCustomTexts((prev) => ({ ...prev, [question.id]: '' }))
    }
  }

  const resolvedAnswer = (qId: string): string => {
    if (answers[qId] === '') return (customTexts[qId] ?? '').trim()
    return (answers[qId] ?? '').trim()
  }

  const allAnswered = clarification.questions.every((q) => resolvedAnswer(q.id) !== '')
  const canProceed = resolvedAnswer(question.id) !== ''
  const showDetailBox =
    answers[question.id] === '' || Boolean(customTexts[question.id])

  const handleConfirm = () => {
    if (!allAnswered || disabled) return
    const items = clarification.questions.map((q) => resolvedAnswer(q.id))
    const payload =
      clarification.questions.length === 1
        ? items[0]
        : items.map((ans, idx) => `${idx + 1}) ${ans}`).join('\n')
    const displaySummary = clarification.questions
      .map((q, idx) => `${q.label || `Ý ${idx + 1}`}: ${items[idx]}`)
      .join('; ')
    onConfirm(message, payload, displaySummary)
  }

  return (
    <div className="clarification-card">
      <div className="clarification-summary">
        ✓ Ý hỏi đã đủ dữ kiện · Cần làm rõ: {clarification.pendingIntentLabel}
      </div>
      <div className="clarification-progress">
        {clarification.questions.map((q, i) => (
          <button
            type="button"
            key={q.id}
            className={`progress-step ${i === index ? 'active' : ''} ${resolvedAnswer(q.id) ? 'complete' : ''}`}
            onClick={() => setIndex(i)}
          >
            {resolvedAnswer(q.id) ? '✓' : i === index ? '●' : ''} {q.label}
          </button>
        ))}
      </div>

      <div className="wizard-question">
        <h3>{question.question}</h3>
        <p>
          {showDetailBox
            ? 'Vui lòng nhập nội dung cụ thể vào ô bên dưới để kích hoạt nút xác nhận.'
            : `Hoàn tất đủ ${totalQ} câu để kích hoạt nút xác nhận.`}
        </p>
        <div className="wizard-options">
          {question.options.map((option) => {
            const isSelected = selectedOptionIds[question.id] === option.id
            return (
              <button
                type="button"
                key={option.id}
                className={`wizard-option ${isSelected ? 'selected' : ''}`}
                onClick={() => handleOptionClick(option)}
              >
                <span className="radio-mark" />
                <span>
                  <strong>{option.label}</strong>
                  {option.description && <small>{option.description}</small>}
                </span>
              </button>
            )
          })}
          {showDetailBox && (
            <textarea
              ref={customInputRef}
              className="custom-answer"
              value={customTexts[question.id] ?? ''}
              onFocus={() => setAnswers((prev) => ({ ...prev, [question.id]: '' }))}
              onChange={(e) => {
                const val = e.target.value
                setCustomTexts((prev) => ({ ...prev, [question.id]: val }))
                setAnswers((prev) => ({ ...prev, [question.id]: '' }))
              }}
              placeholder={placeholders[question.id] || 'Nhập trường hợp cụ thể...'}
            />
          )}
        </div>

        <div className="wizard-actions">
          {index > 0 && (
            <button type="button" className="wizard-back" onClick={() => setIndex((i) => i - 1)}>
              ← Quay lại Câu {index}/{totalQ}
            </button>
          )}
          {index < totalQ - 1 ? (
            <button
              type="button"
              className="wizard-next"
              disabled={!canProceed}
              onClick={() => setIndex((i) => i + 1)}
            >
              Tiếp tục
            </button>
          ) : (
            <button type="button" className="wizard-skip" onClick={() => onSkip(message)}>
              Bỏ qua
            </button>
          )}
          {index === totalQ - 1 && (
            <button
              type="button"
              className="wizard-confirm"
              disabled={!allAnswered || disabled}
              onClick={handleConfirm}
            >
              ✓ Xác nhận ({totalQ}/{totalQ})
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Citation Drawer Body
// ---------------------------------------------------------------------------

function CitationDrawerContent({
  citation,
  isContextType,
  copiedCitation,
  onCopy,
}: {
  citation: Citation
  isContextType: boolean
  copiedCitation: boolean
  onCopy: (text: string) => void
}) {
  const { cleanExcerpt, effectiveDateBadge } = extractEffectiveDateFromExcerpt(citation.excerpt ?? '')

  const titleIsDuplicate =
    citation.documentName &&
    citation.title &&
    (citation.documentName.includes(citation.title) ||
      citation.title.includes(citation.documentName) ||
      citation.documentName.trim() === citation.title.trim())

  const hasRelations = citation.relations && citation.relations.length > 0
  const hasExcerpt = cleanExcerpt?.trim().length > 0

  return (
    <div className="drawer-content">
      <div className="drawer-pills">
        <span>{citation.sourceLabel}</span>
        {isContextType ? (
          <span className="status-context">⚖ Quan hệ pháp lý</span>
        ) : (
          <span className={citation.effectiveStatus === 'confirmed' ? 'status-confirmed' : 'status-provisional'}>
            ● Còn hiệu lực
          </span>
        )}
        {effectiveDateBadge && (
          <span className="status-effective-date">{effectiveDateBadge}</span>
        )}
      </div>

      <h3>{citation.documentName}</h3>
      {!titleIsDuplicate && <div className="source-status">{citation.title}</div>}

      {hasExcerpt && (
        <div className="drawer-excerpt">
          <strong>Nguyên văn trọng tâm</strong>
          <p>
            {citation.highlightedExcerpt
              ? cleanExcerpt
                  .replace(citation.highlightedExcerpt, `<<${citation.highlightedExcerpt}>>`)
                  .split('<<')
                  .map((part, index) =>
                    part.endsWith('>>') ? (
                      <mark key={index}>{part.slice(0, -2)}</mark>
                    ) : (
                      <span key={index}>{part}</span>
                    ),
                  )
              : cleanExcerpt}
          </p>
        </div>
      )}

      <button
        type="button"
        className="copy-citation"
        onClick={() => onCopy(cleanExcerpt || citation.excerpt || '')}
      >
        {copiedCitation ? <span className="copy-feedback">✓ Đã sao chép</span> : <><Copy size={14} /> Chép trích dẫn</>}
      </button>

      {hasRelations && (
        <div className="legal-relations">
          <h4>Quan hệ pháp lý liên quan</h4>
          <ul>{citation.relations!.map((relation, index) => (
            <li key={`${relation.relationType}-${relation.citation}-${index}`}>
              <strong>{relation.title || relation.relationType || 'Quan hệ pháp lý'}</strong>
              {relation.citation && <span> — {relation.citation}</span>}
              {relation.excerpt && <p>{relation.excerpt}</p>}
              {relation.effectiveDate && <small>Hiệu lực: {relation.effectiveDate}</small>}
            </li>
          ))}</ul>
        </div>
      )}

      {citation.sourceUrl && (
        <a className="source-link" href={citation.sourceUrl} target="_blank" rel="noreferrer">Mở nguồn gốc VBPL ↗</a>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [loadingControllers, setLoadingControllers] = useState<Record<string, AbortController>>({})
  const inFlightFingerprints = useRef<Set<string>>(new Set())
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [metadata, setMetadata] = useState<SystemMetadata | null>(null)
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null)
  const [selectedCitationIsContext, setSelectedCitationIsContext] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [copiedCitation, setCopiedCitation] = useState(false)
  const booted = useRef(false)

  useEffect(() => {
    if (booted.current) return
    booted.current = true
    void listSessions().then(setSessions)
    void chatService.getSystemMetadata().then(setMetadata)
  }, [])

  const activeLoading = activeId !== null && activeId in loadingControllers
  const activeSession = sessions.find((s) => s.id === activeId) ?? null

  const persist = async (session: ChatSession) => {
    await saveSession(session)
    setSessions((current) => [session, ...current.filter((s) => s.id !== session.id)])
  }

  const selectSession = (id: string) => {
    setActiveId(id)
    setSelectedCitation(null)
    setSidebarOpen(false)
  }

  const createSession = () => {
    const session = createEmptySession()
    void persist(session)
    setActiveId(session.id)
    setSelectedCitation(null)
    setInput('')
    setSidebarOpen(false)
  }

  const deleteSessionById = async (id: string) => {
    await deleteSession(id)
    setSessions((current) => {
      const remaining = current.filter((s) => s.id !== id)
      if (activeId === id) {
        const next = remaining[0]
        setActiveId(next ? next.id : null)
        setSelectedCitation(null)
      }
      return remaining
    })
  }

  const abortSession = (sessionId: string) => {
    const controller = loadingControllers[sessionId]
    if (controller) controller.abort()
    setLoadingControllers((prev) => {
      const next = { ...prev }
      delete next[sessionId]
      return next
    })
  }

  const handleAbort = () => {
    if (activeId) abortSession(activeId)
  }

  const sendQuestion = async (question = input, overrideSessionId?: string) => {
    const text = question.trim()
    if (!text) return

    let session = overrideSessionId
      ? (sessions.find((s) => s.id === overrideSessionId) ?? activeSession)
      : activeSession

    if (!session) {
      session = createEmptySession()
      session.title = text.length > 38 ? `${text.slice(0, 38)}…` : text
    }

    const targetSessionId = session.id
    const fingerprint = `${targetSessionId}|${requestFingerprint(text, '', 'question')}`
    if (targetSessionId in loadingControllers || inFlightFingerprints.current.has(fingerprint)) return
    inFlightFingerprints.current.add(fingerprint)

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      createdAt: Date.now(),
    }
    const nextSession: ChatSession = {
      ...session,
      title:
        session.messages.length === 0
          ? text.length > 38
            ? `${text.slice(0, 38)}…`
            : text
          : session.title,
      messages: [...session.messages, userMessage],
      updatedAt: Date.now(),
    }
    await persist(nextSession)
    setActiveId(targetSessionId)
    setInput('')

    const controller = new AbortController()
    setLoadingControllers((prev) => ({ ...prev, [targetSessionId]: controller }))

    try {
      const answer = await chatService.sendMessage(nextSession, text, undefined, controller.signal)
      const latestSession = await import('./db/conversations').then((m) =>
        m.getSession(targetSessionId),
      )
      const baseMessages = latestSession?.messages ?? nextSession.messages
      const completed: ChatSession = {
        ...(latestSession ?? nextSession),
        conversationContext: nextSession.conversationContext,
        plannerState: nextSession.plannerState,
        messages: [...baseMessages, answer],
        updatedAt: Date.now(),
      }
      await persist(completed)
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        const abortedMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: 'Đã dừng truy vấn theo yêu cầu.',
          createdAt: Date.now(),
        }
        const latestSession = await import('./db/conversations').then((m) =>
          m.getSession(targetSessionId),
        )
        const baseMessages = latestSession?.messages ?? nextSession.messages
        await persist({
          ...(latestSession ?? nextSession),
          messages: [...baseMessages, abortedMsg],
          updatedAt: Date.now(),
        })
        return
      }
      const errMsg = err instanceof Error ? err.message : String(err)
      const errorMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `⚠️ **Lỗi kết nối hoặc xử lý:**\n\n${errMsg}`,
        createdAt: Date.now(),
      }
      const latestSession = await import('./db/conversations').then((m) =>
        m.getSession(targetSessionId),
      )
      const baseMessages = latestSession?.messages ?? nextSession.messages
      await persist({
        ...(latestSession ?? nextSession),
        messages: [...baseMessages, errorMsg],
        updatedAt: Date.now(),
      })
    } finally {
      inFlightFingerprints.current.delete(fingerprint)
      setLoadingControllers((prev) => {
        const next = { ...prev }
        delete next[targetSessionId]
        return next
      })
    }
  }

  const confirmClarification = async (
    message: ChatMessage,
    userAnswerPayload: string,
    displaySummary: string,
  ) => {
    if (!activeSession) return
    const targetSessionId = activeSession.id
    const lastUserMsg = [...activeSession.messages].reverse().find((m) => m.role === 'user')
    const originalQuestion = lastUserMsg?.content ?? ''
    const fingerprint = `${targetSessionId}|${requestFingerprint(
      originalQuestion,
      userAnswerPayload,
      'clarification',
    )}`
    if (targetSessionId in loadingControllers || inFlightFingerprints.current.has(fingerprint)) return
    inFlightFingerprints.current.add(fingerprint)

    const summaryItems = displaySummary.split('; ')
    const completedMsg: ChatMessage = {
      ...message,
      content: `### Đã xác nhận làm rõ\n${summaryItems.map((s) => `- **${s}**`).join('\n')}\n\n*Đang gửi phản hồi làm rõ đến AgentLaw...*`,
      clarification: {
        ...message.clarification!,
        status: 'completed' as const,
        completedSummary: displaySummary,
      },
    }
    const nextSession: ChatSession = {
      ...activeSession,
      messages: activeSession.messages.map((m) => (m.id === message.id ? completedMsg : m)),
      updatedAt: Date.now(),
    }
    await persist(nextSession)

    const controller = new AbortController()
    setLoadingControllers((prev) => ({ ...prev, [targetSessionId]: controller }))

    try {
      const answer = await chatService.sendMessage(
        nextSession,
        originalQuestion,
        userAnswerPayload,
        controller.signal,
      )
      const latestSession = await import('./db/conversations').then((m) =>
        m.getSession(targetSessionId),
      )
      const baseMessages = latestSession?.messages ?? nextSession.messages
      await persist({
        ...(latestSession ?? nextSession),
        conversationContext: nextSession.conversationContext,
        plannerState: nextSession.plannerState,
        messages: [...baseMessages, answer],
        updatedAt: Date.now(),
      })
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        const abortedMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: 'Đã dừng truy vấn theo yêu cầu.',
          createdAt: Date.now(),
        }
        const latestSession = await import('./db/conversations').then((m) =>
          m.getSession(targetSessionId),
        )
        const baseMessages = latestSession?.messages ?? nextSession.messages
        await persist({
          ...(latestSession ?? nextSession),
          messages: [...baseMessages, abortedMsg],
          updatedAt: Date.now(),
        })
        return
      }
      const errMsg = err instanceof Error ? err.message : String(err)
      const errorMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `⚠️ **Không thể hoàn tất bước làm rõ:**\n\n${errMsg}`,
        createdAt: Date.now(),
      }
      const latestSession = await import('./db/conversations').then((m) =>
        m.getSession(targetSessionId),
      )
      const baseMessages = latestSession?.messages ?? nextSession.messages
      await persist({
        ...(latestSession ?? nextSession),
        messages: [...baseMessages, errorMsg],
        updatedAt: Date.now(),
      })
    } finally {
      inFlightFingerprints.current.delete(fingerprint)
      setLoadingControllers((prev) => {
        const next = { ...prev }
        delete next[targetSessionId]
        return next
      })
    }
  }

  const skipClarification = (message: ChatMessage) => {
    if (!activeSession) return
    const intentLabel =
      message.clarification?.pendingIntentLabel ?? message.clarification?.questions[0]?.question ?? 'ý con này'
    const skippedMsg: ChatMessage = {
      ...message,
      content: `Đã bỏ qua bước làm rõ cho ý: **${intentLabel}** — Chưa đủ dữ kiện pháp lý để kết luận, hệ thống chỉ giữ lại phần trả lời của ý đã đủ căn cứ phía trên.`,
      clarification: message.clarification
        ? { ...message.clarification, status: 'completed' as const }
        : undefined,
    }
    const nextSession: ChatSession = {
      ...activeSession,
      plannerState: null,
      messages: activeSession.messages.map((m) => (m.id === message.id ? skippedMsg : m)),
      updatedAt: Date.now(),
    }
    void persist(nextSession)
  }

  const copyAnswer = async (id: string, content: string) => {
    await navigator.clipboard.writeText(content)
    setCopiedId(id)
    window.setTimeout(() => setCopiedId((curr) => (curr === id ? null : curr)), 2000)
  }

  const copyCitationText = async (text: string) => {
    await navigator.clipboard.writeText(text)
    setCopiedCitation(true)
    window.setTimeout(() => setCopiedCitation(false), 2000)
  }

  const exportAnswer = async (message: ChatMessage) => {
    await chatService.exportToDocx(message.id, message.content)
  }

  const openEvidence = (evidence: Evidence) => {
    const isCtx = evidence.type === 'context'
    const citation = mockCitations.find((item) => item.id === evidence.id) ?? {
      id: evidence.id,
      title: evidence.title,
      documentName: cleanContextCitation(evidence.citation),
      sourceLabel: evidence.source === 'vbpl' ? 'CSDL VBPL' : 'Kho nội bộ',
      effectiveStatus: evidence.effectiveStatus ?? 'unknown',
      excerpt: evidence.excerpt ?? evidence.citation,
      highlightedExcerpt: evidence.title,
      sourceUrl: evidence.sourceUrl,
      relations: evidence.relations,
    }
    setSelectedCitation(citation)
    setSelectedCitationIsContext(isCtx)
  }

  return (
    <div className={`app-shell ${sidebarCollapsed ? 'sidebar-collapsed' : ''} ${selectedCitation ? 'drawer-open' : ''}`}>
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        onNew={createSession}
        onSelect={selectSession}
        onDelete={(id) => void deleteSessionById(id)}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onToggleDesktop={() => setSidebarCollapsed(true)}
      />

      <main className="main-area">
        <header className="top-header">
          <button className="mobile-menu" type="button" onClick={() => setSidebarOpen(true)} aria-label="Mở sidebar"><Menu size={18} /></button>
          {sidebarCollapsed && (
            <button className="desktop-reopen" type="button" onClick={() => setSidebarCollapsed(false)} aria-label="Mở lại sidebar">
              <PanelLeft size={18} />
            </button>
          )}
          <div className="header-title">AgentLaw</div>
          <div className="header-status">
            <MetadataPopover
              label={metadata?.internalDataLabel ?? 'Dữ liệu nội bộ'}
              isMock={metadata?.isMock ?? false}
              documents={metadata?.internalDocuments}
            />
            <div className="ready-status"><span className="ready-dot" />Sẵn sàng</div>
          </div>
        </header>

        {!activeSession || activeSession.messages.length === 0 ? (
          <>
            <section className="workspace home-workspace">
              <h1 className="home-title">Tra cứu pháp luật cùng AgentLaw</h1>
              <p className="home-subtitle">Trợ lý hỏi đáp pháp luật doanh nghiệp và thuế tiếng Việt có trích dẫn nguồn</p>
              <div className="composer-wrap">
                <Composer
                  value={input}
                  onChange={setInput}
                  onSend={() => void sendQuestion()}
                  onAbort={handleAbort}
                  loading={activeLoading}
                />
              </div>
              <div className="sample-heading">KỊCH BẢN TRA CỨU MẪU</div>
              <div className="sample-grid">
                {sampleQuestions.map((item) => (
                  <button className="sample-card" type="button" key={item.label} onClick={() => void sendQuestion(item.question)}>
                    <strong className={item.color}>{item.label}</strong>
                    <span>{item.question}</span>
                  </button>
                ))}
              </div>
            </section>
            <footer className="footer">AgentLaw chỉ hỗ trợ tra cứu tham khảo, không thay thế tư vấn pháp lý chính thức. Vui lòng đối chiếu văn bản gốc trước khi áp dụng.</footer>
          </>
        ) : (
          <>
            <section className="workspace chat-workspace">
              <div className="chat-content">
                {activeSession.messages.map((message) => {
                  if (message.role === 'user') {
                    return (
                      <div className="user-row" key={message.id}>
                        <div className="user-bubble">{message.content}</div>
                      </div>
                    )
                  }
                  const primaryEvidences = (message.evidences ?? []).filter((e) => e.type === 'primary')
                  const contextEvidences = (message.evidences ?? []).filter((e) => e.type === 'context')

                  return (
                    <div className="answer-wrap" key={message.id}>
                      <div className="agent-avatar"><Shield size={16} /></div>
                      <div className="answer-column">
                        {message.trace && message.trace.length > 0 && (
                          <MessageTraceBar trace={message.trace} />
                        )}

                        {message.verificationBanner && (
                          <div className="verification-banner">
                            <strong>⚠ {message.verificationBanner.title}</strong>
                            {message.verificationBanner.message}
                          </div>
                        )}

                        <article className="answer-card answer-card--spaced">
                          <ReactMarkdown>
                            {preprocessMarkdown(
                              stripCitationDump(
                                message.content,
                                primaryEvidences.length > 0 || contextEvidences.length > 0,
                              ),
                            )}
                          </ReactMarkdown>
                        </article>

                        {message.clarification && message.clarification.status === 'pending' && (
                          <ClarificationWizard
                            message={message}
                            onConfirm={(msg, payload, summary) => void confirmClarification(msg, payload, summary)}
                            onSkip={skipClarification}
                            disabled={activeLoading}
                          />
                        )}

                        {/* Dedicated Web Extraction & Source Verification Card */}
                        {message.webContext && message.webContext.length > 0 && (
                          <div className="web-context-section">
                            <div className="web-context-header">
                              <Globe size={14} />
                              <span>THÔNG TIN TRÍCH XUẤT TỪ NGUỒN WEB MỞ RỘNG</span>
                            </div>
                            <div className="web-context-warning">
                              <AlertTriangle size={14} />
                              <span>
                                <strong>Lưu ý đối chiếu:</strong> Dữ liệu dưới đây được trích xuất tự động từ trang web bên ngoài. Vui lòng tìm hiểu thêm và đối chiếu trực tiếp văn bản gốc tại nguồn ghi rõ bên dưới trước khi áp dụng.
                              </span>
                            </div>
                            <div className="web-context-list">
                              {message.webContext.map((item) => (
                                <div className="web-context-card" key={item.id}>
                                  <div className="web-context-card-top">
                                    <span className="web-domain-badge">{item.sourceDomain}</span>
                                    {item.authorityTier === 'official' && (
                                      <span className="web-official-badge">Cổng chính thức</span>
                                    )}
                                    {item.date && <span className="web-date-badge">{item.date}</span>}
                                  </div>
                                  <strong className="web-context-title">{item.title}</strong>
                                  {item.extractedPoints.length > 0 && (
                                    <ul className="web-extracted-points">
                                      {item.extractedPoints.map((pt, idx) => (
                                        <li key={idx}>{pt}</li>
                                      ))}
                                    </ul>
                                  )}
                                  <div className="web-context-footer">
                                    <span className="web-item-warning">{item.warning}</span>
                                    {item.url && (
                                      <a
                                        className="web-source-link"
                                        href={item.url}
                                        target="_blank"
                                        rel="noreferrer"
                                      >
                                        Tra cứu tại {item.sourceDomain} <ExternalLink size={12} />
                                      </a>
                                    )}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {(primaryEvidences.length > 0 || contextEvidences.length > 0) && (
                          <div className="evidence-section">
                            {primaryEvidences.length > 0 && (
                              <>
                                <div className="evidence-heading primary-heading">CĂN CỨ TRẢ LỜI CHÍNH (PRIMARY EVIDENCE)</div>
                                <div className="evidence-pills">
                                  {primaryEvidences.map((evidence) => {
                                    const isSelected = selectedCitation?.id === evidence.id
                                    return (
                                      <button
                                        className={`evidence-pill primary-pill ${isSelected ? 'pill-selected-primary' : ''}`}
                                        type="button"
                                        key={evidence.id}
                                        onClick={() => openEvidence(evidence)}
                                        title={evidence.citation}
                                      >
                                        <strong>{evidence.title}</strong>
                                        <span>{evidence.citation}</span>
                                      </button>
                                    )
                                  })}
                                </div>
                              </>
                            )}
                            {contextEvidences.length > 0 && (
                              <>
                                <div className="evidence-heading context-heading">BỐI CẢNH SỬA ĐỔI / HƯỚNG DẪN (LEGAL CONTEXT)</div>
                                <div className="evidence-pills">
                                  {contextEvidences.map((evidence) => {
                                    const cleanCit = cleanContextCitation(evidence.citation)
                                    const isSelected = selectedCitation?.id === evidence.id
                                    return (
                                      <button
                                        className={`evidence-pill context-pill ${isSelected ? 'pill-selected-context' : ''}`}
                                        type="button"
                                        key={evidence.id}
                                        onClick={() => openEvidence(evidence)}
                                        title={evidence.citation}
                                      >
                                        <strong>{evidence.title}</strong>
                                        <span>{cleanCit}</span>
                                      </button>
                                    )
                                  })}
                                </div>
                              </>
                            )}
                          </div>
                        )}

                        {hasRealAnswer(message) && (
                          <div className="answer-actions">
                            <button className="action-button" type="button" onClick={() => void copyAnswer(message.id, message.content)}>
                              {copiedId === message.id ? <span className="copy-feedback">✓ Đã sao chép</span> : <><Copy size={14} /> Sao chép</>}
                            </button>
                            <button className="action-button" type="button" onClick={() => void exportAnswer(message)}>Xuất văn bản (.docx)</button>
                          </div>
                        )}
                      </div>
                    </div>
                  )
                })}

                {activeLoading && (
                  <div className="loading-answer">
                    <span className="loading-spinner" />
                    <span>AgentLaw đang phân tích và tra cứu văn bản...</span>
                  </div>
                )}
              </div>
            </section>
            <div className="chat-composer-row">
              <Composer
                compact
                value={input}
                onChange={setInput}
                onSend={() => void sendQuestion()}
                onAbort={handleAbort}
                loading={activeLoading}
              />
            </div>
          </>
        )}
      </main>

      {selectedCitation && (
        <aside className="citation-drawer" role="dialog" aria-label="Chi tiết căn cứ pháp lý">
          <div className="drawer-header">
            <h2>Chi tiết căn cứ pháp lý</h2>
            <button type="button" onClick={() => setSelectedCitation(null)} aria-label="Đóng ngăn kéo">×</button>
          </div>
          <CitationDrawerContent
            citation={selectedCitation}
            isContextType={selectedCitationIsContext}
            copiedCitation={copiedCitation}
            onCopy={(text) => void copyCitationText(text)}
          />
        </aside>
      )}
    </div>
  )
}

export default App