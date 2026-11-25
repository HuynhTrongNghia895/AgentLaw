export type EvidenceType = 'primary' | 'context'

export interface LegalRelation {
  title: string
  citation: string
  excerpt: string | null
  relationType: string
  effectiveDate: string | null
}

export interface TraceStep {
  id: string
  label: string
  status: 'done' | 'running' | 'pending' | 'skipped'
  detail?: string
}

export interface Evidence {
  id: string
  type: EvidenceType
  title: string
  citation: string
  excerpt?: string | null
  excerptStatus?: 'available' | 'unavailable'
  source?: 'internal' | 'vbpl' | 'web'
  effectiveStatus?: 'confirmed' | 'provisional' | 'unknown'
  article?: string
  clause?: string
  sourceUrl?: string
  relations?: LegalRelation[]
}

export interface VerificationBanner {
  status: 'confirmed' | 'provisional'
  title: string
  message: string
}

export interface Citation {
  id: string
  title: string
  documentName: string
  sourceLabel: string
  effectiveStatus: 'confirmed' | 'provisional' | 'unknown'
  excerpt: string | null
  excerptStatus?: 'available' | 'unavailable'
  highlightedExcerpt?: string
  sourceUrl?: string
  relations?: LegalRelation[]
}

export interface ClarificationOption {
  id: string
  label: string
  description?: string
  allowCustomText?: boolean
}

export interface ClarificationQuestion {
  id: string
  label: string
  question: string
  options: ClarificationOption[]
  answer?: string
}

export interface ClarificationState {
  status: 'pending' | 'completed'
  currentIndex: number
  questions: ClarificationQuestion[]
  pendingIntentLabel?: string
  completedSummary?: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  createdAt: number
  trace?: TraceStep[]
  evidences?: Evidence[]
  verificationBanner?: VerificationBanner
  citations?: Citation[]
  clarification?: ClarificationState
}

export interface ChatSession {
  id: string
  title: string
  messages: ChatMessage[]
  conversationContext?: unknown
  plannerState?: unknown
  createdAt: number
  updatedAt: number
}

export interface SystemMetadata {
  internalDataLabel: string
  internalDataDate?: string
  readiness: 'ready' | 'loading' | 'offline'
  isMock: boolean
}
