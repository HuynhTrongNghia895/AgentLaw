import { createEmptySession, getMockResponse } from '../mocks/fixtures'
import type { ChatMessage, ChatSession, SystemMetadata } from '../types/chat'

const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true'
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export type ChatRequestKind = 'question' | 'clarification'

export function requestFingerprint(
  question: string,
  userAnswer = '',
  kind: ChatRequestKind = 'question',
): string {
  const normalize = (value: string) => value.trim().replace(/\s+/g, ' ').toLocaleLowerCase()
  return `${kind}|${normalize(question)}|${normalize(userAnswer)}`
}

export interface ChatService {
  sendMessage(
    session: ChatSession,
    question: string,
    userAnswer?: string,
    signal?: AbortSignal,
  ): Promise<ChatMessage>
  getSystemMetadata(): Promise<SystemMetadata>
  exportToDocx(messageId: string, content: string): Promise<void>
}

class ApiChatService implements ChatService {
  async getSystemMetadata(): Promise<SystemMetadata> {
    if (USE_MOCK) {
      return { internalDataLabel: 'Dữ liệu mẫu', readiness: 'ready', isMock: true }
    }
    try {
      const res = await fetch(`${API_BASE_URL}/api/metadata`)
      if (!res.ok) throw new Error(`HTTP error ${res.status}: ${res.statusText}`)
      const data = await res.json()
      return {
        internalDataLabel: data.internalDataLabel || 'Dữ liệu nội bộ',
        internalDataDate: data.internalDataDate,
        internalDocuments: Array.isArray(data.internalDocuments) ? data.internalDocuments : undefined,
        readiness: data.readiness || 'ready',
        isMock: false,
      }
    } catch (err: unknown) {
      console.error('[chatService] Error fetching metadata:', err)
      return { internalDataLabel: 'Dữ liệu nội bộ (Mất kết nối)', readiness: 'offline', isMock: false }
    }
  }

  async sendMessage(
    session: ChatSession,
    question: string,
    userAnswer?: string,
    signal?: AbortSignal,
  ): Promise<ChatMessage> {
    if (USE_MOCK) {
      await new Promise((resolve, reject) => {
        const timer = window.setTimeout(resolve, 450)
        signal?.addEventListener('abort', () => {
          window.clearTimeout(timer)
          reject(new DOMException('Aborted', 'AbortError'))
        })
      })
      return getMockResponse(question)
    }

    const payload = {
      message: question,
      userAnswer: userAnswer || '',
      conversationContext:
        typeof session.conversationContext === 'string' ? session.conversationContext : '',
      plannerState: session.plannerState || null,
    }

    let response: Response
    try {
      response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal,
      })
    } catch (netErr: unknown) {
      if (netErr instanceof DOMException && netErr.name === 'AbortError') throw netErr
      const msg = netErr instanceof Error ? netErr.message : String(netErr)
      throw new Error(
        `Không thể kết nối tới máy chủ AgentLaw tại ${API_BASE_URL}. Vui lòng kiểm tra xem backend đã được khởi động chưa. (${msg})`,
      )
    }

    if (!response.ok) {
      let errorDetail = `Mã lỗi ${response.status} (${response.statusText})`
      try {
        const errorJson = await response.json()
        if (errorJson.detail) {
          errorDetail =
            typeof errorJson.detail === 'string'
              ? errorJson.detail
              : JSON.stringify(errorJson.detail)
        }
      } catch {
        // use default errorDetail
      }
      throw new Error(`Lỗi máy chủ AgentLaw: ${errorDetail}`)
    }

    const data = await response.json()

    if (data.conversationContext) session.conversationContext = data.conversationContext
    session.plannerState = data.plannerState ?? null

    return data.message
  }

  async exportToDocx(messageId: string, content: string): Promise<void> {
    const { Document, Packer, Paragraph, TextRun } = await import('docx')
    const doc = new Document({
      sections: [{
        children: [
          new Paragraph({
            children: [new TextRun({ text: 'AgentLaw - Kết quả tra cứu', bold: true, size: 32 })],
          }),
          ...content
            .split('\n')
            .map((line) => new Paragraph({ children: [new TextRun(line)] })),
        ],
      }],
    })
    const blob = await Packer.toBlob(doc)
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `agentlaw-${messageId}.docx`
    anchor.click()
    URL.revokeObjectURL(url)
  }
}

export const chatService: ChatService = new ApiChatService()
export { createEmptySession }