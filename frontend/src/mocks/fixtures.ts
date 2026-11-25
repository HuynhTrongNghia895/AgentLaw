import type { ChatMessage, ChatSession, Citation, LegalRelation } from '../types/chat'

const mockRelation = (title: string, relationType = ''): LegalRelation => ({
  title,
  citation: title,
  excerpt: null,
  relationType,
  effectiveDate: null,
})

export const sampleQuestions = [
  { label: 'ĐA Ý ĐỊNH (MULTI-INTENT)', color: 'purple', question: 'Nghĩa vụ thuế, thời hạn và mức phạt hộ kinh doanh?' },
  { label: 'HIỆU LỰC THỜI GIAN', color: 'yellow', question: 'So sánh quy định áp dụng tại 22/07/2026 và 23/07/2026?' },
  { label: 'TRA CỨU ĐÍCH DANH', color: 'green', question: 'Nội dung Điều 33 Khoản 1 Nghị định 168 quy định gì?' },
  { label: 'LÀM RÕ NHIỀU Ý', color: 'purple', question: 'Người nộp thuế được giảm trừ gia cảnh thế nào và khoản đóng góp từ thiện có được ưu đãi thuế không?' },
] as const

export const mockCitations: Citation[] = [
  {
    id: 'dieu-44',
    title: 'Điều 44 - Luật Quản lý thuế',
    documentName: 'Luật Quản lý thuế số 38/2019/QH14',
    sourceLabel: 'Kho nội bộ',
    effectiveStatus: 'confirmed',
    excerpt: 'Thời hạn nộp thuế được tính theo quy định của luật thuế và thời hạn do cơ quan thuế thông báo.',
    highlightedExcerpt: 'Thời hạn nộp thuế',
    relations: [mockRelation('Luật 108/2025/QH15 hướng dẫn trách nhiệm nộp thuế', 'GUIDES'), mockRelation('Nghị định 168/2025/NĐ-CP hướng dẫn xử lý vi phạm', 'GUIDES')],
  },
  {
    id: 'dieu-59',
    title: 'Điều 59 - Luật Quản lý thuế',
    documentName: 'Luật Quản lý thuế số 38/2019/QH14',
    sourceLabel: 'Kho nội bộ',
    effectiveStatus: 'confirmed',
    excerpt: 'Tiền chậm nộp thuế phải nộp bằng 0.03%/ngày tính trên số tiền thuế chậm nộp.',
    highlightedExcerpt: '0.03%/ngày',
    relations: [mockRelation('Nghị định 168/2025/NĐ-CP quy định mức xử phạt vi phạm', 'GUIDES')],
  },
  {
    id: 'thong-tu-40',
    title: 'Thông tư 40/2021/TT-BTC',
    documentName: 'Thông tư 40/2021/TT-BTC',
    sourceLabel: 'CSDL VBPL',
    effectiveStatus: 'confirmed',
    excerpt: 'Hướng dẫn lập hồ sơ khai thuế, xác định nghĩa vụ nộp thuế và thực hiện thủ tục thuế.',
    highlightedExcerpt: 'hồ sơ khai thuế',
    relations: [mockRelation('Luật Quản lý thuế 38/2019/QH14')],
    sourceUrl: 'https://vbpl.vn/',
  },
]

const commonTrace = (labels: string[]) => labels.map((label, index) => ({
  id: `step-${index}`,
  label,
  status: 'done' as const,
  detail: index === 0 ? 'Đã phân tách ý hỏi' : 'RRF + Neo4j enrichment',
}))

export const mockResponses: Record<string, ChatMessage> = {
  multi: {
    id: 'mock-multi',
    role: 'assistant',
    createdAt: Date.now(),
    content: `### 1. Nghĩa vụ thuế và thời hạn nộp (Q1, Q2)
Hộ kinh doanh thực hiện kê khai hoặc nộp thuế theo phương pháp quy định cho từng sắc thuế.

### 2. Mức phạt chậm nộp tiền thuế (Q3)
Mức phạt chậm nộp tiền thuế là **0,03%/ngày** trên số tiền thuế chậm nộp kể từ ngày liền sau thời hạn đến ngày nộp đủ.`,
    trace: commonTrace(['Full Legal Retrieval', 'Q1: Nghĩa vụ thuế hộ kinh doanh', 'Q2: Thời hạn nộp', 'Q3: Mức phạt chậm nộp']),
    evidences: [
      { id: 'dieu-44', type: 'primary', title: 'Điều 44 - Luật Quản lý thuế', citation: 'Luật Quản lý thuế 38/2019/QH14', excerpt: 'Thời hạn nộp thuế.', source: 'internal', effectiveStatus: 'confirmed', relations: [mockRelation('Luật 108/2025/QH15')] },
      { id: 'dieu-59', type: 'primary', title: 'Điều 59 - Luật Quản lý thuế', citation: 'Luật Quản lý thuế 38/2019/QH14', excerpt: 'Tiền chậm nộp thuế.', source: 'internal', effectiveStatus: 'confirmed', relations: [mockRelation('NĐ 168/2025/NĐ-CP', 'AMENDS')] },
      { id: 'thong-tu-40', type: 'context', title: 'Thông tư 40/2021/TT-BTC', citation: 'Hướng dẫn', excerpt: 'Hướng dẫn lập hồ sơ khai thuế.', source: 'vbpl', effectiveStatus: 'confirmed', relations: [mockRelation('Luật 38/2019/QH14')] },
    ],
    citations: mockCitations,
  },
  time: {
    id: 'mock-time',
    role: 'assistant',
    createdAt: Date.now(),
    content: `### So sánh hiệu lực tại 22/07/2026 và 23/07/2026
- **Trước ngày 23/07/2026:** áp dụng quy định cũ.
- **Từ ngày 23/07/2026:** quy định mới có hiệu lực; văn bản tiếp theo thay thế hoặc sửa đổi cần được đối chiếu theo ngày áp dụng.

Hệ thống chưa tự khẳng định quy định cụ thể nếu không có căn cứ văn bản chính thức.`,
    verificationBanner: { status: 'provisional', title: 'Cảnh báo xác minh thời gian', message: 'Kết quả tạm thời; cần đối chiếu văn bản gốc và mốc hiệu lực chính thức.' },
    trace: commonTrace(['Time-travel route', 'Q1: Mốc 22/07/2026', 'Q2: Mốc 23/07/2026', 'Conflict check']),
    evidences: [
      { id: 'effective-old', type: 'primary', title: 'Văn bản áp dụng trước 23/07/2026', citation: 'Mốc hiệu lực nội bộ', source: 'internal', effectiveStatus: 'confirmed' },
      { id: 'effective-new', type: 'primary', title: 'Văn bản áp dụng từ 23/07/2026', citation: 'Mốc hiệu lực nội bộ', source: 'internal', effectiveStatus: 'provisional' },
    ],
  },
  exact: {
    id: 'mock-exact',
    role: 'assistant',
    createdAt: Date.now(),
    content: `### Điều 33 Khoản 1 Nghị định 168/2025/NĐ-CP
Exact retrieval cần được đối chiếu trực tiếp trên node neo4j. Nội dung chi tiết của khoản chỉ hiển thị đầy đủ sau khi tải evidence từ kho nội bộ.

**Lưu ý:** đây là dữ liệu mock giao diện; không thay thế nguyên văn văn bản pháp luật.`,
    trace: commonTrace(['Exact anchor route', 'NĐ 168/2025/NĐ-CP', 'Điều 33', 'Khoản 1']),
    evidences: [
      { id: 'nd168-d33k1', type: 'primary', title: 'Điều 33 Khoản 1 - NĐ 168', citation: 'Nghị định 168/2025/NĐ-CP', source: 'internal', effectiveStatus: 'confirmed' },
    ],
  },
  clarification: {
    id: 'mock-clarification',
    role: 'assistant',
    createdAt: Date.now(),
    content: 'Đã nhận đủ hai câu làm rõ. AgentLaw có thể tiếp tục tra cứu theo loại sắc thuế và loại tổ chức nhận đóng góp.',
    trace: commonTrace(['Planner', 'Ý 1: đủ dữ kiện', 'Ý 2: gap-fill hoàn tất', 'Retrieval & rerank']),
    clarification: {
      status: 'completed',
      currentIndex: 1,
      pendingIntentLabel: 'Ưu đãi thuế từ thiện',
      questions: [
        {
          id: 'tax-type',
          label: 'Câu 1/2',
          question: 'Loại sắc thuế nào đang được xem xét?',
          answer: 'Thuế TNDN',
          options: [
            { id: 'tndn', label: 'Thuế TNDN (Thu nhập doanh nghiệp)' },
            { id: 'ttcn', label: 'Thuế TNCN (Thu nhập cá nhân)' },
            { id: 'other', label: 'Khác', allowCustomText: true },
          ],
        },
        {
          id: 'charity-type',
          label: 'Câu 2/2',
          question: 'Khoản đóng góp từ thiện được thực hiện thông qua tổ chức nào?',
          answer: 'Quỹ từ thiện hợp pháp',
          options: [
            { id: 'fund', label: 'Quỹ từ thiện, quỹ nhân đạo được thành lập và cấp phép hợp pháp' },
            { id: 'public', label: 'Tài trợ trực tiếp cho cơ sở y tế, giáo dục công lập' },
            { id: 'other', label: 'Khác (Tự nhập trường hợp cụ thể)', allowCustomText: true, description: 'Doanh nghiệp đóng góp trực tiếp bằng hiện vật cho UBND xã...' },
          ],
        },
      ],
    },
  },
}

export function getMockResponse(question: string): ChatMessage {
  const normalized = question.toLowerCase()
  if (normalized.includes('giảm trừ gia cảnh') || normalized.includes('đóng góp từ thiện')) {
    return {
      ...mockResponses.clarification,
      id: crypto.randomUUID(),
      createdAt: Date.now(),
      clarification: {
        status: 'pending',
        currentIndex: 0,
        pendingIntentLabel: 'Ưu đãi thuế từ thiện',
        questions: mockResponses.clarification.clarification!.questions.map((item) => ({ ...item, answer: undefined })),
      },
    }
  }
  if (normalized.includes('22/07/2026') || normalized.includes('hiệu lực')) return mockResponses.time
  if (normalized.includes('điều 33') || normalized.includes('168') || normalized.includes('đích danh')) return mockResponses.exact
  return mockResponses.multi
}

export function createEmptySession(): ChatSession {
  const now = Date.now()
  return { id: crypto.randomUUID(), title: 'Cuộc hội thoại mới', messages: [], createdAt: now, updatedAt: now }
}
