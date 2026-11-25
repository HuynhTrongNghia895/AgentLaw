# AgentLaw Frontend

Frontend demo được xây bằng Vite, React, TypeScript và Tailwind CSS.

## Chạy local

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Mở `http://localhost:5173`.

## Cấu trúc

- `src/types/chat.ts`: schema cho message, evidence, trace, citation và clarification.
- `src/db/conversations.ts`: IndexedDB qua thư viện `idb`.
- `src/mocks/fixtures.ts`: dữ liệu mẫu.
- `src/services/chatService.ts`: adapter mock; thay bằng API FastAPI sau.
- `src/App.tsx`: Màn 1 Main Chat và Màn 2 Active Chat.
