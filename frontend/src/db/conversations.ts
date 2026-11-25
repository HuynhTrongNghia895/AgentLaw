import { openDB, type DBSchema } from 'idb'
import type { ChatSession } from '../types/chat'

interface AgentLawDB extends DBSchema {
  sessions: {
    key: string
    value: ChatSession
  }
}

const dbPromise = openDB<AgentLawDB>('agentlaw-conversations', 1, {
  upgrade(db) {
    db.createObjectStore('sessions', { keyPath: 'id' })
  },
})

export async function listSessions(): Promise<ChatSession[]> {
  const db = await dbPromise
  return (await db.getAll('sessions')).sort((a, b) => b.updatedAt - a.updatedAt)
}

export async function getSession(id: string): Promise<ChatSession | undefined> {
  return (await dbPromise).get('sessions', id)
}

export async function saveSession(session: ChatSession): Promise<void> {
  await (await dbPromise).put('sessions', session)
}

export async function deleteSession(id: string): Promise<void> {
  await (await dbPromise).delete('sessions', id)
}
