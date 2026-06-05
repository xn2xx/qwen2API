/**
 * 读取当前会话凭证。
 * - 优先使用 localStorage 中用户显式保存的 key
 * - 仅在没有任何配置时回退到 "admin"，避免页面切换/刷新后静默丢失凭证
 */
export function getStoredApiKey(): string {
  try {
    const stored = localStorage.getItem('qwen2api_key')
    if (stored && stored.trim()) return stored.trim()
  } catch {
    // localStorage 不可用时继续走默认值
  }
  return 'admin'
}

export function getAuthHeader(): Record<string, string> {
  const key = getStoredApiKey()
  if (!key) return {}
  return { Authorization: `Bearer ${key}` }
}
