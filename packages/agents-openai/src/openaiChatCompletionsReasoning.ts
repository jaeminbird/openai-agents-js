import type { protocol } from '@openai/agents-core';

/**
 * Non-standard Chat Completions keys that third-party providers use for
 * reasoning text. `reasoning` is used by providers such as OpenRouter and
 * Ollama; `reasoning_content` is used by providers such as DeepSeek, which
 * reject a replayed assistant tool-call message that omits it.
 */
export const CHAT_COMPLETIONS_REASONING_FIELDS = [
  'reasoning',
  'reasoning_content',
] as const;

export type ChatCompletionsReasoningField =
  (typeof CHAT_COMPLETIONS_REASONING_FIELDS)[number];

const REASONING_FIELD_KEY = 'chatCompletionsReasoningField';

export function createReasoningText(
  text: string,
  field: ChatCompletionsReasoningField,
): protocol.ReasoningText {
  return field === 'reasoning'
    ? { type: 'reasoning_text', text }
    : {
        type: 'reasoning_text',
        text,
        providerData: { [REASONING_FIELD_KEY]: field },
      };
}

/**
 * Returns the Chat Completions key that a reasoning item should be replayed
 * under. Reasoning from other sources keeps the released `reasoning` key.
 */
export function getReasoningField(
  item: protocol.ReasoningItem,
): ChatCompletionsReasoningField {
  return item.rawContent?.[0]?.providerData?.[REASONING_FIELD_KEY] ===
    'reasoning_content'
    ? 'reasoning_content'
    : 'reasoning';
}
