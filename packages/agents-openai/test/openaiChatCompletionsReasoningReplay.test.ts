import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Agent, Runner, setTracingDisabled, tool } from '@openai/agents-core';
import { z } from 'zod';
import { OpenAIChatCompletionsModel } from '../src/openaiChatCompletionsModel';

// DeepSeek thinking mode returns `reasoning_content` and rejects a follow-up
// request whose assistant tool-call message omits it:
// https://api-docs.deepseek.com/guides/thinking_mode#tool-calls
function makeClient(stream: boolean) {
  const create = vi.fn().mockImplementation(async (body: any) => {
    const hasToolOutput = body.messages.some((m: any) => m.role === 'tool');
    const message = hasToolOutput
      ? {
          role: 'assistant',
          content: 'Sunny in Paris.',
          reasoning_content: 'The tool answered.',
        }
      : {
          role: 'assistant',
          content: null,
          reasoning_content: 'I should call get_weather.',
          tool_calls: [
            {
              id: 'call_1',
              type: 'function',
              function: { name: 'get_weather', arguments: '{"city":"Paris"}' },
            },
          ],
        };
    const finish_reason = hasToolOutput ? 'stop' : 'tool_calls';
    if (!stream) {
      return {
        id: 'cmpl',
        choices: [{ index: 0, message, finish_reason }],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      };
    }
    const chunk = (delta: any, finish: string | null = null) => ({
      id: 'cmpl',
      created: 0,
      model: 'deepseek-reasoner',
      object: 'chat.completion.chunk',
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
    return {
      async *[Symbol.asyncIterator]() {
        yield chunk({ role: 'assistant', reasoning_content: 'I should ' });
        yield chunk({ reasoning_content: message.reasoning_content!.slice(9) });
        if (message.tool_calls) {
          yield chunk({ tool_calls: [{ index: 0, ...message.tool_calls[0] }] });
        } else {
          yield chunk({ content: message.content });
        }
        yield chunk({}, finish_reason);
      },
    };
  });
  return {
    create,
    client: {
      chat: { completions: { create } },
      baseURL: 'https://api.deepseek.com',
    },
  };
}

describe('Chat Completions reasoning_content replay', () => {
  beforeEach(() => {
    setTracingDisabled(true);
  });

  for (const stream of [false, true]) {
    it(`replays reasoning_content on the assistant tool-call message (stream: ${stream})`, async () => {
      const { create, client } = makeClient(stream);
      const agent = new Agent({
        name: 'DeepSeek agent',
        model: new OpenAIChatCompletionsModel(
          client as any,
          'deepseek-reasoner',
        ),
        tools: [
          tool({
            name: 'get_weather',
            description: 'Get the weather',
            parameters: z.object({ city: z.string() }),
            execute: async ({ city }) => `sunny in ${city}`,
          }),
        ],
      });

      const result = await new Runner().run(agent, 'Weather in Paris?', {
        stream: stream as any,
      });
      if (stream) {
        for await (const _event of result as any) {
          // Consume the stream.
        }
        await (result as any).completed;
      }

      expect(result.finalOutput).toBe('Sunny in Paris.');
      expect(create).toHaveBeenCalledTimes(2);
      const replayed = create.mock.calls[1][0].messages.find(
        (m: any) => m.role === 'assistant',
      );
      expect(replayed).toMatchObject({
        tool_calls: [expect.objectContaining({ id: 'call_1' })],
        reasoning_content: 'I should call get_weather.',
      });
      expect(replayed).not.toHaveProperty('reasoning');
      const reasoningItem = result.newItems.find(
        (item) => item.type === 'reasoning_item',
      );
      expect(reasoningItem?.rawItem).toMatchObject({
        rawContent: [
          { type: 'reasoning_text', text: 'I should call get_weather.' },
        ],
      });
    });
  }
});
