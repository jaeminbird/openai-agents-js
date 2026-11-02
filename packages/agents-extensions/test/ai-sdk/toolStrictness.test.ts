import { describe, expect, test } from 'vitest';
import { Agent, run, setTracingDisabled, tool } from '@openai/agents';
import { ReadableStream } from 'node:stream/web';
import { z } from 'zod';
import { aisdk } from '../../src/ai-sdk/index';

setTracingDisabled(true);

type Guard = (functionTools: any[]) => void;

// DeepSeek: @ai-sdk/deepseek (vercel/ai#19428) rejects strict function tools before
// fetch unless created with a /beta base URL, and rejects mixed strictness in beta mode.
const deepseekGuard =
  (baseURL: string): Guard =>
  (functionTools) => {
    const hasStrictTool = functionTools.some((t) => t.strict === true);
    if (hasStrictTool && !baseURL.endsWith('/beta')) {
      throw new Error(
        'DeepSeek strict tool calls require a beta base URL ending in `/beta`.',
      );
    }
    if (hasStrictTool && functionTools.some((t) => t.strict !== true)) {
      throw new Error(
        'DeepSeek strict mode requires every function tool in the request to set `strict: true`.',
      );
    }
  };

// Anthropic: @ai-sdk/anthropic forwards `strict` for Claude 4.5+ models, and the API
// compiles strict tools into a grammar. Documented limits (structured outputs guide):
// no numeric/string constraints, at most 16 parameters with union types.
const anthropicGuard: Guard = (functionTools) => {
  let unions = 0;
  for (const t of functionTools.filter((t) => t.strict === true)) {
    for (const [name, prop] of Object.entries<any>(
      t.inputSchema.properties ?? {},
    )) {
      for (const k of ['minimum', 'maximum', 'minLength', 'maxLength']) {
        if (k in prop) {
          throw new Error(
            `tools.${t.name}.${name}: '${k}' is not supported in strict mode`,
          );
        }
      }
      if (prop.anyOf || Array.isArray(prop.type)) unions++;
    }
  }
  if (unions > 16) {
    throw new Error('Schema is too complex for compilation.');
  }
};

function stubProvider(provider: string, guard: Guard) {
  const calls: any[] = [];
  const usage = { inputTokens: 1, outputTokens: 1, totalTokens: 2 };
  const respond = (options: any) => {
    calls.push(options);
    guard((options.tools ?? []).filter((t: any) => t.type === 'function'));
    return (options.prompt ?? []).some((m: any) => m.role === 'tool');
  };
  const model = {
    specificationVersion: 'v3',
    provider,
    modelId: 'm',
    supportedUrls: {},
    async doGenerate(options: any) {
      const done = respond(options);
      return {
        content: done
          ? [{ type: 'text', text: 'Booked.' }]
          : [
              {
                type: 'tool-call',
                toolCallId: 'call_1',
                toolName: 'book_flight',
                input: '{"from":"SFO","to":"JFK","passengers":1}',
              },
            ],
        usage,
        response: { id: 'r' },
        finishReason: done ? 'stop' : 'tool-calls',
        warnings: [],
        providerMetadata: {},
      };
    },
    async doStream(options: any) {
      const done = respond(options);
      const parts: any[] = done
        ? [
            { type: 'text-start', id: 't' },
            { type: 'text-delta', id: 't', delta: 'Booked.' },
            { type: 'text-end', id: 't' },
          ]
        : [
            {
              type: 'tool-call',
              toolCallId: 'call_1',
              toolName: 'book_flight',
              input: '{"from":"SFO","to":"JFK","passengers":1}',
            },
          ];
      parts.push({
        type: 'finish',
        finishReason: done ? 'stop' : 'tool-calls',
        usage,
      });
      return {
        stream: new ReadableStream({
          start(controller) {
            for (const p of parts) controller.enqueue(p);
            controller.close();
          },
        }),
      };
    },
  };
  return { model: model as any, calls };
}

const bookFlight = () =>
  tool({
    name: 'book_flight',
    description: 'Book a flight.',
    parameters: z.object({
      from: z.string().length(3),
      to: z.string().length(3),
      passengers: z.number().int().min(1).max(9),
    }),
    execute: async () => 'booked',
  });

async function runAgent(agent: Agent<any, any>, stream: boolean) {
  const result = await run(agent, 'Book SFO to JFK for one.', { stream });
  if (stream) {
    for await (const _event of result as any) {
      // drain
    }
    await (result as any).completed;
  }
  return result.finalOutput;
}

describe('aisdk() tool strictness', () => {
  test.each([false, true])(
    'Anthropic: Zod tools with constraints run (stream: %s)',
    async (stream) => {
      const { model, calls } = stubProvider(
        'anthropic.messages',
        anthropicGuard,
      );
      const agent = new Agent({
        name: 'travel',
        model: aisdk(model),
        tools: [bookFlight()],
      });
      expect(await runAgent(agent, stream)).toBe('Booked.');
      expect(calls[0].tools[0].strict).toBeUndefined();
    },
  );

  test('Anthropic: many optional parameters across tools run', async () => {
    const { model } = stubProvider('anthropic.messages', anthropicGuard);
    const search = (i: number) =>
      tool({
        name: `search_${i}`,
        description: 'Search.',
        parameters: z.object({
          query: z.string(),
          limit: z.number().optional(),
          cursor: z.string().optional(),
          sort: z.enum(['asc', 'desc']).optional(),
        }),
        execute: async () => 'none',
      });
    const agent = new Agent({
      name: 'travel',
      model: aisdk(model),
      tools: [
        tool({
          name: 'book_flight',
          description: 'Book a flight.',
          parameters: z.object({
            from: z.string(),
            to: z.string(),
            passengers: z.number(),
          }),
          execute: async () => 'booked',
        }),
        ...[0, 1, 2, 3, 4, 5].map(search),
      ],
    });
    expect(await runAgent(agent, false)).toBe('Booked.');
  });

  test.each([false, true])(
    'DeepSeek standard endpoint: default tools run (stream: %s)',
    async (stream) => {
      const { model } = stubProvider(
        'deepseek.chat',
        deepseekGuard('https://api.deepseek.com'),
      );
      const agent = new Agent({
        name: 'travel',
        model: aisdk(model),
        tools: [bookFlight()],
      });
      expect(await runAgent(agent, stream)).toBe('Booked.');
    },
  );

  test('DeepSeek beta endpoint: tools plus a handoff run', async () => {
    const { model } = stubProvider(
      'deepseek.chat',
      deepseekGuard('https://api.deepseek.com/beta'),
    );
    const agent = new Agent({
      name: 'travel',
      model: aisdk(model),
      tools: [bookFlight()],
      handoffs: [new Agent({ name: 'billing', model: aisdk(model) })],
    });
    expect(await runAgent(agent, false)).toBe('Booked.');
  });

  test('OpenAI providers keep strict tools', async () => {
    const { model, calls } = stubProvider('openai.responses', () => {});
    const agent = new Agent({
      name: 'travel',
      model: aisdk(model),
      tools: [bookFlight()],
    });
    expect(await runAgent(agent, false)).toBe('Booked.');
    expect(calls[0].tools[0].strict).toBe(true);
  });

  test('strictTools option overrides the default', async () => {
    const forced = stubProvider('anthropic.messages', () => {});
    await runAgent(
      new Agent({
        name: 'travel',
        model: aisdk(forced.model, { strictTools: true }),
        tools: [bookFlight()],
      }),
      false,
    );
    expect(forced.calls[0].tools[0].strict).toBe(true);

    const disabled = stubProvider('openai.responses', () => {});
    await runAgent(
      new Agent({
        name: 'travel',
        model: aisdk(disabled.model, { strictTools: false }),
        tools: [bookFlight()],
      }),
      false,
    );
    expect(disabled.calls[0].tools[0].strict).toBeUndefined();
  });
});
