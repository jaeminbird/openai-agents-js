import { describe, expect, it, vi } from 'vitest';
import { z } from 'zod';
import {
  Agent,
  MemorySession,
  Runner,
  RunState,
  tool,
  type AgentInputItem,
} from '@openai/agents-core';
import {
  ScriptedModel,
  assistantMessage,
  functionCall,
} from '@openai/agents-core/testing';
import { OpenAIResponsesCompactionSession } from '../src';

// responses.compact rejects input whose function calls have no output yet
// (live check in openai/openai-agents-python#5363).
function unresolvedCallIds(items: AgentInputItem[]): string[] {
  const outputs = new Set(
    items
      .filter((item: any) => item.type === 'function_call_output')
      .map((item: any) => item.call_id),
  );
  return items
    .filter(
      (item: any) =>
        item.type === 'function_call' && !outputs.has(item.call_id),
    )
    .map((item: any) => item.call_id);
}

function setup() {
  const compact = vi.fn().mockImplementation(async (request: any) => {
    const pending = unresolvedCallIds(request.input ?? []);
    if (pending.length > 0) {
      throw Object.assign(
        new Error(`400 No tool output found for function call ${pending[0]}.`),
        { status: 400 },
      );
    }
    return {
      output: [
        {
          type: 'compaction',
          id: `cmp_${compact.mock.calls.length}`,
          encrypted_content: 'summary',
        },
      ],
      usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
    };
  });
  const session = new OpenAIResponsesCompactionSession({
    underlyingSession: new MemorySession(),
    compactionMode: 'input',
    client: { responses: { compact } } as any,
    shouldTriggerCompaction: () => true,
  });
  const execute = vi.fn(async () => 'sent');
  const agent = new Agent({
    name: 'Mailer',
    model: new ScriptedModel([
      [functionCall('send_email', '{"to":"a@example.com"}', { callId: 'c1' })],
      [assistantMessage('Email sent.')],
      [assistantMessage('You are welcome.')],
    ]),
    tools: [
      tool({
        name: 'send_email',
        description: 'Send an email.',
        parameters: z.object({ to: z.string() }),
        needsApproval: true,
        execute,
      }),
    ],
  });
  return {
    session,
    compact,
    execute,
    agent,
    runner: new Runner({ tracingDisabled: true }),
  };
}

async function runTurn(
  runner: Runner,
  agent: Agent,
  input: string | RunState<any, any>,
  session: OpenAIResponsesCompactionSession,
  stream: boolean,
) {
  if (stream) {
    const result = await runner.run(agent, input as any, {
      session,
      stream: true,
    });
    await result.completed;
    return result;
  }
  return runner.run(agent, input as any, { session });
}

describe('OpenAIResponsesCompactionSession with pending tool approvals', () => {
  for (const stream of [false, true]) {
    for (const decision of ['approve', 'reject'] as const) {
      it(`returns the approval and keeps the session usable (stream: ${stream}, ${decision})`, async () => {
        const { session, compact, execute, agent, runner } = setup();

        const paused = await runTurn(
          runner,
          agent,
          'Email a@example.com',
          session,
          stream,
        );

        expect(paused.interruptions).toHaveLength(1);
        expect(compact).not.toHaveBeenCalled();
        expect(
          (await session.getItems()).some(
            (item) => item.type === 'function_call' && item.callId === 'c1',
          ),
        ).toBe(true);

        const state = await RunState.fromString(agent, paused.state.toString());
        const [interruption] = state.getInterruptions();
        if (decision === 'approve') {
          state.approve(interruption);
        } else {
          state.reject(interruption);
        }
        const resumed = await runTurn(runner, agent, state, session, stream);
        expect(resumed.finalOutput).toBe('Email sent.');
        expect(execute).toHaveBeenCalledTimes(decision === 'approve' ? 1 : 0);

        const next = await runTurn(runner, agent, 'Thanks', session, stream);
        expect(next.finalOutput).toBe('You are welcome.');
        for (const [request] of compact.mock.calls) {
          expect(unresolvedCallIds(request.input ?? [])).toEqual([]);
        }
      });
    }
  }
});
