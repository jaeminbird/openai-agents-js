import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import { Agent, type ToolUseBehavior } from '../src/agent';
import { MemorySession } from '../src/memory/memorySession';
import { Runner } from '../src/run';
import { RunState } from '../src/runState';
import { ScriptedModel, assistantMessage, functionCall } from '../src/testing';
import { tool } from '../src/tool';

type Behavior = 'stop_at' | 'first' | 'custom';

const cases: Array<{
  behavior: Behavior;
  stream: boolean;
  serialized: boolean;
  reject: boolean;
  gatedFirst: boolean;
  withSession: boolean;
}> = [];
for (const behavior of ['stop_at', 'first', 'custom'] as const) {
  for (const stream of [false, true]) {
    for (const serialized of [false, true]) {
      for (const reject of [false, true]) {
        for (const gatedFirst of [false, true]) {
          for (const withSession of [false, true]) {
            cases.push({
              behavior,
              stream,
              serialized,
              reject,
              gatedFirst,
              withSession,
            });
          }
        }
      }
    }
  }
}

describe('toolUseBehavior after an approval resume', () => {
  it.each(cases)(
    'keeps tool results that completed before the interruption (%o)',
    async ({
      behavior,
      stream,
      serialized,
      reject,
      gatedFirst,
      withSession,
    }) => {
      const effects: string[] = [];
      const seen: string[] = [];
      const submitOrder = tool({
        name: 'submit_order',
        description: 'Submit the order.',
        parameters: z.object({}),
        execute: async () => {
          effects.push('submit');
          return 'order submitted';
        },
      });
      const notifyManager = tool({
        name: 'notify_manager',
        description: 'Notify the manager.',
        parameters: z.object({}),
        needsApproval: true,
        execute: async () => {
          effects.push('notify');
          return 'manager notified';
        },
      });
      const calls = [
        functionCall('submit_order', {}, { callId: 'submit' }),
        functionCall('notify_manager', {}, { callId: 'notify' }),
      ];
      if (gatedFirst) {
        calls.reverse();
      }
      // A second scripted turn lets the bug show up as a duplicate submit instead of a script error.
      const model = new ScriptedModel([
        calls,
        [
          functionCall('submit_order', {}, { callId: 'submit-2' }),
          assistantMessage('done'),
        ],
        [assistantMessage('done')],
      ]);
      const toolUseBehavior: ToolUseBehavior =
        behavior === 'stop_at'
          ? { stopAtToolNames: ['submit_order'] }
          : behavior === 'first'
            ? 'stop_on_first_tool'
            : async (_context, results) => {
                seen.push(...results.map((result) => result.tool.name));
                const first = results[0];
                return {
                  isFinalOutput: true,
                  isInterrupted: undefined,
                  finalOutput:
                    first?.type === 'function_output'
                      ? String(first.output)
                      : '',
                };
              };
      const agent = new Agent({
        name: 'orders',
        model,
        tools: [submitOrder, notifyManager],
        toolUseBehavior,
      });
      const runner = new Runner({ tracingDisabled: true });
      const session = withSession ? new MemorySession() : undefined;
      const run = async (input: string | RunState<any, any>) => {
        if (stream) {
          const result = await runner.run(agent, input, {
            stream: true,
            session,
          });
          for await (const _event of result) {
            // drain
          }
          await result.completed;
          return result;
        }
        return await runner.run(agent, input, { session });
      };

      const first = await run('submit an order');
      expect(effects).toEqual(['submit']);
      expect(first.interruptions).toHaveLength(1);

      let state = first.state;
      if (serialized) {
        state = await RunState.fromString(agent, state.toString());
      }
      for (const interruption of state.getInterruptions()) {
        if (reject) {
          state.reject(interruption, { message: 'notification rejected' });
        } else {
          state.approve(interruption);
        }
      }
      const resumed = await run(state);

      let expected = 'order submitted';
      if (gatedFirst && behavior !== 'stop_at') {
        expected = reject ? 'notification rejected' : 'manager notified';
      }
      expect(resumed.interruptions).toHaveLength(0);
      expect(resumed.finalOutput).toBe(expected);
      expect(model.calls).toHaveLength(1);
      expect(effects).toEqual(reject ? ['submit'] : ['submit', 'notify']);
      if (behavior === 'custom') {
        expect(seen.slice(-2)).toEqual(
          gatedFirst
            ? ['notify_manager', 'submit_order']
            : ['submit_order', 'notify_manager'],
        );
      }
      if (session) {
        const outputs = (await session.getItems()).filter(
          (item) => item.type === 'function_call_result',
        );
        expect(
          outputs.map((item) => (item as { callId: string }).callId),
        ).toHaveLength(2);
      }
    },
  );
});
