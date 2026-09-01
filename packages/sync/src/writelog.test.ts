import { describe, expect, test } from 'bun:test';

import { createWriteLog } from './writelog.js';

const w = (text: string) => ({ text });

describe('WriteLog (CDC op log)', () => {
  test('append assigns monotonic seq and notifies subscribers in order', () => {
    const log = createWriteLog();
    const seen: number[] = [];
    log.subscribe((e) => seen.push(e.seq));

    expect(log.append('c', w('a'))).toBe(1);
    expect(log.append('c', w('b'))).toBe(2);
    expect(seen).toEqual([1, 2]);
    expect(log.head()).toBe(2);
  });

  test('since replays the tail, [] when current, null when fallen off the ring', () => {
    const log = createWriteLog({ capacity: 3 });

    for (const t of ['a', 'b', 'c', 'd', 'e']) {
      log.append('c', w(t));
    } // seq 1..5; ring holds 3..5

    expect(log.since(5)).toEqual([]); // current
    expect(log.since(3)?.map((e) => e.write.text)).toEqual(['d', 'e']); // 4,5
    expect(log.since(2)?.map((e) => e.write.text)).toEqual(['c', 'd', 'e']); // 3,4,5 (all retained)
    expect(log.since(1)).toBeNull(); // seq 2 dropped → gap → cold boot
    expect(log.since(0)).toBeNull();
  });

  test('since(from > seq) is null (a regressed log forces resync, never silently []) ', () => {
    // A caller ahead of the log means the log RESET (server restart / failover /
    // fresh node): seq dropped back below the client cursor. Treating it as current
    // ([]) would silently drop every future write; it must force a cold-boot resync.
    const log = createWriteLog();
    log.append('c', w('a'));
    log.append('c', w('b')); // seq = 2

    expect(log.since(2)).toEqual([]); // exactly current
    expect(log.since(5)).toBeNull(); // ahead of seq → regressed → resync
  });

  test('since(0) from the start when nothing has dropped', () => {
    const log = createWriteLog();
    log.append('c', w('a'));
    log.append('c', w('b'));
    expect(log.since(0)?.map((e) => e.write.text)).toEqual(['a', 'b']);
  });

  test('origin is carried (distinct per participant) for skip-your-own-echo', () => {
    const log = createWriteLog();
    const a = 'client-a';
    const b = 'client-b';

    log.append(a, w('from-a'));
    log.append(b, w('from-b'));
    expect(log.since(0)?.map((e) => e.origin)).toEqual([a, b]);
  });

  test('unsubscribe stops delivery', () => {
    const log = createWriteLog();
    const seen: number[] = [];
    const off = log.subscribe((e) => seen.push(e.seq));

    log.append('c', w('a'));
    off();
    log.append('c', w('b'));
    expect(seen).toEqual([1]);
  });
});
