import { describe, expect, mock, test } from 'bun:test';

import { Emitter } from './Emitter';
import { EmitterEvent } from './EmitterEvent';

describe('Emitter', () => {
  test('it emits', () => {
    const listener = mock();
    const emitter = new Emitter();
    emitter.enable();
    emitter.on('foo', listener);
    emitter.emit(new EmitterEvent('foo', { foo: 'bar' }));
    expect(listener).toHaveBeenCalledWith(new EmitterEvent('foo', { foo: 'bar' }));
  });

  test('it starts disabled', () => {
    const listener = mock();
    const emitter = new Emitter();
    emitter.on('foo', listener);
    emitter.emit(new EmitterEvent('foo', { foo: 'bar' }));
    expect(listener).toHaveBeenCalledTimes(0);
  });

  test('we can use convenience events', () => {
    const listener = mock();
    const emitter = new Emitter({ enabled: true });
    emitter.on('foo', listener);
    emitter.emit(emitter.eventFrom('foo', { foo: 'bar' }));
    expect(listener).toHaveBeenCalledTimes(1);
  });
});
