import { EmitterEvent } from './EmitterEvent';

type NullaryFn<T = any> = () => T;

export type Listener<T extends EmitterEvent<string, any>> = (event: T) => any;

export type EmitterOptions = {
  enabled?: boolean;
};

/**
 * An Event Emitter
 */
export class Emitter<Key extends string, TypeMap extends Record<Key, EmitterEvent<Key, any>>> {
  private readonly listeners: Map<Key, Set<Listener<TypeMap[Key]>>>;

  private enabled: boolean;

  constructor(params: EmitterOptions = { enabled: false }) {
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
    const { enabled = false } = params ?? {};
    this.listeners = new Map();
    this.enabled = enabled;
  }

  enable(): void {
    this.enabled = true;
  }

  disable(): void {
    this.enabled = false;
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  on<TType extends Key, TEvent extends TypeMap[TType]>(
    type: TType,
    listener: Listener<TEvent>,
  ): NullaryFn {
    const listeners = this.listeners.get(type) ?? new Set();

    if (!this.listeners.has(type)) {
      this.listeners.set(type, listeners);
    }

    listeners.add(listener as Listener<TypeMap[Key]>);

    return function unsubscribe() {
      listeners.delete(listener as Listener<TypeMap[Key]>);
    };
  }

  once<TType extends Key, TEvent extends TypeMap[TType]>(
    type: TType,
    listener: Listener<TEvent>,
  ): NullaryFn {
    const listeners = this.listeners.get(type) ?? new Set();

    if (!this.listeners.has(type)) {
      this.listeners.set(type, listeners);
    }

    const wrappedListener: Listener<TEvent> = (event: TEvent) => {
      listeners.delete(wrappedListener as Listener<TypeMap[Key]>);
      listener(event);
    };

    listeners.add(wrappedListener as Listener<TypeMap[Key]>);

    return function unsubscribe() {
      listeners.delete(wrappedListener as Listener<TypeMap[Key]>);
    };
  }

  emit<TType extends Key, TEvent extends TypeMap[TType]>(event: TEvent): TEvent {
    if (!this.enabled) {
      return event;
    }

    const { type } = event;

    for (const listener of this.listeners.get(type) ?? new Set()) {
      listener(event);
    }

    return event;
  }

  eventFrom<TType extends Key, TEvent extends TypeMap[TType]>(
    type: TType,
    payload: Record<string, any>,
  ): TEvent {
    return new EmitterEvent(type, payload) as unknown as TEvent;
  }
}
