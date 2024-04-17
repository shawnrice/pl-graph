export class EmitterEvent<TType extends string, TValue extends Record<string, any>> {
  type: TType;

  value: TValue;

  defaultPrevented: boolean;

  constructor(type: TType, value: TValue) {
    this.type = type;
    this.value = value;
    this.defaultPrevented = false;
  }

  preventDefault(): void {
    // eslint-disable-next-line
    this.defaultPrevented = true;
  }
}
