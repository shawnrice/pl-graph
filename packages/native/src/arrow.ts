/**
 * Real **Apache Arrow IPC** egress for lenke query results — the interop bridge
 * to DuckDB / Polars / pandas / any Arrow consumer, with **zero runtime
 * dependencies**.
 *
 * {@link RustGraph.queryArrow} returns lenke's compact in-process columnar blob
 * (`ARW1`); its typed buffers already ARE Arrow's columnar layout (little-endian,
 * LSB-first validity bitmap, `i32` Utf8 offsets). This module frames those exact
 * buffers as standard Arrow IPC — a small hand-written flatbuffer encoder emits the
 * `Schema` + `RecordBatch` (and, for the file layout, `Footer`) messages, so the
 * bytes are what `pyarrow.ipc`, `polars.read_ipc`, and DuckDB read. Nothing is
 * copied or re-parsed beyond concatenating the already-Arrow buffers.
 *
 * (The output is verified byte-for-byte against `apache-arrow`'s reference decoder
 * in `arrow.test.ts`, where it's a dev-only dependency — it never ships.)
 */
import { ErrorCode, LenkeError } from '@lenke/errors';

// ARW1 column type tags (mirrors crates/lenke-core/src/arrow.rs and graph.ts).
const ARW_FLOAT64 = 1;
const ARW_BOOL = 2;
const ARW_UTF8 = 3;
const ARW_FIXED_LIST = 4; // FixedSizeList<Float64>[dim]; dim rides buf2_len
const ARW_STRUCT = 5; // Struct<field: type, …>; child count rides buf2_len, children follow in pre-order

const HEADER_LEN = 24;
const COLDESC_LEN = 40;

// Arrow flatbuffer enum values we emit.
const METADATA_V5 = 4; // MetadataVersion.V5
const MSG_SCHEMA = 1; // MessageHeader.Schema
const MSG_RECORD_BATCH = 3; // MessageHeader.RecordBatch
const TYPE_FLOATINGPOINT = 3; // Type.FloatingPoint
const TYPE_UTF8 = 5; // Type.Utf8
const TYPE_BOOL = 6; // Type.Bool
const TYPE_STRUCT = 13; // Type.Struct_
const TYPE_FIXEDSIZELIST = 16; // Type.FixedSizeList
const PRECISION_DOUBLE = 2; // Precision.DOUBLE

/**
 * A minimal FlatBuffers builder — just the primitives Arrow's `Schema` /
 * `RecordBatch` / `Footer` need (tables + vtables, offset/struct vectors, strings,
 * inline scalars). Builds back-to-front like the reference implementation: values
 * are written toward the front of `buf`, offsets are measured from the end.
 */
class FlatBufferBuilder {
  private buf: Uint8Array;
  private space: number;
  private minalign = 1;
  private vtable: number[] = [];
  private objectStart = 0;

  constructor(initial = 1024) {
    this.buf = new Uint8Array(initial);
    this.space = initial;
  }

  private view(): DataView {
    return new DataView(this.buf.buffer);
  }

  /** Bytes written so far, measured from the end of the buffer. */
  offset(): number {
    return this.buf.length - this.space;
  }

  private grow(): void {
    const old = this.buf;
    const oldLen = old.length;
    this.buf = new Uint8Array(oldLen * 2);
    this.buf.set(old, oldLen);
    this.space += oldLen;
  }

  /** Align so that after `additional` more bytes a `size`-wide value lands aligned. */
  private prep(size: number, additional: number): void {
    if (size > this.minalign) {
      this.minalign = size;
    }

    const alignSize = (~(this.offset() + additional) + 1) & (size - 1);

    while (this.space < alignSize + size + additional) {
      this.grow();
    }

    for (let i = 0; i < alignSize; i += 1) {
      this.buf[(this.space -= 1)] = 0;
    }
  }

  private ensure(n: number): void {
    while (this.space < n) {
      this.grow();
    }
  }

  /** Write `n` zero bytes with no alignment (struct interior padding). */
  pad(n: number): void {
    this.ensure(n);

    for (let i = 0; i < n; i += 1) {
      this.buf[(this.space -= 1)] = 0;
    }
  }

  addInt8(v: number): void {
    this.prep(1, 0);
    this.buf[(this.space -= 1)] = v & 0xff;
  }

  addInt16(v: number): void {
    this.prep(2, 0);
    this.space -= 2;
    this.view().setInt16(this.space, v, true);
  }

  addInt32(v: number): void {
    this.prep(4, 0);
    this.space -= 4;
    this.view().setInt32(this.space, v, true);
  }

  addInt64(v: bigint): void {
    this.prep(8, 0);
    this.space -= 8;
    this.view().setBigInt64(this.space, v, true);
  }

  /** A forward uoffset to a previously-built object at rev-offset `off`. */
  addOffset(off: number): void {
    this.prep(4, 0);
    const value = this.offset() - off + 4;
    this.space -= 4;
    this.view().setInt32(this.space, value, true);
  }

  // ── strings & vectors ──────────────────────────────────────────────────────
  createString(s: string): number {
    const bytes = new TextEncoder().encode(s);
    this.addInt8(0); // trailing null
    this.prep(4, bytes.length);
    this.ensure(bytes.length);
    this.space -= bytes.length;
    this.buf.set(bytes, this.space);
    this.addInt32(bytes.length);

    return this.offset();
  }

  private startVector(elemSize: number, numElems: number, alignment: number): void {
    this.prep(4, elemSize * numElems);
    this.prep(alignment, elemSize * numElems);
  }

  private endVector(numElems: number): number {
    this.addInt32(numElems);

    return this.offset();
  }

  /** A vector of table/string offsets (built in reverse so it reads forward). */
  offsetVector(offsets: readonly number[]): number {
    this.startVector(4, offsets.length, 4);

    for (let i = offsets.length - 1; i >= 0; i -= 1) {
      this.addOffset(offsets[i]);
    }

    return this.endVector(offsets.length);
  }

  /** A vector of 16-byte `{a, b}` i64 structs (FieldNode / Buffer). */
  structVector16(structs: readonly [bigint, bigint][]): number {
    this.startVector(16, structs.length, 8);

    for (let i = structs.length - 1; i >= 0; i -= 1) {
      // Written back-to-front → forward layout is [a, b].
      this.addInt64(structs[i][1]);
      this.addInt64(structs[i][0]);
    }

    return this.endVector(structs.length);
  }

  /** A vector of 24-byte `Block` structs: offset:i64 @0, metaDataLength:i32 @8
   * (+4 pad), bodyLength:i64 @16. */
  blockVector(
    blocks: readonly { offset: number; metaDataLength: number; bodyLength: number }[],
  ): number {
    this.startVector(24, blocks.length, 8);

    for (let i = blocks.length - 1; i >= 0; i -= 1) {
      const blk = blocks[i];
      // Back-to-front → forward [offset, metaDataLength, pad(4), bodyLength].
      this.addInt64(BigInt(blk.bodyLength));
      this.pad(4);
      this.addInt32(blk.metaDataLength);
      this.addInt64(BigInt(blk.offset));
    }

    return this.endVector(blocks.length);
  }

  // ── tables ─────────────────────────────────────────────────────────────────
  startObject(numfields: number): void {
    this.vtable = new Array<number>(numfields).fill(0);
    this.objectStart = this.offset();
  }

  private slot(voffset: number): void {
    this.vtable[voffset] = this.offset();
  }

  addFieldInt8(voffset: number, value: number, def: number): void {
    if (value !== def) {
      this.addInt8(value);
      this.slot(voffset);
    }
  }

  addFieldInt16(voffset: number, value: number, def: number): void {
    if (value !== def) {
      this.addInt16(value);
      this.slot(voffset);
    }
  }

  addFieldInt32(voffset: number, value: number, def: number): void {
    if (value !== def) {
      this.addInt32(value);
      this.slot(voffset);
    }
  }

  addFieldInt64(voffset: number, value: bigint, def: bigint): void {
    if (value !== def) {
      this.addInt64(value);
      this.slot(voffset);
    }
  }

  addFieldOffset(voffset: number, value: number): void {
    if (value !== 0) {
      this.addOffset(value);
      this.slot(voffset);
    }
  }

  endObject(): number {
    this.addInt32(0); // soffset placeholder
    const vtableloc = this.offset();

    let i = this.vtable.length - 1;

    for (; i >= 0 && this.vtable[i] === 0; i -= 1) {
      // trim trailing absent fields
    }

    const trimmed = i + 1;

    for (; i >= 0; i -= 1) {
      this.addInt16(this.vtable[i] !== 0 ? vtableloc - this.vtable[i] : 0);
    }

    this.addInt16(vtableloc - this.objectStart); // object size
    this.addInt16((trimmed + 2) * 2); // vtable byte size (incl the 2 standard shorts)

    // Point the object's soffset at the vtable we just wrote.
    this.view().setInt32(this.buf.length - vtableloc, this.offset() - vtableloc, true);

    return vtableloc;
  }

  finish(root: number): Uint8Array {
    this.prep(this.minalign, 4);
    this.addOffset(root);

    return this.buf.slice(this.space);
  }
}

/** One column's ARW1 view: its Arrow type + physical buffers (validity/data). A
 * `Struct` carries only its validity buffer and its typed `children`. */
type Arw1Column = {
  name: string;
  type: number;
  nullCount: number;
  /** FixedSizeList list size (0 otherwise); a FixedList adds a child field-node. */
  dim: number;
  buffers: Uint8Array[]; // Arrow buffer order (validity, then values / offsets+data)
  children: Arw1Column[]; // Struct field columns (empty for every other type)
};

/** Parse an ARW1 blob into per-column name/type/null-count + Arrow buffers. A
 * struct's children ride as pre-order descriptors, so the reader recurses; the
 * header's `ncols` counts only top-level columns. */
const parseArw1 = (blob: Uint8Array): { nrows: number; columns: Arw1Column[] } => {
  const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const td = new TextDecoder();

  if (blob.length < HEADER_LEN || td.decode(blob.subarray(0, 4)) !== 'ARW1') {
    throw new LenkeError('lenke: not an ARW1 arrow blob', { code: ErrorCode.Ffi });
  }

  const nrows = Number(dv.getBigUint64(8, true));
  const ncols = Number(dv.getBigUint64(16, true));
  const slice = (off: number, len: number): Uint8Array => blob.subarray(off, off + len);

  // Read descriptor `cursor.i` into a column, advancing past it and (for a struct)
  // its children in pre-order.
  const cursor = { i: 0 };
  const readCol = (): Arw1Column => {
    const d = HEADER_LEN + cursor.i * COLDESC_LEN;
    cursor.i += 1;
    const type = dv.getUint32(d, true);
    const nullCount = dv.getUint32(d + 4, true);
    const nameOff = dv.getUint32(d + 8, true);
    const nameLen = dv.getUint32(d + 12, true);
    const validityOff = dv.getUint32(d + 16, true);
    const validityLen = dv.getUint32(d + 20, true);
    const buf1Off = dv.getUint32(d + 24, true);
    const buf1Len = dv.getUint32(d + 28, true);
    const buf2Off = dv.getUint32(d + 32, true);
    const buf2Len = dv.getUint32(d + 36, true);
    const name = td.decode(blob.subarray(nameOff, nameOff + nameLen));

    // Arrow buffer order: validity, then values (Float64/Bool) or offsets+data (Utf8).
    // A no-null column's validity is a length-0 buffer (readers treat it as all-valid).
    // FixedSizeList: [list validity, child validity (empty), child values (buf1)];
    // its `dim` rides `buf2_len`. Struct: [validity] only; `buf2_len` = child count.
    let dim = 0;
    let buffers: Uint8Array[];
    const children: Arw1Column[] = [];

    if (type === ARW_UTF8) {
      buffers = [slice(validityOff, validityLen), slice(buf1Off, buf1Len), slice(buf2Off, buf2Len)];
    } else if (type === ARW_FIXED_LIST) {
      dim = buf2Len;
      buffers = [slice(validityOff, validityLen), new Uint8Array(0), slice(buf1Off, buf1Len)];
    } else if (type === ARW_STRUCT) {
      buffers = [slice(validityOff, validityLen)];

      for (let k = 0; k < buf2Len; k += 1) {
        children.push(readCol());
      }
    } else {
      buffers = [slice(validityOff, validityLen), slice(buf1Off, buf1Len)];
    }

    return { name, type, nullCount, dim, buffers, children };
  };

  const columns: Arw1Column[] = [];

  for (let c = 0; c < ncols; c += 1) {
    columns.push(readCol());
  }

  return { nrows, columns };
};

/** Concatenate the Arrow buffers into a RecordBatch body (each 8-aligned) in
 * depth-first order (a struct's validity, then each child's buffers), and record
 * the `{offset, length}` for every buffer plus the body's total length. */
const assembleBody = (columns: Arw1Column[]): { body: Uint8Array; buffers: [bigint, bigint][] } => {
  const parts: Uint8Array[] = [];
  const buffers: [bigint, bigint][] = [];
  let pos = 0;
  const align = (): void => {
    const p = (8 - (pos % 8)) % 8;

    if (p > 0) {
      parts.push(new Uint8Array(p));
      pos += p;
    }
  };
  const pushBuffers = (col: Arw1Column): void => {
    for (const b of col.buffers) {
      align(); // every Arrow buffer starts on an 8-byte boundary
      buffers.push([BigInt(pos), BigInt(b.length)]);
      parts.push(b);
      pos += b.length;
    }

    for (const c of col.children) {
      pushBuffers(c);
    }
  };

  for (const col of columns) {
    pushBuffers(col);
  }

  align(); // pad the whole body to a multiple of 8

  const body = new Uint8Array(pos);
  let o = 0;

  for (const p of parts) {
    body.set(p, o);
    o += p.length;
  }

  return { body, buffers };
};

/** Build the Arrow `Field` sub-table for one column and return its offset. */
const buildField = (b: FlatBufferBuilder, col: Arw1Column, emptyChildren: number): number => {
  const nameOff = b.createString(col.name);

  let typeType: number;
  let typeOff: number;
  let children = emptyChildren;

  if (col.type === ARW_FLOAT64) {
    b.startObject(1);
    b.addFieldInt16(0, PRECISION_DOUBLE, 0);
    typeOff = b.endObject();
    typeType = TYPE_FLOATINGPOINT;
  } else if (col.type === ARW_BOOL) {
    b.startObject(0);
    typeOff = b.endObject();
    typeType = TYPE_BOOL;
  } else if (col.type === ARW_UTF8) {
    b.startObject(0);
    typeOff = b.endObject();
    typeType = TYPE_UTF8;
  } else if (col.type === ARW_STRUCT) {
    // Nested types build inner-first: each child Field, then the children vector,
    // then this struct's (empty) type table.
    const childFields = col.children.map((c) => buildField(b, c, emptyChildren));
    children = b.offsetVector(childFields);
    b.startObject(0); // Struct_ has no type-table fields
    typeOff = b.endObject();
    typeType = TYPE_STRUCT;
  } else if (col.type === ARW_FIXED_LIST) {
    // The child `item: Float64` field (nested types build inner-first).
    const childName = b.createString('item');
    b.startObject(1);
    b.addFieldInt16(0, PRECISION_DOUBLE, 0);
    const childType = b.endObject();
    b.startObject(7);
    b.addFieldOffset(0, childName);
    b.addFieldInt8(1, 1, 0); // nullable
    b.addFieldInt8(2, TYPE_FLOATINGPOINT, 0);
    b.addFieldOffset(3, childType);
    b.addFieldOffset(5, emptyChildren);
    const childField = b.endObject();
    children = b.offsetVector([childField]);
    // FixedSizeList type table: `listSize: int` (field 0).
    b.startObject(1);
    b.addFieldInt32(0, col.dim, 0);
    typeOff = b.endObject();
    typeType = TYPE_FIXEDSIZELIST;
  } else {
    throw new LenkeError(`lenke: arrow column '${col.name}' has unknown type ${col.type}`, {
      code: ErrorCode.Ffi,
    });
  }

  b.startObject(7);
  b.addFieldOffset(0, nameOff); // name
  b.addFieldInt8(1, 1, 0); // nullable = true
  b.addFieldInt8(2, typeType, 0); // type_type (union discriminant)
  b.addFieldOffset(3, typeOff); // type (union value)
  b.addFieldOffset(5, children); // children ([item] for FixedSizeList, else empty)

  return b.endObject();
};

/** Build the Arrow `Schema` sub-table for these columns and return its offset. */
const buildSchema = (b: FlatBufferBuilder, columns: Arw1Column[]): number => {
  const emptyChildren = b.offsetVector([]);
  const fields = columns.map((col) => buildField(b, col, emptyChildren));
  const fieldsVec = b.offsetVector(fields);

  b.startObject(4);
  b.addFieldOffset(1, fieldsVec); // fields (endianness defaults to Little)

  return b.endObject();
};

/** A finished, framed `Schema` IPC message (metadata only, no body). */
const schemaMessage = (columns: Arw1Column[]): Uint8Array => {
  const b = new FlatBufferBuilder();
  const schemaOff = buildSchema(b, columns);

  b.startObject(5);
  b.addFieldInt16(0, METADATA_V5, 0);
  b.addFieldInt8(1, MSG_SCHEMA, 0);
  b.addFieldOffset(2, schemaOff);

  return encapsulate(b.finish(b.endObject()), null);
};

/** A finished, framed `RecordBatch` IPC message with its data body. */
const recordBatchMessage = (
  columns: Arw1Column[],
  nrows: number,
  buffers: [bigint, bigint][],
  body: Uint8Array,
): Uint8Array => {
  const b = new FlatBufferBuilder();
  // One field-node per field in depth-first pre-order: a FixedSizeList adds a
  // second node for its `item` child; a Struct adds a node per child (recursively).
  const nodes: [bigint, bigint][] = [];
  const pushNodes = (c: Arw1Column): void => {
    nodes.push([BigInt(nrows), BigInt(c.nullCount)]);

    if (c.type === ARW_FIXED_LIST) {
      nodes.push([BigInt(nrows * c.dim), 0n]);
    } else if (c.type === ARW_STRUCT) {
      for (const child of c.children) {
        pushNodes(child);
      }
    }
  };

  for (const c of columns) {
    pushNodes(c);
  }

  const buffersVec = b.structVector16(buffers);
  const nodesVec = b.structVector16(nodes);

  b.startObject(5);
  b.addFieldInt64(0, BigInt(nrows), 0n); // length
  b.addFieldOffset(1, nodesVec);
  b.addFieldOffset(2, buffersVec);
  const rbOff = b.endObject();

  b.startObject(5);
  b.addFieldInt16(0, METADATA_V5, 0);
  b.addFieldInt8(1, MSG_RECORD_BATCH, 0);
  b.addFieldOffset(2, rbOff);
  b.addFieldInt64(3, BigInt(body.length), 0n); // bodyLength

  return encapsulate(b.finish(b.endObject()), body);
};

/** The file-layout `Footer`: the schema again + a Block per record batch. */
const buildFooter = (
  columns: Arw1Column[],
  blocks: { offset: number; metaDataLength: number; bodyLength: number }[],
): Uint8Array => {
  const b = new FlatBufferBuilder();
  const schemaOff = buildSchema(b, columns);
  const recordBatchesVec = b.blockVector(blocks);

  // Footer: version(0), schema(1), dictionaries(2), recordBatches(3), custom_metadata(4).
  b.startObject(5);
  b.addFieldInt16(0, METADATA_V5, 0);
  b.addFieldOffset(1, schemaOff);
  b.addFieldOffset(3, recordBatchesVec);

  return b.finish(b.endObject());
};

/** Wrap a flatbuffer message in the IPC encapsulation (continuation + size +
 * padding, then the body). Body offset lands on an 8-byte boundary. */
const encapsulate = (meta: Uint8Array, body: Uint8Array | null): Uint8Array => {
  const metaPadded = (meta.length + 7) & ~7;
  const out = new Uint8Array(8 + metaPadded + (body ? body.length : 0));
  const dv = new DataView(out.buffer);
  dv.setUint32(0, 0xffffffff, true); // continuation marker
  dv.setInt32(4, metaPadded, true); // metadata size (incl padding)
  out.set(meta, 8);

  if (body) {
    out.set(body, 8 + metaPadded);
  }

  return out;
};

const concat = (parts: Uint8Array[]): Uint8Array => {
  const total = parts.reduce((s, p) => s + p.length, 0);
  const out = new Uint8Array(total);
  let o = 0;

  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }

  return out;
};

/**
 * Serialize an `ARW1` columnar blob (from {@link RustGraph.queryArrow}) to standard
 * **Arrow IPC** bytes — the flatbuffer-framed format DuckDB / Polars / pandas read.
 * `format` picks the IPC stream layout (default — `pyarrow.ipc.open_stream`,
 * `polars.read_ipc_stream`) or the file / Feather-v2 layout (`pandas.read_feather`,
 * `polars.read_ipc`, `pyarrow.ipc.open_file`). Feed a query straight through:
 * `toArrowIPC(graph.queryArrow('MATCH (n:P) RETURN n.name, n.age'))`. Float64, Bool,
 * Utf8, FixedSizeList, and Struct (record) columns are supported (the tags ARW1
 * emits); nulls carry through the validity bitmap. No runtime dependencies.
 */
export const toArrowIPC = (blob: Uint8Array, format: 'stream' | 'file' = 'stream'): Uint8Array => {
  const { nrows, columns } = parseArw1(blob);
  const { body, buffers } = assembleBody(columns);
  const schemaMsg = schemaMessage(columns);
  const rbMsg = recordBatchMessage(columns, nrows, buffers, body);

  if (format === 'stream') {
    // End-of-stream marker: continuation (0xFFFFFFFF) + zero-length metadata.
    const eos = new Uint8Array(8);
    new DataView(eos.buffer).setUint32(0, 0xffffffff, true);

    return concat([schemaMsg, rbMsg, eos]);
  }

  const magic = new TextEncoder().encode('ARROW1\0\0'); // 8-byte, padded
  const rbOffset = magic.length + schemaMsg.length;
  const metaDataLength = rbMsg.length - body.length; // 8 + padded metadata
  const footer = buildFooter(columns, [
    { offset: rbOffset, metaDataLength, bodyLength: body.length },
  ]);
  const footerLen = new Uint8Array(4);
  new DataView(footerLen.buffer).setInt32(0, footer.length, true);

  return concat([magic, schemaMsg, rbMsg, footer, footerLen, new TextEncoder().encode('ARROW1')]);
};
