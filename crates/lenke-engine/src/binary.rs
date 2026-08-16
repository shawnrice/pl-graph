//! The engine's own binary snapshot format — a compact, versioned serialization of
//! a whole graph (schema + nodes + edges + props). The fast/small counterpart to
//! the NDJSON snapshot, for browser-local persistence between sessions.
//!
//! Layout (all integers little-endian; a `str` is a `u32` length then its UTF-8
//! bytes; a `value` is a tag byte then its payload):
//!
//! ```text
//!   magic    "LNKB"          4 bytes
//!   version  u16             the format version (currently 1)
//!   flags    u16             reserved, 0
//!   constraints  u32 n, each (str label, u32 k, k×str key)
//!   required     u32 n, each (str label, str key)
//!   nodes        u32 n, each (str id, u32 l, l×str label, u32 p, p×(str key, value))
//!   edges        u32 n, each (opt-str id, str from, str to, u32 l, l×str label,
//!                             u32 p, p×(str key, value))
//!   value tags   0 null · 1 bool(u8) · 2 num(f64) · 3 str · 4 list(u32 n, n×value)
//!                5 record(u32 n, n×(str,value)) · 6 temporal(str tag, str iso)
//!                7 map(u32 n, n×(value,value))
//! ```
//!
//! The header lets a future bump be recognized: [`from_binary`] REJECTS a version
//! it does not know rather than mis-decoding older/newer data. Decode funnels
//! through the shared [`crate::ndjson::build_store`], so fidelity matches NDJSON.

use crate::ndjson::{build_store, StagedNdjson};
use crate::store::Store;
use crate::value::{make_record, Value};
use std::sync::Arc;

/// Snapshot magic — the first four bytes of every binary snapshot.
const MAGIC: &[u8; 4] = b"LNKB";
/// The current binary format version (bump on any layout change).
const VERSION: u16 = 1;

// ------------------------------------------------------------------- writer ---

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: usize) {
        self.buf.extend_from_slice(&(v as u32).to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len());
        self.buf.extend_from_slice(s.as_bytes());
    }
    fn opt_str(&mut self, s: Option<&str>) {
        match s {
            Some(x) => {
                self.buf.push(1);
                self.str(x);
            }
            None => self.buf.push(0),
        }
    }
    fn value(&mut self, v: &Value) {
        match v {
            Value::Null => self.buf.push(0),
            Value::Bool(b) => {
                self.buf.push(1);
                self.buf.push(u8::from(*b));
            }
            Value::Num(x) => {
                self.buf.push(2);
                self.f64(*x);
            }
            Value::Str(s) => {
                self.buf.push(3);
                self.str(s);
            }
            Value::List(items) => {
                self.buf.push(4);
                self.u32(items.len());
                for it in items.iter() {
                    self.value(it);
                }
            }
            Value::Record(fields) => {
                self.buf.push(5);
                self.u32(fields.len());
                for (k, val) in fields.iter() {
                    self.str(k);
                    self.value(val);
                }
            }
            Value::Temporal(t) => {
                self.buf.push(6);
                self.str(t.tag());
                self.str(&t.format());
            }
            Value::Map(pairs) => {
                self.buf.push(7);
                self.u32(pairs.len());
                for (k, val) in pairs.iter() {
                    self.value(k);
                    self.value(val);
                }
            }
        }
    }
}

// ------------------------------------------------------------------- reader ---

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or("binary: length overflow")?;
        if end > self.buf.len() {
            return Err("binary: unexpected end of input".into());
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn count(&mut self) -> Result<usize, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }
    fn str(&mut self) -> Result<String, String> {
        let n = self.count()?;
        let bytes = self.take(n)?;
        std::str::from_utf8(bytes)
            .map(str::to_string)
            .map_err(|_| "binary: invalid UTF-8 string".to_string())
    }
    fn opt_str(&mut self) -> Result<Option<String>, String> {
        if self.u8()? == 1 {
            Ok(Some(self.str()?))
        } else {
            Ok(None)
        }
    }
    fn str_list(&mut self) -> Result<Vec<String>, String> {
        let n = self.count()?;
        (0..n).map(|_| self.str()).collect()
    }
    fn props(&mut self) -> Result<Vec<(String, Value)>, String> {
        let n = self.count()?;
        (0..n).map(|_| Ok((self.str()?, self.value()?))).collect()
    }
    fn value(&mut self) -> Result<Value, String> {
        Ok(match self.u8()? {
            0 => Value::Null,
            1 => Value::Bool(self.u8()? != 0),
            2 => Value::Num(self.f64()?),
            3 => Value::Str(Arc::from(self.str()?.as_str())),
            4 => {
                let n = self.count()?;
                Value::List((0..n).map(|_| self.value()).collect::<Result<_, _>>()?)
            }
            5 => {
                let n = self.count()?;
                let pairs: Vec<(Arc<str>, Value)> = (0..n)
                    .map(|_| Ok((Arc::from(self.str()?.as_str()), self.value()?)))
                    .collect::<Result<_, String>>()?;
                make_record(pairs)
            }
            6 => {
                let tag = self.str()?;
                let iso = self.str()?;
                crate::temporal::Temporal::parse(&tag, &iso)
                    .map(Value::Temporal)
                    .map_err(|e| format!("binary: bad temporal value: {e}"))?
            }
            7 => {
                let n = self.count()?;
                let pairs: Vec<(Value, Value)> = (0..n)
                    .map(|_| Ok((self.value()?, self.value()?)))
                    .collect::<Result<_, String>>()?;
                Value::Map(Arc::new(pairs))
            }
            t => return Err(format!("binary: unknown value tag {t}")),
        })
    }
}

// -------------------------------------------------------------------- codec ---

/// Serialize a whole graph to the binary snapshot format (see the module docs).
/// Full-fidelity: schema, node/edge external ids, all labels, and all props.
#[must_use]
pub fn to_binary(store: &Store) -> Vec<u8> {
    let mut w = Writer::new();
    w.buf.extend_from_slice(MAGIC);
    w.u16(VERSION);
    w.u16(0); // flags (reserved)

    let mut uniques = store.unique_constraints();
    uniques.sort();
    w.u32(uniques.len());
    for (label, keys) in &uniques {
        w.str(label);
        w.u32(keys.len());
        for k in keys {
            w.str(k);
        }
    }

    let mut required = store.required_constraints();
    required.sort();
    w.u32(required.len());
    for (label, key) in &required {
        w.str(label);
        w.str(key);
    }

    // Live nodes, id order (matches to_ndjson).
    let node_keys = store.prop_keys();
    let live: Vec<u32> = (0..u32::try_from(store.node_count()).unwrap_or(u32::MAX))
        .filter(|&id| store.is_alive(id))
        .collect();
    w.u32(live.len());
    for &id in &live {
        w.str(&store.node_ext_id(id).unwrap_or_default());
        let labels = store.labels_of(id);
        w.u32(labels.len());
        for l in &labels {
            w.str(l);
        }
        let present: Vec<&String> = node_keys.iter().filter(|k| store.has_prop(id, k)).collect();
        w.u32(present.len());
        for k in present {
            w.str(k);
            w.value(&store.prop(id, k));
        }
    }

    // Edges via out-adjacency (matches to_ndjson's enumeration).
    let edge_keys = store.edge_prop_keys();
    let mut edges: Vec<(u32, u32, u32)> = Vec::new(); // (eid, from, to)
    for &from in &live {
        for a in store.out(from) {
            edges.push((a.eid, from, a.nbr));
        }
    }
    w.u32(edges.len());
    for (eid, from, to) in &edges {
        w.opt_str(store.edge_ext_id(*eid).as_deref());
        w.str(&store.node_ext_id(*from).unwrap_or_default());
        w.str(&store.node_ext_id(*to).unwrap_or_default());
        let labels = store.edge_labels_of(*eid);
        w.u32(labels.len());
        for l in &labels {
            w.str(l);
        }
        let present: Vec<&String> = edge_keys
            .iter()
            .filter(|k| store.has_edge_prop(*eid, k))
            .collect();
        w.u32(present.len());
        for k in present {
            w.str(k);
            w.value(&store.edge_prop(*eid, k));
        }
    }
    w.buf
}

/// Load a graph from the binary snapshot format. Rejects a bad magic or a version
/// this build does not know (rather than mis-decoding).
pub fn from_binary(bytes: &[u8]) -> Result<Store, String> {
    let mut r = Reader::new(bytes);
    if r.take(4)? != MAGIC {
        return Err("binary: bad magic (not a lenke binary snapshot)".into());
    }
    let version = r.u16()?;
    if version != VERSION {
        return Err(format!(
            "binary: unsupported snapshot version {version} (this build reads version {VERSION})"
        ));
    }
    let _flags = r.u16()?;

    let nc = r.count()?;
    let mut constraints = Vec::with_capacity(nc);
    for _ in 0..nc {
        let label = r.str()?;
        constraints.push((label, r.str_list()?));
    }
    let rq = r.count()?;
    let mut required = Vec::with_capacity(rq);
    for _ in 0..rq {
        required.push((r.str()?, r.str()?));
    }
    let nn = r.count()?;
    let mut nodes = Vec::with_capacity(nn);
    for _ in 0..nn {
        let ext = r.str()?;
        let labels = r.str_list()?;
        nodes.push((ext, labels, r.props()?));
    }
    let ne = r.count()?;
    let mut edges = Vec::with_capacity(ne);
    for _ in 0..ne {
        let ext = r.opt_str()?;
        let from = r.str()?;
        let to = r.str()?;
        let labels = r.str_list()?;
        edges.push((from, to, ext, labels, r.props()?));
    }

    build_store(StagedNdjson {
        constraints,
        required,
        nodes,
        edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndjson::{from_ndjson, to_ndjson};

    fn fixture() -> Store {
        from_ndjson(
            "{\"schema\":\"unique\",\"label\":\"P\",\"keys\":[\"email\"]}\n\
             {\"id\":\"1\",\"labels\":[\"P\"],\"props\":{\"email\":\"a@x\",\"age\":30}}\n\
             {\"id\":\"2\",\"labels\":[\"P\"],\"props\":{\"email\":\"b@x\"}}\n\
             {\"from\":\"1\",\"to\":\"2\",\"id\":\"e0\",\"type\":\"KNOWS\",\"props\":{\"since\":2020}}\n",
        )
        .unwrap()
    }

    #[test]
    fn binary_round_trips_via_ndjson_equality() {
        let st = fixture();
        let bytes = to_binary(&st);
        assert_eq!(&bytes[0..4], MAGIC);
        let back = from_binary(&bytes).unwrap();
        // Same logical content: the NDJSON dump of the reload matches the original's.
        assert_eq!(to_ndjson(&back), to_ndjson(&st));
        // The unique constraint survived (schema round-trips).
        assert_eq!(back.unique_constraints(), st.unique_constraints());
    }

    #[test]
    fn empty_graph_round_trips() {
        let st = Store::default();
        let back = from_binary(&to_binary(&st)).unwrap();
        assert_eq!(back.node_count(), 0);
        assert_eq!(back.edge_count(), 0);
    }

    #[test]
    fn bad_magic_is_rejected() {
        // Store is not Debug, so unwrap_err() is unavailable — take the Err directly.
        assert!(from_binary(b"XXXX\x01\x00\x00\x00")
            .err()
            .unwrap()
            .contains("magic"));
    }

    #[test]
    fn unknown_version_is_rejected_not_misdecoded() {
        let mut bytes = to_binary(&fixture());
        bytes[4] = 0xFF; // corrupt the version field
        bytes[5] = 0xFF;
        assert!(from_binary(&bytes).err().unwrap().contains("version"));
    }

    #[test]
    fn truncated_input_errors_cleanly() {
        let bytes = to_binary(&fixture());
        // Chop the body: header intact, data cut short — must Err, not panic.
        assert!(from_binary(&bytes[..bytes.len() - 5]).is_err());
    }
}
