//! I2b — Arrow ARW1 egress byte-parity with lenke-core.
//!
//! The strongest possible verifier for the columnar blob: build the SAME logical
//! table in `lenke-engine` and `lenke-core` (a dev-dependency) and assert the
//! `to_arrow` bytes are IDENTICAL — no apache-arrow round-trip needed, since
//! agreement with the reference encoder (which the TS/apache-arrow verifier
//! already checks) is exactly what parity means. Covers the scalar types and the
//! nested `FixedSizeList<Float64>` and `Struct` columns this slice added.
//!
//! Every shape ORDERs BY a unique, projected key so both engines materialize rows
//! in the same order — the blob is row-order-sensitive, and row order is otherwise
//! unspecified. The sort key is projected because lenke-engine scopes ORDER BY to
//! output columns (a documented divergence; see the J1 conformance suite).

use lenke_core::gql::eval::Params as CoreParams;

const ENGINE_ND: &str = r#"{"id":0,"labels":["P"],"props":{"name":"alice","age":30,"ok":true}}
{"id":1,"labels":["P"],"props":{"name":"bob","age":25,"ok":false}}
{"id":2,"labels":["P"],"props":{"name":"carol","age":45,"ok":true}}
"#;

const CORE_ND: &str = r#"{"type":"node","id":"0","labels":["P"],"properties":{"name":"alice","age":30,"ok":true}}
{"type":"node","id":"1","labels":["P"],"properties":{"name":"bob","age":25,"ok":false}}
{"type":"node","id":"2","labels":["P"],"properties":{"name":"carol","age":45,"ok":true}}
"#;

fn engine_blob(q: &str) -> Vec<u8> {
    let store = lenke_engine::ndjson::from_ndjson(ENGINE_ND).expect("engine load");
    let rows = lenke_engine::exec::run(
        &lenke_engine::gql::parse(q).unwrap_or_else(|e| panic!("engine parse `{q}`: {e}")),
        &store,
    );
    lenke_engine::arrow::to_arrow(&rows)
}

fn core_blob(q: &str) -> Vec<u8> {
    let mut g = lenke_core::ndjson::decode(CORE_ND).expect("core load");
    let rs = lenke_core::gql::prepare(q)
        .unwrap_or_else(|e| panic!("core parse `{q}`: {e}"))
        .execute(&mut g, &CoreParams::new())
        .unwrap_or_else(|e| panic!("core exec `{q}`: {e:?}"));
    lenke_core::arrow::to_arrow(&rs)
}

fn assert_blob_parity(q: &str) {
    let e = engine_blob(q);
    let c = core_blob(q);
    assert_eq!(
        e.len(),
        c.len(),
        "blob length differs for `{q}`: engine {} vs core {}",
        e.len(),
        c.len()
    );
    assert_eq!(e, c, "blob bytes differ for `{q}`");
}

fn engine_ipc(q: &str, file: bool) -> Vec<u8> {
    let store = lenke_engine::ndjson::from_ndjson(ENGINE_ND).expect("engine load");
    let rows = lenke_engine::exec::run(
        &lenke_engine::gql::parse(q).unwrap_or_else(|e| panic!("engine parse `{q}`: {e}")),
        &store,
    );
    lenke_engine::arrow::to_arrow_ipc(&rows, file)
}

fn core_ipc(q: &str, file: bool) -> Vec<u8> {
    let mut g = lenke_core::ndjson::decode(CORE_ND).expect("core load");
    let rs = lenke_core::gql::prepare(q)
        .unwrap_or_else(|e| panic!("core parse `{q}`: {e}"))
        .execute(&mut g, &CoreParams::new())
        .unwrap_or_else(|e| panic!("core exec `{q}`: {e:?}"));
    lenke_core::arrow::to_arrow_ipc(&rs, file)
}

/// Assert IPC byte-parity in BOTH the stream and file (Feather-v2) layouts.
fn assert_ipc_parity(q: &str) {
    for file in [false, true] {
        let e = engine_ipc(q, file);
        let c = core_ipc(q, file);
        assert_eq!(
            e.len(),
            c.len(),
            "IPC (file={file}) length differs for `{q}`: engine {} vs core {}",
            e.len(),
            c.len()
        );
        assert_eq!(e, c, "IPC (file={file}) bytes differ for `{q}`");
    }
}

#[test]
fn scalar_columns_match_core() {
    // Float64, Utf8, Bool, and a Utf8 column with a null (a missing prop).
    assert_blob_parity("MATCH (n:P) RETURN n.name AS name, n.age AS age, n.ok AS ok ORDER BY name");
}

#[test]
fn fixed_size_list_column_matches_core() {
    // Each cell is an all-numeric list of the same length → FixedSizeList<Float64>[2].
    assert_blob_parity("MATCH (n:P) RETURN n.name AS name, [n.age, n.age] AS pair ORDER BY name");
}

#[test]
fn struct_column_matches_core() {
    // A record literal → an Arrow Struct with sorted child fields (a: Float64,
    // z: Utf8). lenke-core carries this as a result-side Map; both flatten to the
    // same Struct descriptors + child columns.
    assert_blob_parity(
        "MATCH (n:P) RETURN n.name AS name, {a: n.age, z: n.name} AS rec ORDER BY name",
    );
}

#[test]
fn nested_struct_with_list_child_matches_core() {
    // A struct whose child is itself a fixed numeric list → nested descriptors in
    // pre-order (struct, then its child).
    assert_blob_parity(
        "MATCH (n:P) RETURN n.name AS name, {pair: [n.age, n.age]} AS rec ORDER BY name",
    );
}

// ── I2b-2: Arrow-IPC flatbuffer envelope (stream + file layouts) ──────────────

#[test]
fn ipc_scalar_matches_core() {
    assert_ipc_parity("MATCH (n:P) RETURN n.name AS name, n.age AS age, n.ok AS ok ORDER BY name");
}

#[test]
fn ipc_fixed_size_list_matches_core() {
    assert_ipc_parity("MATCH (n:P) RETURN n.name AS name, [n.age, n.age] AS pair ORDER BY name");
}

#[test]
fn ipc_struct_matches_core() {
    assert_ipc_parity(
        "MATCH (n:P) RETURN n.name AS name, {a: n.age, z: n.name} AS rec ORDER BY name",
    );
}

#[test]
fn ipc_nested_struct_with_list_matches_core() {
    assert_ipc_parity(
        "MATCH (n:P) RETURN n.name AS name, {pair: [n.age, n.age]} AS rec ORDER BY name",
    );
}

/// The stream and file layouts differ in the expected ways: the file layout wraps
/// the same messages in the `ARROW1` magic + trailing footer.
#[test]
fn ipc_envelope_shape() {
    let q = "MATCH (n:P) RETURN n.name AS name, n.age AS age ORDER BY name";
    let stream = engine_ipc(q, false);
    let file = engine_ipc(q, true);
    // Stream starts with a continuation marker (0xFFFFFFFF) and ends with the
    // end-of-stream marker (0xFFFFFFFF, 0,0,0,0).
    assert_eq!(&stream[0..4], &[0xFF, 0xFF, 0xFF, 0xFF]);
    assert_eq!(
        &stream[stream.len() - 8..],
        &[0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0]
    );
    // File layout is Feather-v2: `ARROW1\0\0` prefix and `ARROW1` suffix.
    assert_eq!(&file[0..8], b"ARROW1\0\0");
    assert_eq!(&file[file.len() - 6..], b"ARROW1");
}
