//! CSV codec — a Neo4j-`admin-import`-style pair of typed CSVs (nodes + edges)
//! joined into one document by a `=== EDGES ===` sentinel line.
//!
//! **nodes** columns: `id`, `:LABEL` (label set, `;`-joined), then one typed
//! column per property key (`key:string|integer|float|boolean`, lists as
//! `key:integer[]` with `;`-joined elements). **edges** columns: `id`,
//! `:START_ID`, `:END_ID`, `:TYPE`, then the same typed property columns.
//!
//! The hard parts, ported faithfully from the TS codec:
//!   - **null / empty-string / absent** are three distinct on-wire states:
//!     absent = empty *unquoted* cell; null = the `\N` token; present `""` =
//!     quoted empty. Null is a first-class stored value, so `\N` round-trips as
//!     a present null (distinct from absent).
//!   - **heterogeneous keys**: a cell whose concrete type differs from its
//!     column's type carries an inline `\T<code>:` override sigil (`s|i|f|b`,
//!     `[]` for lists), so a key used with mixed types still round-trips.
//!   - RFC-4180 quoting; list elements escape `;` and `\`.
//!
//! Known limitation: an empty list `[]` and a single-empty-string-element list
//! `[""]` both encode to a quoted-empty cell and both decode to `[]` — the
//! quote bit is already spent distinguishing absent from present-`[]`, leaving
//! no room for a third empty-content state without a wire-format change. `[""]`
//! is a degenerate input; the collapse is accepted rather than break every
//! existing list encoding for it.
//!
//! Core divergences (see [`crate::codec`]): edge `:TYPE` is the single edge type
//! (not a set); the edge `id` column round-trips an assigned external id and is
//! empty for id-less edges.

use crate::codec::{element_props, is_intish};
use crate::error::CodeResult;
use crate::graph::{Builder, EdgeRec, Graph, NodeRec, Value};

const NULL_TOKEN: &str = "\\N";
const LIST_SEP: char = ';';
const OVERRIDE_PREFIX: &str = "\\T";
/// Element type-override code for a null LIST element (`\Tn:`, empty body).
const NULL_ELEMENT_CODE: &str = "n";
/// The lone marker row that separates the nodes section from the edges section.
const EDGES_MARKER: &str = "=== EDGES ===";
/// The marker as written between the two sections (its own line).
const SEPARATOR: &str = "\n=== EDGES ===\n";

// ---------------------------------------------------------------------------
// Column types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scalar {
    Str,
    Int,
    Float,
    Bool,
    Date,
    Time,
    DateTime,
    ZonedTime,
    ZonedDateTime,
    Duration,
}

impl Scalar {
    fn as_str(self) -> &'static str {
        match self {
            Self::Str => "string",
            Self::Int => "integer",
            Self::Float => "float",
            Self::Bool => "boolean",
            Self::Date => "date",
            Self::Time => "localtime",
            Self::DateTime => "datetime",
            Self::ZonedTime => "zoned_time",
            Self::ZonedDateTime => "zoned_datetime",
            Self::Duration => "duration",
        }
    }
    fn from_str(s: &str) -> Self {
        match s {
            "integer" => Self::Int,
            "float" => Self::Float,
            "boolean" => Self::Bool,
            "date" => Self::Date,
            "localtime" => Self::Time,
            "datetime" => Self::DateTime,
            "zoned_time" => Self::ZonedTime,
            "zoned_datetime" => Self::ZonedDateTime,
            "duration" => Self::Duration,
            _ => Self::Str,
        }
    }
    fn code(self) -> char {
        match self {
            Self::Str => 's',
            Self::Int => 'i',
            Self::Float => 'f',
            Self::Bool => 'b',
            Self::Date => 'd',
            Self::Time => 'l',
            Self::DateTime => 't',
            Self::ZonedTime => 'w',
            Self::ZonedDateTime => 'z',
            Self::Duration => 'u',
        }
    }
    fn from_code(c: &str) -> Self {
        match c {
            "i" => Self::Int,
            "f" => Self::Float,
            "b" => Self::Bool,
            "d" => Self::Date,
            "l" => Self::Time,
            "t" => Self::DateTime,
            "w" => Self::ZonedTime,
            "z" => Self::ZonedDateTime,
            "u" => Self::Duration,
            _ => Self::Str,
        }
    }
    /// The kind tag for a temporal scalar type, or `None` for a non-temporal type.
    fn temporal_tag(self) -> Option<&'static str> {
        match self {
            Self::Date => Some("date"),
            Self::Time => Some("localtime"),
            Self::DateTime => Some("datetime"),
            Self::ZonedTime => Some("zoned_time"),
            Self::ZonedDateTime => Some("zoned_datetime"),
            Self::Duration => Some("duration"),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ColType {
    scalar: Scalar,
    list: bool,
}

fn scalar_of(v: &Value) -> Scalar {
    use crate::temporal::Temporal;
    match v {
        Value::Bool(_) => Scalar::Bool,
        Value::Num(x) => {
            if is_intish(*x) {
                Scalar::Int
            } else {
                Scalar::Float
            }
        }
        Value::Temporal(Temporal::Date(_)) => Scalar::Date,
        Value::Temporal(Temporal::Time(_)) => Scalar::Time,
        Value::Temporal(Temporal::DateTime(_)) => Scalar::DateTime,
        Value::Temporal(Temporal::ZonedTime(_)) => Scalar::ZonedTime,
        Value::Temporal(Temporal::ZonedDateTime(_)) => Scalar::ZonedDateTime,
        Value::Temporal(Temporal::Duration(_)) => Scalar::Duration,
        _ => Scalar::Str,
    }
}

/// The scalar type to attribute to a list element (null / nested list → string).
fn scalar_of_element(el: &Value) -> Scalar {
    match el {
        Value::Null | Value::List(_) => Scalar::Str,
        other => scalar_of(other),
    }
}

fn infer_column(v: &Value) -> ColType {
    match v {
        Value::List(elems) => {
            let scalar = match elems.first() {
                Some(first) if !matches!(first, Value::Null | Value::List(_)) => scalar_of(first),
                _ => Scalar::Str,
            };
            ColType { scalar, list: true }
        }
        other => ColType {
            scalar: scalar_of(other),
            list: false,
        },
    }
}

fn column_header(key: &str, t: ColType) -> String {
    format!(
        "{}:{}{}",
        guard_field(key),
        t.scalar.as_str(),
        if t.list { "[]" } else { "" }
    )
}

// A property KEY is arbitrary text, so a header cell must be quoted exactly like
// a data cell — an unquoted `,`/`"`/newline in a key would break column
// alignment on decode. The decoder parses the header with the same quote-aware
// parser, so quoting is transparent to `parse_header`.
fn header_line(cells: &[String]) -> String {
    cells
        .iter()
        .map(|c| quote_field(c))
        .collect::<Vec<_>>()
        .join(",")
}

// Leading chars a spreadsheet reads as a formula (`= + - @`, plus TAB/CR). A
// STRING value starting with one is neutralized on encode by reusing the
// leading-backslash escape (see encode_cell); numbers (`-5`) are left alone.
fn starts_with_formula(s: &str) -> bool {
    matches!(s.chars().next(), Some('=' | '+' | '-' | '@' | '\t' | '\r'))
}

fn parse_header(header: &str) -> (String, ColType) {
    let colon = header.rfind(':').unwrap_or(header.len());
    let key = unguard_field(&header[..colon]);
    let mut type_part = if colon < header.len() {
        &header[colon + 1..]
    } else {
        ""
    };
    let list = type_part.ends_with("[]");
    if list {
        type_part = &type_part[..type_part.len() - 2];
    }
    (
        key,
        ColType {
            scalar: Scalar::from_str(type_part),
            list,
        },
    )
}

fn type_code(t: ColType) -> String {
    format!("{}{}", t.scalar.code(), if t.list { "[]" } else { "" })
}

// ---------------------------------------------------------------------------
// Scalar (de)serialization
// ---------------------------------------------------------------------------

fn num_str(x: f64) -> String {
    if x.is_finite() {
        crate::jsonfmt::js_number(x)
    } else {
        "null".to_string()
    }
}

/// One scalar's raw (pre-quoting) text, of the given column scalar type.
fn scalar_to_raw(scalar: Scalar, v: &Value) -> String {
    if scalar == Scalar::Bool {
        return match v {
            Value::Bool(true) => "true".to_string(),
            _ => "false".to_string(),
        };
    }
    match v {
        Value::Num(x) => num_str(*x),
        Value::Str(s) => s.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Temporal(t) => t.format(),
        _ => String::new(),
    }
}

fn raw_to_scalar(scalar: Scalar, raw: &str) -> Value {
    if let Some(tag) = scalar.temporal_tag() {
        // A well-formed temporal decodes to `Value::Temporal`; a malformed cell
        // falls back to a string (lenient, matching the other scalar paths).
        return crate::temporal::Temporal::parse(tag, raw)
            .map_or_else(|_| Value::Str(raw.into()), Value::Temporal);
    }
    match scalar {
        Scalar::Bool => Value::Bool(raw == "true"),
        Scalar::Int | Scalar::Float => Value::Num(raw.parse().unwrap_or(f64::NAN)),
        _ => Value::Str(raw.into()),
    }
}

fn escape_element(s: &str) -> String {
    s.replace('\\', "\\\\").replace(';', "\\;")
}

// Neutralization for a RAW-TEXT cell with no escape namespace (a node/edge id,
// an endpoint id, or a property KEY in a header): prefix `\` when it begins with
// `\` (so a genuine leading backslash survives) or a formula char (so a
// spreadsheet won't evaluate it); `unguard_field` strips one leading `\`.
fn guard_field(s: &str) -> String {
    if s.starts_with('\\') || starts_with_formula(s) {
        format!("\\{s}")
    } else {
        s.to_string()
    }
}

fn unguard_field(s: &str) -> String {
    s.strip_prefix('\\').unwrap_or(s).to_string()
}

// Neutralization for an already-`escape_element`-escaped label / string list
// element. Its body already doubles a leading `\`, so only a leading formula
// char needs guarding; `split_list`'s generic `\X`→X strip reverses it.
fn guard_element(escaped: String) -> String {
    if starts_with_formula(&escaped) {
        format!("\\{escaped}")
    } else {
        escaped
    }
}

/// Split a list cell on unescaped `;`, unescaping `\;` and `\\` inline.
fn split_list(raw: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(n) = chars.next() {
                cur.push(n);
            }
        } else if c == LIST_SEP {
            out.push(std::mem::take(&mut cur));
        } else {
            cur.push(c);
        }
    }
    out.push(cur);
    out
}

fn element_to_raw(elem_scalar: Scalar, el: &Value) -> String {
    // A null LIST ELEMENT. The scalar `\N` token can't be reused inside a list
    // (no per-element quote bit, and `split_list` strips one backslash), so null
    // gets its own type-override code — `\Tn:` (empty body), the sibling of the
    // `\Ti:`/`\Ts:` element tags — so `[1, null, 2]` round-trips exactly rather
    // than losing the null to the string `"null"`.
    if matches!(el, Value::Null) {
        return escape_element(&format!("{OVERRIDE_PREFIX}{NULL_ELEMENT_CODE}:"));
    }
    let actual = scalar_of_element(el);
    let raw = scalar_to_raw(actual, el);
    if actual == elem_scalar {
        // Guard a formula-leading STRING element; a number stays bare, and an
        // override element already begins with `\T`.
        let body = escape_element(&raw);
        if actual == Scalar::Str {
            guard_element(body)
        } else {
            body
        }
    } else {
        escape_element(&format!("{OVERRIDE_PREFIX}{}:{}", actual.code(), raw))
    }
}

fn raw_to_element(elem_scalar: Scalar, part: &str) -> Value {
    if let Some(rest) = part.strip_prefix(OVERRIDE_PREFIX) {
        if let Some(colon) = rest.find(':') {
            let code = &rest[..colon];
            if code == NULL_ELEMENT_CODE {
                return Value::Null;
            }
            return raw_to_scalar(Scalar::from_code(code), &rest[colon + 1..]);
        }
    }
    raw_to_scalar(elem_scalar, part)
}

fn value_to_raw(t: ColType, v: &Value) -> String {
    if t.list {
        if let Value::List(elems) = v {
            return elems
                .iter()
                .map(|el| element_to_raw(t.scalar, el))
                .collect::<Vec<_>>()
                .join(";");
        }
        return String::new();
    }
    scalar_to_raw(t.scalar, v)
}

fn raw_to_value(t: ColType, raw: &str) -> Value {
    if t.list {
        if raw.is_empty() {
            return Value::List(Vec::new());
        }
        return Value::List(
            split_list(raw)
                .iter()
                .map(|p| raw_to_element(t.scalar, p))
                .collect(),
        );
    }
    raw_to_scalar(t.scalar, raw)
}

// ---------------------------------------------------------------------------
// Cell (de)coding
// ---------------------------------------------------------------------------

struct Encoded {
    raw: String,
    force_quote: bool,
}

fn encode_cell(column: ColType, v: &Value) -> Encoded {
    if matches!(v, Value::Null) {
        return Encoded {
            raw: NULL_TOKEN.to_string(),
            force_quote: false,
        };
    }
    let actual = infer_column(v);
    if actual == column {
        if column.scalar == Scalar::Str && !column.list {
            // scalar strings are always force-quoted (so present "" ≠ absent),
            // and a leading backslash is doubled so a literal `\N`/`\T…` can't be
            // read as a sentinel (decode strips exactly one leading backslash).
            //
            // A leading FORMULA char (`= + - @` / TAB / CR) gets the same single-
            // backslash escape: the on-disk cell then begins with `\` (inert to a
            // spreadsheet — no formula injection), and since neither `\N` nor
            // `\T` starts with a formula char, `\=…` falls through the sentinel
            // checks to the string branch's one-backslash strip, round-tripping.
            let s = match v {
                Value::Str(s) => s.to_string(),
                _ => String::new(),
            };
            let raw = if s.starts_with('\\') || starts_with_formula(&s) {
                format!("\\{s}")
            } else {
                s
            };
            return Encoded {
                raw,
                force_quote: true,
            };
        }
        let raw = value_to_raw(column, v);
        let force = raw.is_empty(); // present-but-empty (e.g. empty list) must quote
        return Encoded {
            raw,
            force_quote: force,
        };
    }
    // heterogeneous cell: tag with its concrete type
    let raw = format!(
        "{OVERRIDE_PREFIX}{}:{}",
        type_code(actual),
        value_to_raw(actual, v)
    );
    Encoded {
        raw,
        force_quote: false,
    }
}

/// `None` = absent (key not on this element).
fn decode_cell(column: ColType, cell: &Cell) -> Option<Value> {
    let text = &cell.text;
    if !cell.quoted && text.is_empty() {
        return None; // absent
    }
    let sentinel = text.starts_with('\\') && !text.starts_with("\\\\");
    if sentinel && text == NULL_TOKEN {
        return Some(Value::Null);
    }
    if sentinel {
        if let Some(rest) = text.strip_prefix(OVERRIDE_PREFIX) {
            if let Some(colon) = rest.find(':') {
                let mut code = &rest[..colon];
                let list = code.ends_with("[]");
                if list {
                    code = &code[..code.len() - 2];
                }
                let ot = ColType {
                    scalar: Scalar::from_code(code),
                    list,
                };
                return Some(raw_to_value(ot, &rest[colon + 1..]));
            }
        }
    }
    if column.scalar == Scalar::Str && !column.list {
        // literal string: undo the leading-backslash escape
        return Some(Value::Str(text.strip_prefix('\\').unwrap_or(text).into()));
    }
    Some(raw_to_value(column, text))
}

// ---------------------------------------------------------------------------
// RFC-4180 plumbing
// ---------------------------------------------------------------------------

struct Cell {
    text: String,
    quoted: bool,
}

fn quote_field(raw: &str) -> String {
    if raw.contains(',')
        || raw.contains('"')
        || raw.contains('\n')
        || raw.contains('\r')
        || raw.contains(LIST_SEP)
    {
        format!("\"{}\"", raw.replace('"', "\"\""))
    } else {
        raw.to_string()
    }
}

/// Single-pass RFC-4180 parser. Each cell carries whether it was quoted (needed
/// to tell `''` from absent/`\N`).
fn parse_csv(input: &str) -> Vec<Vec<Cell>> {
    let mut rows: Vec<Vec<Cell>> = Vec::new();
    let mut row: Vec<Cell> = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut in_quotes = false;
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    // Push the current field, then start a fresh one (resetting the quoted flag).
    macro_rules! end_field {
        () => {{
            row.push(Cell {
                text: std::mem::take(&mut field),
                quoted,
            });
            quoted = false;
        }};
    }

    while i < chars.len() {
        let c = chars[i];
        if in_quotes {
            if c == '"' {
                if chars.get(i + 1) == Some(&'"') {
                    field.push('"');
                    i += 2;
                    continue;
                }
                in_quotes = false;
                i += 1;
                continue;
            }
            field.push(c);
            i += 1;
            continue;
        }
        match c {
            '"' => {
                quoted = true;
                in_quotes = true;
            }
            ',' => end_field!(),
            '\r' => {} // swallow; CRLF handled by the \n branch
            '\n' => {
                end_field!();
                rows.push(std::mem::take(&mut row));
            }
            _ => field.push(c),
        }
        i += 1;
    }
    // Flush the trailing field/row (no reset needed at end of input).
    if !field.is_empty() || quoted || !row.is_empty() {
        row.push(Cell {
            text: field,
            quoted,
        });
        rows.push(row);
    }
    rows
}

// ---------------------------------------------------------------------------
// Column-set computation (header = union of all keys, first-seen order)
// ---------------------------------------------------------------------------

type Bag = Vec<(String, Value)>;

fn bag_get<'a>(bag: &'a Bag, key: &str) -> Option<&'a Value> {
    bag.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn compute_columns(bags: &[Bag]) -> (Vec<String>, std::collections::HashMap<String, ColType>) {
    let mut keys: Vec<String> = Vec::new();
    let mut types: std::collections::HashMap<String, ColType> = std::collections::HashMap::new();
    let mut seen = std::collections::HashSet::new();
    for bag in bags {
        for (key, value) in bag {
            if seen.insert(key.clone()) {
                keys.push(key.clone());
            }
            if !matches!(value, Value::Null) && !types.contains_key(key) {
                types.insert(key.clone(), infer_column(value));
            }
        }
    }
    for key in &keys {
        types.entry(key.clone()).or_insert(ColType {
            scalar: Scalar::Str,
            list: false,
        });
    }
    (keys, types)
}

fn build_row(
    fixed: &[&str],
    keys: &[String],
    types: &std::collections::HashMap<String, ColType>,
    bag: &Bag,
) -> String {
    let mut cells: Vec<String> = fixed.iter().map(|s| quote_field(s)).collect();
    for key in keys {
        match bag_get(bag, key) {
            None => cells.push(String::new()), // absent
            Some(v) => {
                let enc = encode_cell(types[key], v);
                cells.push(if enc.force_quote {
                    format!("\"{}\"", enc.raw.replace('"', "\"\""))
                } else {
                    quote_field(&enc.raw)
                });
            }
        }
    }
    cells.join(",")
}

/// Join a label set into a `;`-separated cell, escaping `;`/`\` inside each label
/// (same scheme as list elements) so a label containing `;` round-trips.
fn join_labels<'a>(labels: impl IntoIterator<Item = &'a str>) -> String {
    labels
        .into_iter()
        .map(|l| guard_element(escape_element(l)))
        .collect::<Vec<_>>()
        .join(";")
}

fn split_labels(text: &str) -> Vec<String> {
    if text.is_empty() {
        Vec::new()
    } else {
        // Escape-aware split, so a label that contains the `;` separator (encoded
        // as `\;`) is not torn into two labels.
        split_list(text)
    }
}

fn prop_cols_from_header(header: &[Cell], fixed: usize) -> Vec<(String, ColType)> {
    header
        .iter()
        .skip(fixed)
        .map(|c| parse_header(&c.text))
        .collect()
}

fn props_from_row(row: &[Cell], prop_cols: &[(String, ColType)], fixed: usize) -> Bag {
    let mut props = Bag::new();
    for (c, (key, t)) in prop_cols.iter().enumerate() {
        let Some(cell) = row.get(c + fixed) else {
            continue;
        };
        match decode_cell(*t, cell) {
            // A `\N` cell is a PRESENT null (kept); only a genuinely empty cell
            // (`None`) is absent. Null is a first-class stored value.
            None => {}
            Some(v) => props.push((key.clone(), v)),
        }
    }
    props
}

// ---------------------------------------------------------------------------
// Bags from the graph
// ---------------------------------------------------------------------------

fn vertex_bags(g: &Graph) -> Vec<(u32, Bag)> {
    (0..g.n as u32)
        .filter(|&vi| g.is_vertex_live(vi))
        .map(|vi| {
            let bag = element_props(&g.props, &g.strs, vi as usize)
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
            (vi, bag)
        })
        .collect()
}

fn edge_bags(g: &Graph) -> Vec<(usize, Bag)> {
    (0..g.edge_slots())
        .filter(|&i| g.is_edge_live(i as u32))
        .map(|i| {
            let bag = element_props(&g.edge_props, &g.strs, i)
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
            (i, bag)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Public nodes / edges CSV
// ---------------------------------------------------------------------------

pub fn encode_nodes(g: &Graph) -> String {
    let entries = vertex_bags(g);
    let bags: Vec<Bag> = entries.iter().map(|(_, b)| b.clone()).collect();
    let (keys, types) = compute_columns(&bags);

    let header = {
        let mut h = vec!["id".to_string(), ":LABEL".to_string()];
        h.extend(keys.iter().map(|k| column_header(k, types[k])));
        header_line(&h)
    };
    let mut rows = vec![header];
    for (vi, bag) in &entries {
        let labels = join_labels(crate::codec::node_labels(g, *vi));
        let id = guard_field(g.vid.text(*vi));
        rows.push(build_row(&[&id, &labels], &keys, &types, bag));
    }
    rows.join("\n")
}

pub fn encode_edges(g: &Graph) -> String {
    let entries = edge_bags(g);
    let bags: Vec<Bag> = entries.iter().map(|(_, b)| b.clone()).collect();
    let (keys, types) = compute_columns(&bags);

    let header = {
        let mut h = vec![
            "id".to_string(),
            ":START_ID".to_string(),
            ":END_ID".to_string(),
            ":TYPE".to_string(),
        ];
        h.extend(keys.iter().map(|k| column_header(k, types[k])));
        header_line(&h)
    };
    let mut rows = vec![header];
    for (i, bag) in &entries {
        let id = guard_field(&g.edge_id(*i as u32));
        let from = guard_field(g.vid.text(g.e_src[*i]));
        let to = guard_field(g.vid.text(g.e_dst[*i]));
        let etype = guard_element(escape_element(g.etype.text(g.e_type[*i])));
        rows.push(build_row(&[&id, &from, &to, &etype], &keys, &types, bag));
    }
    rows.join("\n")
}

/// Encode a graph to the combined single string: nodes CSV, sentinel, edges CSV.
pub fn encode(g: &Graph) -> String {
    format!("{}{}{}", encode_nodes(g), SEPARATOR, encode_edges(g))
}

/// Decode the combined single-string form into a fresh graph (nodes, then edges).
pub fn decode(input: &str) -> CodeResult<Graph> {
    // Parse the whole document first (quote-aware), THEN split at the sentinel
    // *row* — a lone unquoted `=== EDGES ===` cell. Splitting the raw string on
    // the literal sentinel (as a plain substring) would fire inside a quoted
    // property value that happens to contain `\n=== EDGES ===\n`, truncating the
    // nodes section mid-field; a row-level split cannot be fooled this way.
    let all_rows = parse_csv(input);
    let split = all_rows
        .iter()
        .position(|r| r.len() == 1 && !r[0].quoted && r[0].text == EDGES_MARKER);
    let (node_rows, edge_rows): (&[Vec<Cell>], &[Vec<Cell>]) = match split {
        Some(i) => (&all_rows[..i], &all_rows[i + 1..]),
        None => (&all_rows, &[]),
    };

    let mut b = Builder::default();

    if let Some(header) = node_rows.first() {
        let prop_cols = prop_cols_from_header(header, 2);
        for row in node_rows.iter().skip(1) {
            let id = unguard_field(row.first().map(|c| c.text.as_str()).unwrap_or(""));
            let labels = split_labels(row.get(1).map(|c| c.text.as_str()).unwrap_or(""));
            b.nodes.push(NodeRec {
                id,
                labels,
                props: props_from_row(row, &prop_cols, 2),
            });
        }
    }

    if let Some(header) = edge_rows.first() {
        let prop_cols = prop_cols_from_header(header, 4);
        for row in edge_rows.iter().skip(1) {
            let id = row
                .first()
                .map(|c| c.text.clone())
                .filter(|s| !s.is_empty())
                .map(|s| unguard_field(&s));
            let src = unguard_field(row.get(1).map(|c| c.text.as_str()).unwrap_or(""));
            let dst = unguard_field(row.get(2).map(|c| c.text.as_str()).unwrap_or(""));
            let etype = split_labels(row.get(3).map(|c| c.text.as_str()).unwrap_or(""))
                .into_iter()
                .next()
                .unwrap_or_default();
            b.edges.push(EdgeRec {
                src,
                dst,
                etype,
                props: props_from_row(row, &prop_cols, 4),
                id,
            });
        }
    }

    // Batch CSV is strict: an edge endpoint must be a declared node (the
    // streaming decoder creates them on demand, but the combined form lists nodes
    // first). A dangling endpoint is MissingVertex, matching the TS codec.
    b.finalize_strict()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Graph {
        // heterogeneous: missing keys, lists, int/float/bool/string, empty string,
        // and a key (`mix`) used as int on one node and string on another.
        crate::codec::pg_json::decode(
            r#"{"nodes":[
              {"id":"a","labels":["P","Q"],"properties":{"n":42,"w":3.5,"ok":true,"name":"ann","tags":["x","y"],"mix":7,"blank":""}},
              {"id":"b","labels":["P"],"properties":{"name":"bo","mix":"hi"}}
            ],"edges":[{"from":"a","to":"b","labels":["KNOWS"],"properties":{"since":2020,"strength":0.9}}]}"#,
        )
        .unwrap()
    }

    #[test]
    fn edge_to_undeclared_vertex_is_missing_vertex() {
        // Batch CSV declares nodes first; an edge endpoint that was never declared
        // is MissingVertex (strict), matching the TS codec.
        let doc = "id,:LABEL\n=== EDGES ===\nid,:START_ID,:END_ID,:TYPE\ne1,x,y,KNOWS";
        match decode(doc) {
            Err(e) => assert_eq!(e.code, crate::error_codes::ErrorCode::MissingVertex),
            Ok(_) => panic!("expected MissingVertex"),
        }
    }

    #[test]
    fn round_trip() {
        let g = sample();
        let g2 = decode(&encode(&g)).unwrap();
        assert_eq!(g2.vertex_count(), 2);
        assert_eq!(g2.edge_count(), 1);

        let a = g2.vid.get("a").unwrap() as usize;
        let b = g2.vid.get("b").unwrap() as usize;
        assert_eq!(g2.props.value(a, "n", &g2.strs), Value::Num(42.0));
        assert_eq!(g2.props.value(a, "w", &g2.strs), Value::Num(3.5));
        assert_eq!(g2.props.value(a, "ok", &g2.strs), Value::Bool(true));
        assert_eq!(
            g2.props.value(a, "name", &g2.strs),
            Value::Str("ann".into())
        );
        assert_eq!(
            g2.props.value(a, "tags", &g2.strs),
            Value::List(vec![Value::Str("x".into()), Value::Str("y".into())]),
        );
        // present empty string survives (≠ absent)
        assert_eq!(g2.props.value(a, "blank", &g2.strs), Value::Str("".into()));
        // multi-label node
        assert_eq!(crate::codec::node_labels(&g2, a as u32).len(), 2);

        // heterogeneous `mix`: int on a, string on b — both recovered via the sigil
        assert_eq!(g2.props.value(a, "mix", &g2.strs), Value::Num(7.0));
        assert_eq!(g2.props.value(b, "mix", &g2.strs), Value::Str("hi".into()));

        // absent key stays absent (b has no `n`)
        assert_eq!(g2.props.value(b, "n", &g2.strs), Value::Null);
    }

    #[test]
    fn quoting_and_separators() {
        let g = crate::codec::pg_json::decode(
            r#"{"nodes":[{"id":"a","labels":[],"properties":{"s":"has,comma \"quote\" and ;semi"}}],"edges":[]}"#,
        )
        .unwrap();
        let g2 = decode(&encode(&g)).unwrap();
        let a = g2.vid.get("a").unwrap() as usize;
        assert_eq!(
            g2.props.value(a, "s", &g2.strs),
            Value::Str("has,comma \"quote\" and ;semi".into())
        );
    }

    #[test]
    fn label_containing_separator_round_trips() {
        // A label (or edge type) containing the `;` list-separator must be
        // escaped, not torn into multiple labels.
        let g = crate::codec::pg_json::decode(
            r#"{"nodes":[
              {"id":"a","labels":["has;semi","Plain"],"properties":{}},
              {"id":"b","labels":[],"properties":{}}
            ],"edges":[{"from":"a","to":"b","labels":["REL;X"],"properties":{}}]}"#,
        )
        .unwrap();
        let g2 = decode(&encode(&g)).unwrap();
        let a = g2.vid.get("a").unwrap();
        let mut labels = crate::codec::node_labels(&g2, a);
        labels.sort();
        assert_eq!(labels, vec!["Plain", "has;semi"]);
        assert_eq!(g2.etype.text(g2.e_type[0]), "REL;X");
    }

    #[test]
    fn section_marker_inside_a_value_does_not_split() {
        // A property value containing the literal `\n=== EDGES ===\n` marker must
        // not be mistaken for the nodes/edges section boundary.
        let g = crate::codec::pg_json::decode(
            r#"{"nodes":[
              {"id":"a","labels":["N"],"properties":{"note":"x\n=== EDGES ===\ny"}},
              {"id":"b","labels":["N"],"properties":{}}
            ],"edges":[{"from":"a","to":"b","labels":["R"],"properties":{}}]}"#,
        )
        .unwrap();
        let g2 = decode(&encode(&g)).unwrap();
        assert_eq!(g2.vertex_count(), 2, "premature split dropped a node");
        assert_eq!(g2.edge_count(), 1, "premature split dropped the edge");
        let a = g2.vid.get("a").unwrap() as usize;
        assert_eq!(
            g2.props.value(a, "note", &g2.strs),
            Value::Str("x\n=== EDGES ===\ny".into())
        );
    }

    #[test]
    fn property_key_with_delimiters_round_trips_via_quoted_header() {
        // An arbitrary key (`a,b`) would break column alignment if the header
        // cell weren't quoted like a data cell.
        let g = crate::codec::pg_json::decode(
            r#"{"nodes":[{"id":"n1","labels":["N"],"properties":{"a,b":1}}],"edges":[]}"#,
        )
        .unwrap();
        let enc = encode(&g);
        assert!(enc.lines().next().unwrap().contains("\"a,b:integer\""));
        let g2 = decode(&enc).unwrap();
        let n1 = g2.vid.get("n1").unwrap() as usize;
        assert_eq!(g2.props.value(n1, "a,b", &g2.strs), Value::Num(1.0));
    }

    #[test]
    fn formula_leading_strings_are_neutralized_and_round_trip() {
        // A string value starting with a spreadsheet formula char is escaped to a
        // leading backslash on the wire (inert to a spreadsheet), then restored.
        let g = crate::codec::pg_json::decode(
            r#"{"nodes":[{"id":"n1","labels":["N"],"properties":{"name":"=1+2","cmd":"@x","dash":"-danger"}}],"edges":[]}"#,
        )
        .unwrap();
        let enc = encode(&g);
        assert!(enc.contains("\"\\=1+2\""), "= not neutralized: {enc}");
        assert!(enc.contains("\"\\@x\""), "@ not neutralized");
        assert!(enc.contains("\"\\-danger\""), "- not neutralized");
        let g2 = decode(&enc).unwrap();
        let n1 = g2.vid.get("n1").unwrap() as usize;
        assert_eq!(
            g2.props.value(n1, "name", &g2.strs),
            Value::Str("=1+2".into())
        );
        assert_eq!(
            g2.props.value(n1, "dash", &g2.strs),
            Value::Str("-danger".into())
        );
    }

    #[test]
    fn negative_numbers_are_not_neutralized() {
        // A number is not a formula — a spreadsheet reads `-5` as a number, and
        // prefixing it would corrupt the round trip.
        let g = crate::codec::pg_json::decode(
            r#"{"nodes":[{"id":"n1","labels":["N"],"properties":{"balance":-5}}],"edges":[]}"#,
        )
        .unwrap();
        let enc = encode(&g);
        assert!(!enc.contains("\\-5"), "number wrongly neutralized: {enc}");
        let g2 = decode(&enc).unwrap();
        let n1 = g2.vid.get("n1").unwrap() as usize;
        assert_eq!(g2.props.value(n1, "balance", &g2.strs), Value::Num(-5.0));
    }

    /// Quote-aware split into each cell's spreadsheet-visible (RFC-4180) content.
    fn csv_cells(csv: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut f = String::new();
        let mut in_q = false;
        let chars: Vec<char> = csv.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if in_q {
                if c == '"' {
                    if chars.get(i + 1) == Some(&'"') {
                        f.push('"');
                        i += 1;
                    } else {
                        in_q = false;
                    }
                } else {
                    f.push(c);
                }
            } else if c == '"' {
                in_q = true;
            } else if c == ',' || c == '\n' {
                out.push(std::mem::take(&mut f));
            } else if c != '\r' {
                f.push(c);
            }
            i += 1;
        }
        out.push(f);
        out
    }

    /// No cell a spreadsheet would evaluate as a formula, except the fixed
    /// `=== EDGES ===` structural marker (constant, never attacker data).
    fn assert_no_formula_cells(csv: &str) {
        for cell in csv_cells(csv) {
            if cell == "=== EDGES ===" {
                continue;
            }
            assert!(
                !matches!(
                    cell.chars().next(),
                    Some('=' | '+' | '-' | '@' | '\t' | '\r')
                ),
                "spreadsheet-dangerous cell: {cell:?}"
            );
        }
    }

    #[test]
    fn formula_leading_content_is_neutralized_across_every_surface() {
        // ids, labels, edge type, keys, string values, and string list elements,
        // for each printable formula char — all round-trip and none stays dangerous.
        for l in ['=', '+', '-', '@'] {
            let doc = format!(
                "{{\"nodes\":[\
                   {{\"id\":\"{l}nid\",\"labels\":[\"{l}Lab\",\"Plain\"],\
                     \"properties\":{{\"{l}k\":\"{l}v\",\"{l}list\":[\"{l}e\",\"ok\"]}}}},\
                   {{\"id\":\"plain\",\"labels\":[],\"properties\":{{}}}}\
                 ],\"edges\":[\
                   {{\"id\":\"{l}eid\",\"from\":\"{l}nid\",\"to\":\"plain\",\
                     \"labels\":[\"{l}T\"],\"properties\":{{\"{l}ek\":\"{l}ev\"}}}}\
                 ]}}"
            );
            let lead = l;
            let g = crate::codec::pg_json::decode(&doc).unwrap();
            let enc = encode(&g);
            assert_no_formula_cells(&enc);

            let g2 = decode(&enc).unwrap();
            assert_eq!(g2.vertex_count(), 2);
            assert_eq!(g2.edge_count(), 1);
            let nid = g2.vid.get(&format!("{lead}nid")).unwrap() as usize;
            assert_eq!(
                g2.props.value(nid, &format!("{lead}k"), &g2.strs),
                Value::Str(format!("{lead}v").into())
            );
            assert_eq!(
                g2.props.value(nid, &format!("{lead}list"), &g2.strs),
                Value::List(vec![
                    Value::Str(format!("{lead}e").into()),
                    Value::Str("ok".into())
                ])
            );
            let mut labels = crate::codec::node_labels(&g2, nid as u32);
            labels.sort();
            assert!(labels.contains(&format!("{lead}Lab").as_str()));
            assert!(g2.vid.get(&format!("{lead}eid")).is_some() || g2.edge_count() == 1);
        }
    }

    #[test]
    fn control_char_leading_cells_are_neutralized() {
        // TAB / CR leading a string value or id must not survive as the cell's
        // first char (a spreadsheet treats leading TAB/CR as a formula too).
        for esc in ["\\t", "\\r"] {
            let doc = format!(
                r#"{{"nodes":[{{"id":"{esc}nid","labels":["N"],"properties":{{"name":"{esc}v"}}}}],"edges":[]}}"#
            );
            let g = crate::codec::pg_json::decode(&doc).unwrap();
            let enc = encode(&g);
            assert_no_formula_cells(&enc);
            let g2 = decode(&enc).unwrap();
            assert_eq!(g2.vertex_count(), 1);
        }
    }

    #[test]
    fn null_list_element_round_trips() {
        // A null INSIDE a list must round-trip exactly — it was lost to the string
        // "null" before this fix. It encodes with the `\Tn:` element sigil.
        let g = crate::codec::pg_json::decode(
            r#"{"nodes":[{"id":"t","labels":["T"],"properties":{"dims":[1,null,2],"oneNull":[null]}}],"edges":[]}"#,
        )
        .unwrap();
        let enc = encode(&g);
        assert!(
            enc.contains("Tn:"),
            "null element should use the \\Tn: sigil, got: {enc}"
        );

        let g2 = decode(&enc).unwrap();
        let t = g2.vid.get("t").unwrap() as usize;
        assert_eq!(
            g2.props.value(t, "dims", &g2.strs),
            Value::List(vec![Value::Num(1.0), Value::Null, Value::Num(2.0)])
        );
        assert_eq!(
            g2.props.value(t, "oneNull", &g2.strs),
            Value::List(vec![Value::Null])
        );
    }

    #[test]
    fn genuine_backslash_leading_content_round_trips() {
        // A genuine leading backslash (plain, and backslash-then-formula) in ids,
        // labels, keys, and string values must survive intact.
        let g = crate::codec::pg_json::decode(
            r#"{"nodes":[{"id":"\\node","labels":["\\Label","\\=trap"],"properties":{"\\key":"\\value","\\list":["\\a","\\=b"]}},{"id":"\\=weird","labels":[],"properties":{}}],"edges":[{"id":"\\edge","from":"\\node","to":"\\=weird","labels":["\\R"],"properties":{}}]}"#,
        )
        .unwrap();
        let enc = encode(&g);
        assert_no_formula_cells(&enc);
        let g2 = decode(&enc).unwrap();
        assert_eq!(g2.vertex_count(), 2);
        assert_eq!(g2.edge_count(), 1);
        let node = g2.vid.get("\\node").unwrap() as usize;
        assert_eq!(
            g2.props.value(node, "\\key", &g2.strs),
            Value::Str("\\value".into())
        );
        assert!(g2.vid.get("\\=weird").is_some());
    }
}
