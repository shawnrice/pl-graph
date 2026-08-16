//! The neutral graph + value model the codecs operate on. Both `lenke-core`
//! (`Graph`) and `lenke-engine` (`Store`) convert to/from these types, so the
//! codec logic never touches either engine's storage or `Value`.

/// A neutral property value — the same variant set both engines' `Value` types
/// carry. A temporal is kept as its `(tag, iso)` strings (e.g. `("date",
/// "2024-01-15")`); the ISO string is validated when the host converts this back
/// into its own `Value`, not here.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Temporal {
        tag: String,
        iso: String,
    },
    List(Vec<Value>),
    /// An ordered key→value object (covers both a Gremlin map and a record).
    Map(Vec<(String, Value)>),
}

/// A node in the neutral model: external id, labels (stored order), and present
/// properties (in the order the source yields them).
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub id: String,
    pub labels: Vec<String>,
    pub props: Vec<(String, Value)>,
}

/// An edge in the neutral model. `labels[0]` is the edge's type; the rest are
/// secondary labels. `id` is the external edge id when the source carries one.
#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub id: Option<String>,
    pub from: String,
    pub to: String,
    pub labels: Vec<String>,
    pub props: Vec<(String, Value)>,
}

/// A whole graph's worth of neutral records — the codec input/output.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphData {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

/// The temporal tags recognized in a JSON tagged-temporal object `{"@<tag>": iso}`
/// (matches the engines' temporal kinds). A single-key object whose key strips to
/// one of these is a temporal; any other object is a map.
pub const TEMPORAL_TAGS: &[&str] = &[
    "date",
    "localtime",
    "time",
    "datetime",
    "localdatetime",
    "zoned_time",
    "zoned_datetime",
    "duration",
];

/// Whether `tag` (already stripped of a leading `@`) names a temporal kind.
#[must_use]
pub fn is_temporal_tag(tag: &str) -> bool {
    TEMPORAL_TAGS.contains(&tag)
}
