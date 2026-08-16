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

/// The temporal kind tags (matching the engines' `Temporal::tag()`): the key in a
/// JSON tagged-temporal object `{"@<tag>": iso}`. A single-key object whose key
/// strips to one of these is a temporal; any other object is a map.
pub const TEMPORAL_TAGS: &[&str] = &[
    "date",
    "localtime",
    "datetime",
    "zoned_time",
    "zoned_datetime",
    "duration",
];

/// Whether `tag` (already stripped of a leading `@`) names a temporal kind.
#[must_use]
pub fn is_temporal_tag(tag: &str) -> bool {
    TEMPORAL_TAGS.contains(&tag)
}

/// A temporal kind tag → its GraphSON v3 `@type` name (TinkerPop extended types).
/// Mirrors `Temporal::graphson_type`. Returns `None` for an unknown tag.
#[must_use]
pub fn graphson_type(tag: &str) -> Option<&'static str> {
    Some(match tag {
        "date" => "gx:LocalDate",
        "localtime" => "gx:LocalTime",
        "datetime" => "gx:LocalDateTime",
        "zoned_time" => "gx:OffsetTime",
        "zoned_datetime" => "gx:OffsetDateTime",
        "duration" => "gx:Duration",
        _ => return None,
    })
}

/// A GraphSON `@type` name → the temporal kind tag (the inverse of
/// [`graphson_type`]). Mirrors `Temporal::graphson_tag`.
#[must_use]
pub fn graphson_tag(ty: &str) -> Option<&'static str> {
    Some(match ty {
        "gx:LocalDate" => "date",
        "gx:LocalTime" => "localtime",
        "gx:LocalDateTime" => "datetime",
        "gx:OffsetTime" => "zoned_time",
        "gx:OffsetDateTime" => "zoned_datetime",
        "gx:Duration" => "duration",
        _ => return None,
    })
}
