//! Constraint & schema methods on `Graph`: unique / required / type constraints
//! (vertex and edge), cardinality (degree) bounds, custom GQL-predicate validators,
//! whole-graph invariants, and the `dump_schema` replay serialization. A separate
//! `impl Graph` block (a submodule sees the struct's private fields); shares helper
//! types/functions via `use super::*`.
use super::*;

// Shared accessors for the key-set constraint maps (`HashMap<name, Vec<key>>`),
// used identically by the unique / required families on both vertices (keyed by
// label) and edges (keyed by edge type). Centralizing them keeps the "drop empties
// the entry" and "listing is sorted" rules in one place instead of four copies.
fn keyset_drop(map: &mut HashMap<String, Vec<String>>, name: &str, key: &str) {
    if let Some(keys) = map.get_mut(name) {
        keys.retain(|k| k != key);
        if keys.is_empty() {
            map.remove(name);
        }
    }
}

fn keyset_get<'a>(map: &'a HashMap<String, Vec<String>>, name: &str) -> &'a [String] {
    map.get(name).map_or(&[], Vec::as_slice)
}

fn keyset_has(map: &HashMap<String, Vec<String>>, name: &str, key: &str) -> bool {
    map.get(name).is_some_and(|ks| ks.iter().any(|k| k == key))
}

fn keyset_list(map: &HashMap<String, Vec<String>>) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = map
        .iter()
        .flat_map(|(n, ks)| ks.iter().map(move |k| (n.clone(), k.clone())))
        .collect();
    out.sort();
    out
}

impl Graph {
    // --- unique constraints (declared over `(label, property key)`) ---------
    // At most one live vertex carrying `label` may hold a given non-null value
    // for `key`. Backed by the vertex property index (so lookups seek). This is
    // the Pattern-B primitive `_MERGE` keys on; see `docs/design/gql-extensions.md`.

    /// Declare a UNIQUE constraint on `(label, key)`. Creates the backing vertex
    /// index if absent, then registers the constraint. Idempotent. Fails with
    /// [`ErrorCode::ConstraintViolation`] if the *current* data already violates
    /// it — an already-broken constraint is meaningless (SQL rejects the unique
    /// index build the same way).
    pub fn create_unique_constraint(&mut self, label: &str, key: &str) -> CodeResult<()> {
        if !self.vertex_indexed(key) {
            self.create_vertex_index(key);
        }
        if self.first_label_prop_duplicate(label, key).is_some() {
            return Err(CodeError::new(
                ErrorCode::ConstraintViolation,
                "existing data already violates the unique constraint being declared",
            ));
        }
        let keys = self.v_unique.entry(label.to_string()).or_default();
        if !keys.iter().any(|k| k == key) {
            keys.push(key.to_string());
            keys.sort();
        }
        Ok(())
    }

    /// Drop a unique constraint. The backing index is left in place (drop it via
    /// [`Graph::drop_vertex_index`] if unwanted). Idempotent.
    pub fn drop_unique_constraint(&mut self, label: &str, key: &str) {
        keyset_drop(&mut self.v_unique, label, key);
    }

    /// Property keys under a unique constraint for `label` (sorted; empty if
    /// none). `_MERGE` intersects this with the pattern to infer the conflict key.
    pub fn unique_keys(&self, label: &str) -> &[String] {
        keyset_get(&self.v_unique, label)
    }

    /// True iff `(label, key)` carries a unique constraint.
    pub fn has_unique_constraint(&self, label: &str, key: &str) -> bool {
        keyset_has(&self.v_unique, label, key)
    }

    /// Every declared unique constraint as sorted `(label, key)` pairs — a
    /// deterministic listing for host introspection.
    pub fn unique_constraints(&self) -> Vec<(String, String)> {
        keyset_list(&self.v_unique)
    }

    /// The single live vertex carrying `label` whose `key == value`, if any (≤1
    /// under the constraint). The `_MERGE` create-vs-update decision. A non-null
    /// scalar `value` seeks the index; null/list yield `None` (exempt).
    pub fn unique_lookup(&self, label: &str, key: &str, value: &Value) -> Option<u32> {
        self.vertices_with_label_value(label, key, value)
            .into_iter()
            .next()
    }

    /// If adding a vertex with `labels` + `props` would break a unique constraint,
    /// the offending `(label, key, existing vertex)`. Drives INSERT enforcement;
    /// `exclude` skips one vertex (itself, for a re-check). Only constrained keys
    /// present in `props` are checked; null/list values are exempt.
    pub fn unique_conflict(
        &self,
        labels: &[String],
        props: &[(String, Value)],
        exclude: Option<u32>,
    ) -> Option<(String, String, u32)> {
        if self.v_unique.is_empty() {
            return None;
        }
        for label in labels {
            for key in self.unique_keys(label) {
                let Some((_, value)) = props.iter().find(|(k, _)| k == key) else {
                    continue;
                };
                let hit = self
                    .vertices_with_label_value(label, key, value)
                    .into_iter()
                    .find(|&v| Some(v) != exclude);
                if let Some(existing) = hit {
                    return Some((label.clone(), key.clone(), existing));
                }
            }
        }
        None
    }

    // --- required constraints -------------------------------------------------
    // Every live vertex carrying `label` must hold a present, non-null value for
    // each required `key`. Enforced in the write path (INSERT/SET/REMOVE) like
    // `unique`; declarative (no closures), so it is byte-identical to the TS core.
    // No backing index is needed — enforcement is a presence check.

    /// Declare a REQUIRED constraint on `(label, key)`. Idempotent. Fails with
    /// [`ErrorCode::ConstraintViolation`] if any live vertex with `label` lacks a
    /// present, non-null `key` — an already-violated constraint is meaningless.
    pub fn create_required_constraint(&mut self, label: &str, key: &str) -> CodeResult<()> {
        if let Some(lid) = self.labels.get(label) {
            for vi in self.vertex_indices() {
                if self.vlabels[vi as usize].contains(&lid)
                    && matches!(self.props.value(vi as usize, key, &self.strs), Value::Null)
                {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the required constraint being declared",
                    ));
                }
            }
        }
        let keys = self.v_required.entry(label.to_string()).or_default();
        if !keys.iter().any(|k| k == key) {
            keys.push(key.to_string());
            keys.sort();
        }
        Ok(())
    }

    /// Property keys required for `label` (sorted; empty if none).
    pub fn required_keys(&self, label: &str) -> &[String] {
        keyset_get(&self.v_required, label)
    }

    /// Every declared required constraint as sorted `(label, key)` pairs.
    pub fn required_constraints(&self) -> Vec<(String, String)> {
        keyset_list(&self.v_required)
    }

    /// The first `(label, key)` a new vertex with these `labels`/`props` would
    /// violate by omitting a required key (absent or null value), or `None`.
    pub fn missing_required(
        &self,
        labels: &[String],
        props: &[(String, Value)],
    ) -> Option<(String, String)> {
        if self.v_required.is_empty() && self.v_type_not_null.is_empty() {
            return None;
        }
        for label in labels {
            for key in self.effective_required_keys(label) {
                let present = props
                    .iter()
                    .any(|(k, v)| k == key && !matches!(v, Value::Null));
                if !present {
                    return Some((label.clone(), key.to_string()));
                }
            }
        }
        None
    }

    /// The keys REQUIRED (present + non-null) for `label`: the declared required
    /// constraints UNION the scalar keys declared `NOT NULL` on a type constraint.
    fn effective_required_keys(&self, label: &str) -> Vec<&str> {
        let mut ks: Vec<&str> = self
            .required_keys(label)
            .iter()
            .map(String::as_str)
            .collect();
        if let Some(nn) = self.v_type_not_null.get(label) {
            for k in nn {
                if !ks.contains(&k.as_str()) {
                    ks.push(k);
                }
            }
        }
        ks
    }

    // --- type constraints -----------------------------------------------------
    // Every present, non-null value under a constrained `key` on a vertex with
    // `label` must be of the declared scalar type. Null/absent are exempt.
    // Enforced in the write path; byte-identical to the TS core.

    /// Declare a TYPE constraint on `(label, key)` requiring `type_name` (one of
    /// string/number/boolean/date/datetime/duration/list). Fails with
    /// `InvalidValue` for an unknown type name, or `ConstraintViolation` if any
    /// existing vertex holds a present, non-null `key` of a different type.
    pub fn create_type_constraint(
        &mut self,
        label: &str,
        key: &str,
        type_name: &str,
    ) -> CodeResult<()> {
        let Some((spec, not_null)) = TypeSpec::parse_with_not_null(type_name) else {
            return Err(CodeError::new(
                ErrorCode::InvalidValue,
                "unknown or malformed type name for a type constraint",
            ));
        };
        // Validate existing data against the declared type (a null is exempt) — and,
        // when `NOT NULL`, that no label vertex already holds an absent/null value.
        if let Some(lid) = self.labels.get(label) {
            for vi in self.vertex_indices() {
                if !self.vlabels[vi as usize].contains(&lid) {
                    continue;
                }
                let v = self.props.value(vi as usize, key, &self.strs);
                if !value_matches(&v, &spec) {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the type constraint being declared",
                    ));
                }
                if not_null && matches!(v, Value::Null) {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the NOT NULL constraint being declared",
                    ));
                }
            }
        }
        // Scalar → the `Copy` `v_type` map (unchanged fast path); a record shape →
        // the parallel `v_record` map.
        match spec {
            TypeSpec::Scalar(ty) => {
                self.v_type
                    .entry(label.to_string())
                    .or_default()
                    .insert(key.to_string(), ty);
                if not_null {
                    self.v_type_not_null
                        .entry(label.to_string())
                        .or_default()
                        .insert(key.to_string());
                }
            }
            record => {
                // De-box the store for this key into typed sub-columns — the shape
                // is now a contract (see [`Column::Record`]). A no-op for `AnyRecord`
                // (no field contract → stays boxed). Future writes scatter in place.
                self.props.debox_record(key, &record, &mut self.strs);
                self.v_record
                    .entry(label.to_string())
                    .or_default()
                    .insert(key.to_string(), record);
                // A record-level `NOT NULL` makes the whole property required.
                if not_null {
                    self.v_type_not_null
                        .entry(label.to_string())
                        .or_default()
                        .insert(key.to_string());
                }
            }
        }
        Ok(())
    }

    /// Drop a type constraint (scalar or record). Idempotent.
    pub fn drop_type_constraint(&mut self, label: &str, key: &str) {
        if let Some(keys) = self.v_type.get_mut(label) {
            keys.remove(key);
            if keys.is_empty() {
                self.v_type.remove(label);
            }
        }
        if let Some(keys) = self.v_record.get_mut(label) {
            keys.remove(key);
            if keys.is_empty() {
                self.v_record.remove(label);
            }
        }
        // Drop this constraint's `NOT NULL` (leaving any independently-declared
        // required constraint on the same key intact).
        if let Some(keys) = self.v_type_not_null.get_mut(label) {
            keys.remove(key);
            if keys.is_empty() {
                self.v_type_not_null.remove(label);
            }
        }
        // Re-box the column once NO label still constrains this key as a record.
        if !self.v_record.values().any(|ks| ks.contains_key(key)) {
            self.props.rebox_record(key, &self.strs);
        }
    }

    /// The first `(label, key)` a new vertex with these `labels`/`props` would
    /// violate by holding a wrong-typed value, or `None`.
    pub fn type_violation(
        &self,
        labels: &[String],
        props: &[(String, Value)],
    ) -> Option<(String, String)> {
        if self.v_type.is_empty() && self.v_record.is_empty() {
            return None;
        }
        for label in labels {
            if let Some(cs) = self.v_type.get(label) {
                for (key, ty) in cs {
                    if let Some((_, v)) = props.iter().find(|(k, _)| k == key) {
                        if let Some(got) = value_type(v) {
                            if got != *ty {
                                return Some((label.clone(), key.clone()));
                            }
                        }
                    }
                }
            }
            if let Some(cs) = self.v_record.get(label) {
                for (key, spec) in cs {
                    if let Some((_, v)) = props.iter().find(|(k, _)| k == key) {
                        if !value_matches(v, spec) {
                            return Some((label.clone(), key.clone()));
                        }
                    }
                }
            }
        }
        None
    }

    /// True iff setting `vi.key = value` would break a type constraint on one of
    /// `vi`'s labels. A null value is exempt.
    pub fn type_conflict_on_set(&self, vi: u32, key: &str, value: &Value) -> bool {
        // Scalar constraints: a non-scalar (null/map) has no scalar type, so it
        // can't conflict with a scalar declaration here.
        if let Some(got) = value_type(value) {
            for (label, cs) in &self.v_type {
                if let Some(ty) = cs.get(key) {
                    if let Some(lid) = self.labels.get(label) {
                        if self.vlabels[vi as usize].contains(&lid) && got != *ty {
                            return true;
                        }
                    }
                }
            }
        }
        // Record constraints: a map value must match the declared shape (a null is
        // exempt via `value_matches`).
        for (label, cs) in &self.v_record {
            if let Some(spec) = cs.get(key) {
                if let Some(lid) = self.labels.get(label) {
                    if self.vlabels[vi as usize].contains(&lid) && !value_matches(value, spec) {
                        return true;
                    }
                }
            }
        }
        false
    }

    // --- edge constraints (edge types) --------------------------------------
    // Direct mirror of the vertex unique/required/type constraints, keyed by edge
    // TYPE instead of node label, enforced against the edge property store
    // (`edge_props`) and the edge property index (`eidx`). Byte-identical to the
    // TS edge constraints. Enforcement is deferred to commit (see
    // `run_deferred_checks`), exactly like the vertex ones.

    /// Declare a UNIQUE constraint on `(edge_type, key)`. Creates the backing edge
    /// index if absent. Fails with `ConstraintViolation` if the current data
    /// already violates it. Idempotent.
    pub fn create_edge_unique_constraint(&mut self, etype: &str, key: &str) -> CodeResult<()> {
        if !self.edge_indexed(key) {
            self.create_edge_index(key);
        }
        if self.first_etype_prop_duplicate(etype, key).is_some() {
            return Err(CodeError::new(
                ErrorCode::ConstraintViolation,
                "existing data already violates the edge unique constraint being declared",
            ));
        }
        let keys = self.e_unique.entry(etype.to_string()).or_default();
        if !keys.iter().any(|k| k == key) {
            keys.push(key.to_string());
            keys.sort();
        }
        Ok(())
    }

    /// Drop an edge unique constraint. The backing index is left in place. Idempotent.
    pub fn drop_edge_unique_constraint(&mut self, etype: &str, key: &str) {
        keyset_drop(&mut self.e_unique, etype, key);
    }

    /// Property keys under a unique constraint for `etype` (sorted; empty if none).
    pub fn edge_unique_keys(&self, etype: &str) -> &[String] {
        keyset_get(&self.e_unique, etype)
    }

    /// True iff `(edge_type, key)` carries a unique constraint.
    pub fn has_edge_unique_constraint(&self, etype: &str, key: &str) -> bool {
        keyset_has(&self.e_unique, etype, key)
    }

    /// Every declared edge unique constraint as sorted `(edge_type, key)` pairs.
    pub fn edge_unique_constraints(&self) -> Vec<(String, String)> {
        keyset_list(&self.e_unique)
    }

    /// If adding an edge of `etypes` with `props` would break a unique constraint,
    /// the offending `(edge_type, key, existing edge)`. `exclude` skips one edge.
    pub fn edge_unique_conflict(
        &self,
        etypes: &[String],
        props: &[(String, Value)],
        exclude: Option<u32>,
    ) -> Option<(String, String, u32)> {
        if self.e_unique.is_empty() {
            return None;
        }
        for etype in etypes {
            for key in self.edge_unique_keys(etype) {
                let Some((_, value)) = props.iter().find(|(k, _)| k == key) else {
                    continue;
                };
                let hit = self
                    .edges_with_etype_value(etype, key, value)
                    .into_iter()
                    .find(|&e| Some(e) != exclude);
                if let Some(existing) = hit {
                    return Some((etype.clone(), key.clone(), existing));
                }
            }
        }
        None
    }

    /// Declare a REQUIRED constraint on `(edge_type, key)`. Fails with
    /// `ConstraintViolation` if any live edge of `etype` lacks a present, non-null
    /// `key`. Idempotent.
    pub fn create_edge_required_constraint(&mut self, etype: &str, key: &str) -> CodeResult<()> {
        if let Some(edges) = self.edges_with_etype_name(etype) {
            for &ei in edges {
                if matches!(
                    self.edge_props.value(ei as usize, key, &self.strs),
                    Value::Null
                ) {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the edge required constraint being declared",
                    ));
                }
            }
        }
        let keys = self.e_required.entry(etype.to_string()).or_default();
        if !keys.iter().any(|k| k == key) {
            keys.push(key.to_string());
            keys.sort();
        }
        Ok(())
    }

    /// Property keys required for edge type `etype` (sorted; empty if none).
    pub fn edge_required_keys(&self, etype: &str) -> &[String] {
        keyset_get(&self.e_required, etype)
    }

    /// True iff `(edge_type, key)` carries a required constraint.
    pub fn has_edge_required_constraint(&self, etype: &str, key: &str) -> bool {
        keyset_has(&self.e_required, etype, key)
    }

    /// Every declared edge required constraint as sorted `(edge_type, key)` pairs.
    pub fn edge_required_constraints(&self) -> Vec<(String, String)> {
        keyset_list(&self.e_required)
    }

    /// The first `(edge_type, key)` a new edge with these `etypes`/`props` would
    /// violate by omitting a required key (absent or null value), or `None`.
    pub fn edge_missing_required(
        &self,
        etypes: &[String],
        props: &[(String, Value)],
    ) -> Option<(String, String)> {
        if self.e_required.is_empty() && self.e_type_not_null.is_empty() {
            return None;
        }
        for etype in etypes {
            let mut keys: Vec<&str> = self
                .edge_required_keys(etype)
                .iter()
                .map(String::as_str)
                .collect();
            if let Some(nn) = self.e_type_not_null.get(etype) {
                for k in nn {
                    if !keys.contains(&k.as_str()) {
                        keys.push(k);
                    }
                }
            }
            for key in keys {
                let present = props
                    .iter()
                    .any(|(k, v)| k == key && !matches!(v, Value::Null));
                if !present {
                    return Some((etype.clone(), key.to_string()));
                }
            }
        }
        None
    }

    /// Declare a TYPE constraint on `(edge_type, key)` requiring `type_name`. Fails
    /// with `InvalidValue` for an unknown type name, or `ConstraintViolation` if
    /// any existing edge holds a present, non-null `key` of a different type.
    pub fn create_edge_type_constraint(
        &mut self,
        etype: &str,
        key: &str,
        type_name: &str,
    ) -> CodeResult<()> {
        let Some((spec, not_null)) = TypeSpec::parse_with_not_null(type_name) else {
            return Err(CodeError::new(
                ErrorCode::InvalidValue,
                "unknown or malformed type name for an edge type constraint",
            ));
        };
        if let Some(edges) = self.edges_with_etype_name(etype) {
            for &ei in edges {
                let v = self.edge_props.value(ei as usize, key, &self.strs);
                if !value_matches(&v, &spec) {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the edge type constraint being declared",
                    ));
                }
                if not_null && matches!(v, Value::Null) {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the NOT NULL constraint being declared",
                    ));
                }
            }
        }
        match spec {
            TypeSpec::Scalar(ty) => {
                self.e_type_constraints
                    .entry(etype.to_string())
                    .or_default()
                    .insert(key.to_string(), ty);
                if not_null {
                    self.e_type_not_null
                        .entry(etype.to_string())
                        .or_default()
                        .insert(key.to_string());
                }
            }
            record => {
                self.edge_props.debox_record(key, &record, &mut self.strs);
                self.e_record
                    .entry(etype.to_string())
                    .or_default()
                    .insert(key.to_string(), record);
                if not_null {
                    self.e_type_not_null
                        .entry(etype.to_string())
                        .or_default()
                        .insert(key.to_string());
                }
            }
        }
        Ok(())
    }

    /// Drop an edge type constraint (scalar or record). Idempotent.
    pub fn drop_edge_type_constraint(&mut self, etype: &str, key: &str) {
        if let Some(keys) = self.e_type_constraints.get_mut(etype) {
            keys.remove(key);
            if keys.is_empty() {
                self.e_type_constraints.remove(etype);
            }
        }
        if let Some(keys) = self.e_record.get_mut(etype) {
            keys.remove(key);
            if keys.is_empty() {
                self.e_record.remove(etype);
            }
        }
        if let Some(keys) = self.e_type_not_null.get_mut(etype) {
            keys.remove(key);
            if keys.is_empty() {
                self.e_type_not_null.remove(etype);
            }
        }
        if !self.e_record.values().any(|ks| ks.contains_key(key)) {
            self.edge_props.rebox_record(key, &self.strs);
        }
    }

    /// The declared type for edge `(edge_type, key)`, or `None`.
    pub fn edge_type_constraint(&self, etype: &str, key: &str) -> Option<PropType> {
        self.e_type_constraints.get(etype)?.get(key).copied()
    }

    /// The first `(edge_type, key)` a new edge with these `etypes`/`props` would
    /// violate by holding a wrong-typed value, or `None`.
    pub fn edge_type_violation(
        &self,
        etypes: &[String],
        props: &[(String, Value)],
    ) -> Option<(String, String)> {
        if self.e_type_constraints.is_empty() && self.e_record.is_empty() {
            return None;
        }
        for etype in etypes {
            if let Some(cs) = self.e_type_constraints.get(etype) {
                for (key, ty) in cs {
                    if let Some((_, v)) = props.iter().find(|(k, _)| k == key) {
                        if let Some(got) = value_type(v) {
                            if got != *ty {
                                return Some((etype.clone(), key.clone()));
                            }
                        }
                    }
                }
            }
            if let Some(cs) = self.e_record.get(etype) {
                for (key, spec) in cs {
                    if let Some((_, v)) = props.iter().find(|(k, _)| k == key) {
                        if !value_matches(v, spec) {
                            return Some((etype.clone(), key.clone()));
                        }
                    }
                }
            }
        }
        None
    }

    /// Live edges of type `etype` whose property `key == value`. Seeks the backing
    /// edge index (a constraint always creates one), falling back to a scan.
    /// Non-indexable values (null/list) yield an empty set — exempt from uniqueness.
    fn edges_with_etype_value(&self, etype: &str, key: &str, value: &Value) -> Vec<u32> {
        let Some(idxk) = IdxKey::from_value(value) else {
            return Vec::new();
        };
        let Some(tid) = self.etype.get(etype) else {
            return Vec::new();
        };
        match self.edges_by_prop(key, &idxk) {
            Some(ids) => ids
                .iter()
                .copied()
                .filter(|&e| self.is_edge_live(e) && self.edge_has_label(e, tid))
                .collect(),
            None => (0..self.e_src.len() as u32)
                .filter(|&e| {
                    self.is_edge_live(e)
                        && self.edge_has_label(e, tid)
                        && self.edge_props.value(e as usize, key, &self.strs) == *value
                })
                .collect(),
        }
    }

    /// The first pair of live `etype`-edges that share a value for `key` — for
    /// validating an edge unique constraint against existing data at declare time.
    fn first_etype_prop_duplicate(&self, etype: &str, key: &str) -> Option<(u32, u32)> {
        let tid = self.etype.get(etype)?;
        let bt = self.eidx.get(key)?;
        for ids in bt.values() {
            let mut with_type = ids
                .iter()
                .copied()
                .filter(|&e| self.is_edge_live(e) && self.edge_has_label(e, tid));
            if let (Some(a), Some(b)) = (with_type.next(), with_type.next()) {
                return Some((a, b));
            }
        }
        None
    }

    /// EVERY type name an edge carries (empty vec for a type-less edge) — the
    /// edge analogue of a vertex's label list.
    ///
    /// This decides which constraints apply to an edge at all: it feeds
    /// `edge_missing_required` on the write path and `check_validators_edge`.
    /// Returning only `e_type`, the first, let a two-type edge escape every
    /// constraint declared on its second — silently, since an unenforced
    /// constraint just never fires.
    pub(crate) fn edge_type_names(&self, ei: u32) -> Vec<String> {
        self.edge_labels(ei)
            .into_iter()
            .map(|t| self.etype.text(t).to_string())
            .filter(|n| !n.is_empty())
            .collect()
    }

    /// A live edge's present properties as `(key, value)` pairs — the shape the edge
    /// constraint predicates consume. Edge analogue of `vertex_props`.
    pub(crate) fn edge_props_of(&self, ei: u32) -> Vec<(String, Value)> {
        let i = ei as usize;
        let mut out = Vec::new();
        for kid in 0..self.edge_props.cols.len() as u32 {
            if self.edge_props.is_present_id(i, kid) {
                let key = self.edge_props.keys.text(kid).to_string();
                let val = self.edge_props.value_id(i, kid, &self.strs);
                out.push((key, val));
            }
        }
        out
    }

    // --- cardinality constraints (degree bounds) ----------------------------
    // Bound the degree of every vertex carrying `label` over `etype` in
    // `direction` (0 = out / the vertex is the edge source, 1 = in / the target).
    // Max is deferred to commit against touched endpoints (the GQL layer runs
    // every statement in an auto-commit frame, so a single over-max edge INSERT is
    // caught there); min is commit-time only (unsatisfiable by a single write).
    // The edge write paths note both endpoints as touched; `run_deferred_checks`
    // re-checks them. Byte-identical to the TS core.

    /// Number of live `etype` edges for which `vi` is the SOURCE (out-degree). The
    /// adjacency lists hold only live edges, so this is a filtered count. A
    /// self-loop appears in `out` once, so it counts once here (and once for `in`).
    pub fn out_degree(&self, vi: u32, etype: &str) -> u32 {
        let Some(tid) = self.etype.get(etype) else {
            return 0;
        };
        self.out[vi as usize]
            .iter()
            .filter(|a| a.etype == tid)
            .count() as u32
    }

    /// Number of live `etype` edges for which `vi` is the TARGET (in-degree).
    pub fn in_degree(&self, vi: u32, etype: &str) -> u32 {
        let Some(tid) = self.etype.get(etype) else {
            return 0;
        };
        self.in_[vi as usize]
            .iter()
            .filter(|a| a.etype == tid)
            .count() as u32
    }

    /// Degree of `vi` over `etype` in `direction` (0 = out, 1 = in).
    fn degree_dir(&self, vi: u32, etype: &str, direction: u8) -> u32 {
        if direction == 0 {
            self.out_degree(vi, etype)
        } else {
            self.in_degree(vi, etype)
        }
    }

    /// Declare a CARDINALITY constraint bounding the degree of every vertex
    /// carrying `label` over `etype` in `direction` (0 = out, 1 = in) to
    /// `min..=max` (`max: None` unbounded). Re-declaring `(label, etype,
    /// direction)` replaces the bounds. Fails with `ConstraintViolation` if any
    /// existing vertex already violates it (mirrors unique/required declare-time).
    pub fn create_cardinality_constraint(
        &mut self,
        label: &str,
        etype: &str,
        direction: u8,
        min: u32,
        max: Option<u32>,
    ) -> CodeResult<()> {
        if let Some(lid) = self.labels.get(label) {
            for vi in self.vertex_indices() {
                if !self.vlabels[vi as usize].contains(&lid) {
                    continue;
                }
                let d = self.degree_dir(vi, etype, direction);
                if d < min || max.is_some_and(|m| d > m) {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the cardinality constraint being declared",
                    ));
                }
            }
        }
        let rule = CardinalityRule {
            label: label.to_string(),
            etype: etype.to_string(),
            direction,
            min,
            max,
        };
        if let Some(existing) = self.v_cardinality.iter_mut().find(|c| {
            c.label == rule.label && c.etype == rule.etype && c.direction == rule.direction
        }) {
            *existing = rule;
        } else {
            self.v_cardinality.push(rule);
        }
        Ok(())
    }

    /// Drop a cardinality constraint on `(label, etype, direction)`. Idempotent.
    pub fn drop_cardinality_constraint(&mut self, label: &str, etype: &str, direction: u8) {
        self.v_cardinality
            .retain(|c| !(c.label == label && c.etype == etype && c.direction == direction));
    }

    /// Every declared cardinality constraint as sorted `(label, etype, direction,
    /// min, max)` tuples — introspection, sorted for a deterministic listing.
    pub fn cardinality_constraints(&self) -> Vec<(String, String, u8, u32, Option<u32>)> {
        let mut out: Vec<(String, String, u8, u32, Option<u32>)> = self
            .v_cardinality
            .iter()
            .map(|c| (c.label.clone(), c.etype.clone(), c.direction, c.min, c.max))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        out
    }

    // --- VALIDATORS (custom GQL-predicate constraints) -----------------------

    /// Declare a VALIDATOR on `label` (a vertex label OR an edge type): every
    /// element carrying `label` must satisfy the GQL boolean `predicate`, with the
    /// element bound to `var`. Appends (a label may carry several). The predicate
    /// is parsed+lowered once here. Two failure modes, distinguished by error code
    /// so the FFI can map them: an unparseable predicate returns
    /// `ErrorCode::Syntax`; existing data that already evaluates to a definite
    /// `false` returns `ErrorCode::ConstraintViolation` (the declare-time scan).
    /// SQL-`CHECK` semantics — a null/unknown result passes.
    pub fn create_validator(&mut self, label: &str, var: &str, predicate: &str) -> CodeResult<()> {
        let expr = crate::gql::parser::parse_predicate(predicate)
            .map_err(|e| CodeError::new(ErrorCode::Syntax, e.message))?;

        // Reject a predicate that references any variable *other* than the declared
        // `var` at DECLARE time. Such a name (`x.age` when the binding is `u`, or a
        // bare `age`) is unbound → the predicate reads UNKNOWN → the SQL-`CHECK`
        // never fires and the validator silently does nothing. A predicate with no
        // variable at all (a constant like `1 = 1`) is legitimately allowed. Uses
        // `ErrorCode::Syntax` (the FFI already maps a bad predicate to `-2`/`E_SYNTAX`)
        // so both engines reject identically.
        if let Some(name) = crate::gql::plan::free_predicate_vars(&expr)
            .into_iter()
            .find(|n| n != var)
        {
            return Err(CodeError::new(
                ErrorCode::Syntax,
                format!(
                    "validator predicate references unbound variable `{name}` \
                     (only the declared variable `{var}` is in scope)"
                ),
            ));
        }

        let pred = crate::gql::plan::lower_predicate(var, &expr);

        // Declare-time scan: reject if any existing element carrying `label` (a
        // vertex OR an edge — one namespace) currently evaluates to a definite
        // false. An already-violated validator is meaningless (mirrors the other
        // constraints). A predicate evaluation fault (e.g. an unknown function)
        // surfaces verbatim via `?`.
        if let Some(lid) = self.labels.get(label) {
            for vi in self.vertex_indices() {
                if self.vlabels[vi as usize].contains(&lid)
                    && crate::gql::eval::eval_predicate(
                        self,
                        &pred,
                        crate::gql::eval::Val::Node(vi),
                    )? == Some(false)
                {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the validator being declared",
                    ));
                }
            }
        }

        if let Some(tid) = self.etype.get(label) {
            // `edges_with_etype` borrows `self.by_etype`; copy the indices out so the
            // per-edge `eval_predicate(self, …)` isn't a second overlapping borrow.
            let eids: Vec<u32> = self.edges_with_etype(tid).to_vec();
            for ei in eids {
                if self.is_edge_live(ei)
                    && crate::gql::eval::eval_predicate(
                        self,
                        &pred,
                        crate::gql::eval::Val::Edge(ei),
                    )? == Some(false)
                {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the validator being declared",
                    ));
                }
            }
        }

        self.v_validators
            .entry(label.to_string())
            .or_default()
            .push(ValidatorRule {
                var: var.to_string(),
                src: predicate.to_string(),
                pred,
            });
        Ok(())
    }

    /// Drop every validator declared on `label`. Idempotent.
    pub fn drop_validator(&mut self, label: &str) {
        self.v_validators.remove(label);
    }

    /// Every declared validator as `(label, var, src)`, sorted by `(label, src)`.
    /// The compiled predicate is internal. Introspection for tests/tooling.
    pub fn validators(&self) -> Vec<(String, String, String)> {
        let mut out: Vec<(String, String, String)> = self
            .v_validators
            .iter()
            .flat_map(|(label, rules)| {
                rules
                    .iter()
                    .map(move |r| (label.clone(), r.var.clone(), r.src.clone()))
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)));
        out
    }

    /// Check every validator declared on a touched vertex `vi`. `Ok(())` if all
    /// pass (a null/unknown result passes); `Err` on a definite `false` or an
    /// evaluation fault. The commit-time check (the eager per-write gate is the
    /// statement's auto-commit, which runs this via `run_deferred_checks`).
    pub(crate) fn check_validators_vertex(&self, vi: u32) -> CodeResult<()> {
        if self.v_validators.is_empty() {
            return Ok(());
        }
        for &lid in &self.vlabels[vi as usize] {
            let name = self.labels.text(lid);
            if let Some(rules) = self.v_validators.get(name) {
                for rule in rules {
                    if crate::gql::eval::eval_predicate(
                        self,
                        &rule.pred,
                        crate::gql::eval::Val::Node(vi),
                    )? == Some(false)
                    {
                        return Err(CodeError::new(
                            ErrorCode::ConstraintViolation,
                            format!("validator '{}' on '{}' violated", rule.src, name),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Edge analogue of [`Graph::check_validators_vertex`].
    pub(crate) fn check_validators_edge(&self, ei: u32) -> CodeResult<()> {
        if self.v_validators.is_empty() {
            return Ok(());
        }
        for name in self.edge_type_names(ei) {
            if let Some(rules) = self.v_validators.get(&name) {
                for rule in rules {
                    if crate::gql::eval::eval_predicate(
                        self,
                        &rule.pred,
                        crate::gql::eval::Val::Edge(ei),
                    )? == Some(false)
                    {
                        return Err(CodeError::new(
                            ErrorCode::ConstraintViolation,
                            format!("validator '{}' on '{}' violated", rule.src, name),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Declare a graph-level INVARIANT `name` = a whole-graph GQL `query` that must
    /// hold after every write transaction. The query is parsed+lowered once here;
    /// an unparseable query returns [`ErrorCode::Syntax`] (mapped to `-2` at the
    /// FFI). VIOLATED iff any cell in its result set is boolean `false` (everything
    /// else — `true`/`null`/non-boolean/empty — holds). A declare-time run rejects
    /// with [`ErrorCode::ConstraintViolation`] if the current graph already
    /// violates it (an already-broken invariant is meaningless, mirroring the
    /// validators/constraints). Re-declaring the same `name` replaces the prior
    /// query. Byte-identical with the TS `createInvariant`.
    pub fn create_invariant(&mut self, name: &str, query: &str) -> CodeResult<()> {
        let plan =
            crate::gql::prepare(query).map_err(|e| CodeError::new(ErrorCode::Syntax, e.message))?;

        // Declare-time run against the current graph: reject on a definite-`false`
        // cell (or surface an evaluation fault verbatim via `?`).
        let rows = crate::gql::run_invariant(&plan, self)?;
        if Self::invariant_violated(&rows) {
            return Err(CodeError::new(
                ErrorCode::ConstraintViolation,
                format!("existing data already violates the invariant '{name}'"),
            ));
        }

        // Replace any prior invariant of the same name (declare is idempotent-ish:
        // last query wins), then append.
        self.v_invariants.retain(|r| r.name != name);
        self.v_invariants.push(InvariantRule {
            name: name.to_string(),
            src: query.to_string(),
            plan: std::sync::Arc::new(plan),
        });
        Ok(())
    }

    /// Drop the graph-level invariant named `name`. Idempotent.
    pub fn drop_invariant(&mut self, name: &str) {
        self.v_invariants.retain(|r| r.name != name);
    }

    /// Every declared invariant as `(name, src)`, sorted by name. The compiled
    /// query plan is internal. Introspection for tests/tooling.
    pub fn invariants(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = self
            .v_invariants
            .iter()
            .map(|r| (r.name.clone(), r.src.clone()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// The full active schema as a JSON array of replayable **op objects** — the
    /// read side of [`create_*`] (the inverse of applying them). Each element
    /// mirrors the TS `SchemaOp` union (`{"op":"createUniqueConstraint","label":…,
    /// "key":…}`, …), emitted in a fixed section order (indexes → node constraints →
    /// edge constraints → cardinality → validators → invariants), each section
    /// sorted, so the output is **deterministic** for a given schema. The snapshot
    /// codec calls this to persist schema alongside the graph NDJSON; a cold boot
    /// replays each op via `applySchemaOp`, so a restored replica keeps the
    /// constraints/validators/indexes it can't reconstruct from data alone.
    pub fn dump_schema(&self) -> String {
        let mut ops: Vec<String> = Vec::new();

        for k in self.vertex_indexes() {
            ops.push(schema_op("createVertexIndex", &[("key", Jv::S(&k))]));
        }
        for k in self.edge_indexes() {
            ops.push(schema_op("createEdgeIndex", &[("key", Jv::S(&k))]));
        }
        // Edge interval (RI-tree) indexes: emitted as a granular op carrying the
        // `[loKey, hiKey)` pair, so an as-of/overlap accelerator survives a
        // snapshot reload and replicates over the CDC schema stream (like the
        // hash indexes above). `applySchemaOp` routes it back to `createIndex`.
        for (lo, hi) in self.edge_interval_index_specs() {
            ops.push(schema_op(
                "createEdgeIntervalIndex",
                &[("loKey", Jv::S(lo)), ("hiKey", Jv::S(hi))],
            ));
        }
        for (label, key) in self.unique_constraints() {
            ops.push(schema_op(
                "createUniqueConstraint",
                &[("label", Jv::S(&label)), ("key", Jv::S(&key))],
            ));
        }
        for (label, key) in self.required_constraints() {
            ops.push(schema_op(
                "createRequiredConstraint",
                &[("label", Jv::S(&label)), ("key", Jv::S(&key))],
            ));
        }
        // Node type constraints carry the type name (scalar OR a `record{…}`),
        // which the `(label, key)` readers drop — iterate both maps directly
        // (sorted for determinism), so a record constraint round-trips too.
        let mut vtypes: Vec<(String, String, String)> = self
            .v_type
            .iter()
            .flat_map(|(label, ks)| {
                ks.iter().map(move |(k, t)| {
                    // Round-trip a scalar `NOT NULL` type constraint.
                    let nn = self
                        .v_type_not_null
                        .get(label)
                        .is_some_and(|s| s.contains(k));
                    let ty = if nn {
                        format!("{} NOT NULL", t.to_name())
                    } else {
                        t.to_name().to_string()
                    };
                    (label.clone(), k.clone(), ty)
                })
            })
            .chain(self.v_record.iter().flat_map(|(label, ks)| {
                ks.iter().map(move |(k, spec)| {
                    let nn = self
                        .v_type_not_null
                        .get(label)
                        .is_some_and(|s| s.contains(k));
                    let ty = if nn {
                        format!("{} NOT NULL", spec.to_name())
                    } else {
                        spec.to_name()
                    };
                    (label.clone(), k.clone(), ty)
                })
            }))
            .collect();
        vtypes.sort();
        for (label, key, ty) in vtypes {
            ops.push(schema_op(
                "createTypeConstraint",
                &[
                    ("label", Jv::S(&label)),
                    ("key", Jv::S(&key)),
                    ("type", Jv::S(&ty)),
                ],
            ));
        }
        for (etype, key) in self.edge_unique_constraints() {
            ops.push(schema_op(
                "createEdgeUniqueConstraint",
                &[("edgeType", Jv::S(&etype)), ("key", Jv::S(&key))],
            ));
        }
        for (etype, key) in self.edge_required_constraints() {
            ops.push(schema_op(
                "createEdgeRequiredConstraint",
                &[("edgeType", Jv::S(&etype)), ("key", Jv::S(&key))],
            ));
        }
        let mut etypes: Vec<(String, String, String)> = self
            .e_type_constraints
            .iter()
            .flat_map(|(et, ks)| {
                ks.iter().map(move |(k, t)| {
                    let nn = self.e_type_not_null.get(et).is_some_and(|s| s.contains(k));
                    let ty = if nn {
                        format!("{} NOT NULL", t.to_name())
                    } else {
                        t.to_name().to_string()
                    };
                    (et.clone(), k.clone(), ty)
                })
            })
            .chain(self.e_record.iter().flat_map(|(et, ks)| {
                ks.iter().map(move |(k, spec)| {
                    let nn = self.e_type_not_null.get(et).is_some_and(|s| s.contains(k));
                    let ty = if nn {
                        format!("{} NOT NULL", spec.to_name())
                    } else {
                        spec.to_name()
                    };
                    (et.clone(), k.clone(), ty)
                })
            }))
            .collect();
        etypes.sort();
        for (etype, key, ty) in etypes {
            ops.push(schema_op(
                "createEdgeTypeConstraint",
                &[
                    ("edgeType", Jv::S(&etype)),
                    ("key", Jv::S(&key)),
                    ("type", Jv::S(&ty)),
                ],
            ));
        }
        for (label, etype, dir, min, max) in self.cardinality_constraints() {
            let direction = if dir == 0 { "out" } else { "in" };
            ops.push(schema_op(
                "createCardinalityConstraint",
                &[
                    ("label", Jv::S(&label)),
                    ("edgeType", Jv::S(&etype)),
                    ("direction", Jv::S(direction)),
                    ("min", Jv::N(min)),
                    ("max", Jv::NOpt(max)),
                ],
            ));
        }
        for (label, var, predicate) in self.validators() {
            ops.push(schema_op(
                "createValidator",
                &[
                    ("label", Jv::S(&label)),
                    ("varName", Jv::S(&var)),
                    ("predicate", Jv::S(&predicate)),
                ],
            ));
        }
        for (name, query) in self.invariants() {
            ops.push(schema_op(
                "createInvariant",
                &[("name", Jv::S(&name)), ("query", Jv::S(&query))],
            ));
        }

        let mut json = String::from("[");
        json.push_str(&ops.join(","));
        json.push(']');
        json
    }

    /// `false`-only-fails: a result set VIOLATES an invariant iff any cell is a
    /// boolean `false`. A `true`, a `null`, a non-boolean value (number/string/
    /// list/map/temporal), or an empty result set all HOLD. Byte-identical to the
    /// TS `invariantViolated` (`cell === false`).
    fn invariant_violated(rows: &crate::rowset::RowSet) -> bool {
        rows.data.iter().any(|v| matches!(v, Value::Bool(false)))
    }

    /// Run every declared invariant against the fully-staged graph. Called from
    /// [`Graph::commit_tx`] only when the transaction actually wrote something.
    /// `Ok(())` if all hold; `Err` carrying the failing invariant's error (a
    /// `ConstraintViolation` for a `false` cell, or an evaluation fault's own code).
    pub(crate) fn check_invariants(&mut self) -> CodeResult<()> {
        if self.v_invariants.is_empty() {
            return Ok(());
        }
        // Move the rules out so the read-only `run_invariant(&plan, self)` can take
        // `&mut self` without overlapping the borrow; the run never mutates the
        // registry, so restoring the same Vec afterwards is exact.
        let rules = std::mem::take(&mut self.v_invariants);
        let mut failure: Option<CodeError> = None;
        for rule in &rules {
            match crate::gql::run_invariant(&rule.plan, self) {
                Ok(rows) if Self::invariant_violated(&rows) => {
                    failure = Some(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        format!("invariant '{}' violated", rule.name),
                    ));
                    break;
                }
                Ok(_) => {}
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
        }
        self.v_invariants = rules;
        match failure {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// True iff a touched vertex `vi` violates any cardinality constraint on one of
    /// its labels (degree below `min` or above `max`). The commit-time check.
    pub(crate) fn cardinality_violation(&self, vi: u32) -> bool {
        if self.v_cardinality.is_empty() {
            return false;
        }
        let lids = &self.vlabels[vi as usize];
        for c in &self.v_cardinality {
            let Some(lid) = self.labels.get(&c.label) else {
                continue;
            };
            if !lids.contains(&lid) {
                continue;
            }
            let d = self.degree_dir(vi, &c.etype, c.direction);
            if d < c.min || c.max.is_some_and(|m| d > m) {
                return true;
            }
        }
        false
    }

    /// Note both endpoints of edge `ei` as touched for the commit-time cardinality
    /// recheck (their degree changed). No-op outside a transaction / during a
    /// rollback replay, or when no cardinality constraint is declared. Called by
    /// the edge write paths (`add_edge` / `remove_edge`), so a vertex-delete
    /// cascade re-checks the surviving neighbor too — mirrors the TS core, whose
    /// `insertEdge` / `removeEdge` note endpoints at the same core boundary.
    pub(crate) fn cardinality_note_endpoints(&mut self, ei: u32) {
        if self.v_cardinality.is_empty() || !self.tx_active() {
            return;
        }
        let i = ei as usize;
        let (from, to) = (self.e_src[i], self.e_dst[i]);
        self.tx_touched.push(from);
        self.tx_touched.push(to);
    }

    /// Live vertices carrying `label` whose property `key == value`. Seeks the
    /// backing index (a constraint always creates one), falling back to a scan if
    /// somehow unindexed. Non-indexable values (null/list) yield an empty set —
    /// exempt from uniqueness (SQL: NULLs distinct), matching the value index.
    fn vertices_with_label_value(&self, label: &str, key: &str, value: &Value) -> Vec<u32> {
        let Some(idxk) = IdxKey::from_value(value) else {
            return Vec::new();
        };
        let Some(lid) = self.labels.get(label) else {
            return Vec::new();
        };
        match self.vertices_by_prop(key, &idxk) {
            Some(ids) => ids
                .iter()
                .copied()
                .filter(|&v| self.vlabels[v as usize].contains(&lid))
                .collect(),
            None => self
                .vertex_indices()
                .filter(|&v| {
                    self.vlabels[v as usize].contains(&lid)
                        && self.props.value(v as usize, key, &self.strs) == *value
                })
                .collect(),
        }
    }

    /// The first pair of live `label`-vertices that share a value for `key` — for
    /// validating a unique constraint against existing data at declare time.
    /// Reuses the (freshly built) backing index; null/list values are exempt.
    fn first_label_prop_duplicate(&self, label: &str, key: &str) -> Option<(u32, u32)> {
        let lid = self.labels.get(label)?;
        let bt = self.vidx.get(key)?;
        for ids in bt.values() {
            let mut with_label = ids
                .iter()
                .copied()
                .filter(|&v| self.vlabels[v as usize].contains(&lid));
            if let (Some(a), Some(b)) = (with_label.next(), with_label.next()) {
                return Some((a, b));
            }
        }
        None
    }

    /// Equality seek over vertices: live vertices whose `key` == `value` (None = no index).
    pub fn vertices_by_prop(&self, key: &str, value: &IdxKey) -> Option<&[u32]> {
        Some(
            self.vidx
                .get(key)?
                .get(value)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        )
    }
    /// Equality seek over edges.
    pub fn edges_by_prop(&self, key: &str, value: &IdxKey) -> Option<&[u32]> {
        Some(
            self.eidx
                .get(key)?
                .get(value)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        )
    }
    /// Range seek over vertices (union of buckets in `bound`, type-block bounded).
    pub fn vertices_by_prop_range(&self, key: &str, bound: &RangeBound) -> Option<Vec<u32>> {
        range_seek(self.vidx.get(key)?, bound)
    }
    /// Range seek over edges.
    pub fn edges_by_prop_range(&self, key: &str, bound: &RangeBound) -> Option<Vec<u32>> {
        range_seek(self.eidx.get(key)?, bound)
    }

    /// The cached CSR snapshot, built (once) from `out`/`in_` on first use and
    /// reused until a topology mutation drops it. Disjoint-field capture lets the
    /// init closure read `out`/`in_` while `get_or_init` holds `csr`.
    /// Drop the CSR snapshot — called by every topology mutation so a later read
    /// rebuilds it. A no-op cost when it was never built.
    pub(crate) fn invalidate_csr(&mut self) {
        self.csr.take();
        self.csr_reads
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// One vertex's adjacency slots, `out` selecting the forward index.
    ///
    /// The CSR snapshot is a **cache-locality optimization for bulk traversal, not
    /// a correctness requirement**: a single vertex's slots are equally available
    /// from the `out`/`in_` delta, in O(degree), and in the identical order
    /// (`csr_pack` concatenates the per-vertex `Vec`s as-is). Building it is
    /// O(V+E), so serving reads exclusively through `get_or_init` meant the first
    /// read after *every* write repacked the entire graph — an interleaved
    /// write→read workload paid O(V+E) per read and went quadratic overall, while
    /// warm read-only scans looked perfectly fine.
    ///
    /// So: use the snapshot when it already exists, otherwise read the delta and
    /// only rebuild once enough reads have accumulated to amortize the repack. A
    /// bulk scan crosses the threshold almost immediately and gets its locality; a
    /// write-heavy workload never pays for a snapshot it would discard.
    #[inline]
    fn adj(&self, v: u32, out: bool) -> &[Adj] {
        if let Some(c) = self.csr.get() {
            return if out { c.out(v) } else { c.in_(v) };
        }

        if self
            .csr_reads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            >= CSR_WARM_READS
        {
            let c = self.csr.get_or_init(|| Csr::build(&self.out, &self.in_));

            return if out { c.out(v) } else { c.in_(v) };
        }

        let delta = if out { &self.out } else { &self.in_ };

        delta.get(v as usize).map_or(&[][..], Vec::as_slice)
    }

    /// Out-edges of `v` as adjacency slots.
    pub fn out_adj(&self, v: u32) -> impl Iterator<Item = Adj> + '_ {
        self.adj(v, true).iter().copied()
    }
    /// In-edges of `v` as adjacency slots (the reverse index).
    pub fn in_adj(&self, v: u32) -> impl Iterator<Item = Adj> + '_ {
        self.adj(v, false).iter().copied()
    }
    /// Labels carried by vertex `v`, as label ids.
    pub fn vertex_labels(&self, v: u32) -> &[u32] {
        &self.vlabels[v as usize]
    }
    /// Does vertex `v` carry label id `l`?
    /// REJECTED, measured: the edge treatment — a dense `v_label0` first label
    /// plus this list for the rest, and an `extra_mask` Bloom filter over labels
    /// that appear as non-first.
    ///
    /// It does not transfer. Edges already HAD a dense first label (every
    /// adjacency entry mirrors it), so the split cost no storage and the Bloom
    /// filter only avoided touching a NEW sparse side table — which is why one
    /// multi-label edge could otherwise cost 2.4x. A vertex has no such dense
    /// label, so the same shape means a net-new parallel array, a second source
    /// of truth to keep in sync across every label mutation, and:
    ///
    /// ```text
    ///   (a:P)-[:R]->(b:Q), 200k vertices     check costs
    ///   single-label vertices                0.28 ms -> 0.19   dense hits
    ///   half the vertices multi-label        0.36    -> 0.42   dense always misses
    /// ```
    ///
    /// It helps only when the label asked for happens to be stored first, and
    /// costs a wasted compare when it is not. Multi-label vertices measured ~2%
    /// against single-label ones here, so there is no pathology to fix — unlike
    /// edges, where there was.
    pub fn has_label(&self, v: u32, l: u32) -> bool {
        self.vlabels[v as usize].contains(&l)
    }
    /// Live vertices carrying label `l`.
    pub fn vertices_with_label(&self, l: u32) -> &[u32] {
        self.by_label.get(&l).map_or(&[], |v| v.as_slice())
    }
    /// Live edges of type id `t` (the seed for `()-[:T]->()` patterns).
    pub fn edges_with_etype(&self, t: u32) -> &[u32] {
        self.by_etype.get(&t).map_or(&[], |e| e.as_slice())
    }
    /// Live edges of type `name`, or `None` if the type was never interned.
    pub fn edges_with_etype_name(&self, name: &str) -> Option<&[u32]> {
        self.etype.get(name).map(|t| self.edges_with_etype(t))
    }

    /// The id of edge `eidx`: its assigned external id, or — since every edge has
    /// an id — the canonical `e{index}` derived from its dense index. The
    /// synthetic id is computed on demand, so the id overlay stays lazy and the
    /// load path pays nothing. Used by codecs (which always emit it) and the
    /// engines' `id()` step.
    /// [`Graph::edge_id`] as an `Arc<str>`, without copying a stored one.
    ///
    /// An assigned id is ALREADY an `Arc<str>`; handing it back through `Cow` and
    /// rebuilding it with `Arc::from` allocates a fresh copy of a string that is
    /// already refcounted. A vertex id goes through `vid.arc(i)` and is a
    /// refcount bump, which is why `g.E().id()` cost 47ns an edge where
    /// `g.V().id()` cost 9ns a vertex.
    ///
    /// The SYNTHESIZED form still allocates: `e{n}` for an id-less edge does not
    /// exist until someone asks for it.
    #[must_use]
    pub fn edge_id_arc(&self, eidx: u32) -> std::sync::Arc<str> {
        match self.eid_fwd.get(&eidx) {
            Some(s) => s.clone(),
            None => std::sync::Arc::from(format!("e{eidx}").as_str()),
        }
    }

    pub fn edge_id(&self, eidx: u32) -> std::borrow::Cow<'_, str> {
        match self.eid_fwd.get(&eidx) {
            Some(s) => std::borrow::Cow::Borrowed(s.as_ref()),
            None => std::borrow::Cow::Owned(format!("e{eidx}")),
        }
    }
    /// The edge carrying id `id` — the reverse of [`Graph::edge_id`]. Resolves an
    /// assigned external id first, then the canonical `e{index}` form of a live,
    /// id-less edge (an explicit id shadows a colliding `e{n}`).
    pub fn edge_by_id(&self, id: &str) -> Option<u32> {
        if let Some(&e) = self.eid_rev.get(id) {
            return Some(e);
        }
        let n: u32 = id.strip_prefix('e')?.parse().ok()?;
        self.is_edge_live(n).then_some(n)
    }

    /// The dense index of the vertex with external `id`, or `None`. Non-mutating
    /// (unlike `vid.intern`) — used to detect id clashes on bulk append.
    pub fn vertex_by_id(&self, id: &str) -> Option<u32> {
        self.vid.get(id)
    }
}
