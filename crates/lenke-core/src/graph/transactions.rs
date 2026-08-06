//! Transactions and reactive change-tracking on `Graph`: the atomic mutation
//! boundary (begin/commit/rollback via an undo-log), deferred constraint checks
//! at commit, buffered events, and the version/epoch mutation counters. A separate
//! `impl Graph` block; shares helper types via `use super::*`.
use super::*;

/// Collapse repeated keys in a new element's property list to their **last**
/// value. `set_value` already stores last-wins (`INSERT (:P {k: 1, k: 2})` keeps
/// `2`, and the TS engine agrees), but the insert loop below applies the index
/// once per PAIR — so without this collapse a repeated key left one live index
/// entry per repeat while storage kept only the final value. Those stale entries
/// never produced a wrong-value match (the final WHERE re-verifies) but they did
/// duplicate the element in any candidate set seeded from the index, so the same
/// query answered by a seek returned more rows than the scan.
///
/// The common case has no repeats and returns the list untouched; only a list
/// that actually repeats a key pays for the rebuild.
fn dedupe_props_last_wins(props: Vec<(String, Value)>) -> Vec<(String, Value)> {
    let has_repeat = {
        let mut seen = std::collections::HashSet::with_capacity(props.len());
        props.iter().any(|(k, _)| !seen.insert(k.as_str()))
    };
    if !has_repeat {
        return props;
    }

    let mut kept: Vec<(String, Value)> = Vec::with_capacity(props.len());
    for (k, v) in props.into_iter().rev() {
        if !kept.iter().any(|(seen, _)| *seen == k) {
            kept.push((k, v));
        }
    }
    kept.reverse();

    kept
}

impl Graph {
    // --- reactive change tracking ----------------------------------------

    /// Monotonic mutation counter. An unchanged value means nothing has mutated
    /// since it was last read — the O(1) check a `getSnapshot` uses to return a
    /// referentially-stable snapshot.
    pub fn version(&self) -> u64 {
        self.version
    }
    /// Per-token change epoch for a label / edge-type / property-key `name`
    /// (0 if never touched). Lets a live query recompute only when one of its
    /// declared dependencies actually changed.
    pub fn epoch(&self, name: &str) -> u64 {
        self.epochs.get(name).copied().unwrap_or(0)
    }
    /// Bump the global version (called by every mutation).
    fn bump(&mut self) {
        self.version = self.version.wrapping_add(1);
    }
    /// Bump one token's epoch.
    fn touch(&mut self, name: &str) {
        *self.epochs.entry(name.to_string()).or_insert(0) += 1;
    }

    /// Assign (or replace) edge `eidx`'s external id. No-op for a dead edge.
    pub fn set_edge_id(&mut self, eidx: u32, id: &str) {
        if !self.is_edge_live(eidx) {
            return;
        }
        self.bump();
        // Drop any prior id for this edge (and its reverse entry) before re-binding.
        if let Some(old) = self.eid_fwd.remove(&eidx) {
            self.eid_rev.remove(&old);
        }
        let arc: Arc<str> = Arc::from(id);
        self.eid_fwd.insert(eidx, arc.clone());
        self.eid_rev.insert(arc, eidx);
    }

    // --- transactions -------------------------------------------------------
    // An atomic mutation boundary with rollback + deferred constraint checks.
    // Mechanism: eager-apply + undo-log + deferred-check-at-commit. Writes apply
    // immediately (read-your-writes), each recording an inverse op; the built-in
    // constraint checks defer to commit, run once against the fully-staged graph;
    // on failure the whole transaction rolls back via the undo log. The engine is
    // single-writer and synchronous — no concurrency, MVCC, or isolation levels.
    // Byte-identical to the TS core (`packages/core/src/core/Graph.ts`).

    /// True while a transaction is open and recording writes (not during a
    /// rollback replay). Mutations consult this to decide whether to record undo /
    /// note a touched vertex.
    #[inline]
    pub fn tx_active(&self) -> bool {
        self.tx_depth > 0 && !self.applying_undo
    }

    /// Is a transaction currently open (at any nesting depth)?
    #[inline]
    pub fn in_transaction(&self) -> bool {
        self.tx_depth > 0
    }

    /// Is the active explicit transaction READ ONLY? Set by ISO GQL
    /// `START TRANSACTION READ ONLY`, cleared on commit/rollback. The GQL statement
    /// executor consults this to reject a write statement in a read-only transaction.
    #[inline]
    pub fn tx_read_only(&self) -> bool {
        self.tx_read_only
    }

    /// Set/clear the active transaction's READ ONLY access mode (see [`Graph::tx_read_only`]).
    #[inline]
    pub fn set_tx_read_only(&mut self, read_only: bool) {
        self.tx_read_only = read_only;
    }

    /// The configured operator-chain ceiling. Reads through the config space; kept
    /// as a named getter because the parse sites ask for exactly this one value.
    #[must_use]
    pub fn max_operator_chain(&self) -> usize {
        self.config.limits.operator_chain as usize
    }

    /// Set the operator-chain ceiling. A thin alias for the config setter, kept so
    /// the construction-time `maxOperatorChain` option and the napi binding have a
    /// named entry point.
    pub fn set_max_operator_chain(&mut self, n: usize) {
        self.set_config(crate::graph::ConfigId::LimitsOperatorChain, n as u64);
    }

    /// This graph's settings (see [`crate::graph::GraphConfig`]).
    #[must_use]
    pub fn config(&self) -> &crate::graph::GraphConfig {
        &self.config
    }

    /// This graph's resource ceilings — shorthand for `config().limits`, which is
    /// what every guard site reads.
    #[must_use]
    pub fn limits(&self) -> &crate::graph::GraphLimits {
        &self.config.limits
    }

    /// Set one setting by its stable id. Returns false for an unrecognized id, so
    /// a host talking to an older artifact can report the unknown setting instead
    /// of silently running with the default. A zero value is rejected the same way
    /// — every setting here is a ceiling, and a ceiling of zero would fail every
    /// query, which is never the intent.
    pub fn set_config(&mut self, id: crate::graph::ConfigId, value: u64) -> bool {
        use crate::graph::ConfigId;
        if value == 0 {
            return false;
        }
        match id {
            ConfigId::LimitsRange => self.config.limits.range = value,
            ConfigId::LimitsTrail => self.config.limits.trail = value,
            ConfigId::LimitsIntermediate => self.config.limits.intermediate = value,
            ConfigId::LimitsOperatorChain => self.config.limits.operator_chain = value,
        }
        true
    }

    /// Open a transaction frame. Nesting increments depth; the outermost frame
    /// owns commit/rollback (flat, savepoint-less), matching the TS core.
    pub fn begin_tx(&mut self) {
        self.tx_depth += 1;
    }

    /// Check every interned name added since the last check, and remember how far
    /// we got. The dictionaries are append-only and a well-formed name stays
    /// well-formed, so this is O(names added by this transaction) — normally zero.
    ///
    /// This runs at COMMIT rather than at each write site because the write sites
    /// are many (the GQL evaluator alone mutates through eighteen of them) and a
    /// missed one is invisible: the engine happily builds a graph it then refuses
    /// to read back, since the codec ingestion path *does* validate. Catching it
    /// here means every write path — GQL, Gremlin, the algorithms' `writeProperty`,
    /// and anything added later — is covered by one check that a new caller cannot
    /// forget.
    ///
    /// The watermark advances even when a name is rejected. The transaction is
    /// rolled back so nothing REFERENCES the bad name, but interning is not undone,
    /// and re-reporting a dangling dictionary entry on every later commit would
    /// wedge the graph. A dangling name is inert: serialization walks each
    /// element's own properties, never the dictionary.
    fn validate_new_names(&mut self) -> CodeResult<()> {
        let marks = self.names_checked;
        let lens = [
            self.labels.strings.len(),
            self.etype.strings.len(),
            self.props.keys.strings.len(),
            self.edge_props.keys.strings.len(),
        ];

        self.names_checked = lens;

        for name in self.labels.strings[marks[0]..].iter() {
            validate_label(name)?;
        }
        for name in self.etype.strings[marks[1]..].iter() {
            validate_label(name)?;
        }
        for name in self.props.keys.strings[marks[2]..].iter() {
            validate_prop_key(name)?;
        }
        for name in self.edge_props.keys.strings[marks[3]..].iter() {
            validate_prop_key(name)?;
        }

        Ok(())
    }

    /// Close the current frame. An inner commit just decrements depth. The
    /// outermost commit runs the deferred constraint checks against the fully
    /// staged graph — on failure it rolls the whole transaction back via the undo
    /// log and returns the failure — then discards the undo/touched state.
    pub fn commit_tx(&mut self) -> Result<(), TxCommitError> {
        if self.tx_depth == 0 {
            return Err(TxCommitError::NoTx);
        }
        self.tx_depth -= 1;
        if self.tx_depth > 0 {
            return Ok(()); // an inner commit — the outermost frame finalizes
        }
        // Names first: a malformed one makes the graph unserializable-and-
        // reloadable, so it is not worth running the (more expensive) per-element
        // constraint checks against a graph that is already invalid.
        if let Err(e) = self.validate_new_names() {
            self.apply_undo_and_reset();
            return Err(TxCommitError::MalformedName(e));
        }
        if let Err(e) = self.run_deferred_checks() {
            self.apply_undo_and_reset();
            return Err(e);
        }
        // Graph-level invariants (cross-write assertions): run ONCE against the
        // fully-staged graph, AFTER the per-element deferred checks, but only if
        // this transaction actually wrote something — a pure-read commit skips
        // them (no spurious cost/throw). The undo log is non-empty iff a write was
        // recorded during the frame. On failure, roll the whole transaction back.
        if !self.tx_undo.is_empty() {
            if let Err(e) = self.check_invariants() {
                self.apply_undo_and_reset();
                return Err(TxCommitError::Invariant(e));
            }
        }
        // Snapshot the touched vertices so a caller can derive this write's
        // value-scope (`last_write_scope`) after the transaction closes. Only a
        // write leaves a non-empty undo log — a pure-read commit clears the snapshot.
        if self.tx_undo.is_empty() {
            self.last_touched.clear();
        } else {
            self.last_touched.clone_from(&self.tx_touched);
        }
        self.tx_undo.clear();
        self.tx_touched.clear();
        self.tx_touched_edges.clear();
        Ok(())
    }

    /// Roll the current transaction back: replay the undo log in reverse, discard
    /// the touched set. A no-op if no transaction is open. Idempotent.
    pub fn rollback_tx(&mut self) {
        if self.tx_depth == 0 {
            return;
        }
        self.apply_undo_and_reset();
    }

    /// Record an inverse op to replay on rollback (no-op outside a transaction or
    /// during an undo replay).
    #[inline]
    fn record_undo(&mut self, inverse: Undo) {
        if self.tx_active() {
            self.tx_undo.push(inverse);
        }
    }

    /// Note a vertex whose built-in constraints must be re-checked at commit. The
    /// per-write gates (in the GQL eval layer) call this instead of throwing
    /// immediately while a transaction is open, so an intermediate state — a node
    /// added before its mandatory property, two rows that momentarily collide —
    /// doesn't trip a constraint the final state satisfies.
    #[inline]
    pub fn tx_note_touched(&mut self, vi: u32) {
        if self.tx_active() {
            self.tx_touched.push(vi);
        }
    }

    /// Note an edge whose built-in edge constraints must be re-checked at commit —
    /// the edge analogue of [`Graph::tx_note_touched`] (deferred to commit for edges).
    #[inline]
    pub fn tx_note_touched_edge(&mut self, ei: u32) {
        if self.tx_active() {
            self.tx_touched_edges.push(ei);
        }
    }

    /// The current undo-log depth, for [`Graph::rollback_statement`]. Taken before
    /// a statement's frame opens so its writes can be undone without touching the
    /// writes an enclosing transaction already staged.
    #[inline]
    pub fn tx_undo_mark(&self) -> usize {
        self.tx_undo.len()
    }

    /// Roll back only the writes recorded since `mark`, closing ONE frame and
    /// leaving any enclosing transaction **open and usable**.
    ///
    /// This is what a failing *statement* inside an explicit transaction needs:
    /// per-statement atomicity (a faulting statement leaves no trace) without
    /// destroying the surrounding transaction. [`Graph::rollback_tx`] cannot serve
    /// here — it resets `tx_depth` to 0 unconditionally, so an application that
    /// caught a statement error (probing an optional feature, say) would silently
    /// fall out of its transaction and auto-commit every subsequent write, and the
    /// later `throw` that should roll everything back would find no frame open.
    ///
    /// The touched sets are deliberately not trimmed: a stale entry is harmless
    /// (`run_deferred_checks` skips anything no longer live) and trimming would
    /// need a parallel mark per set.
    pub fn rollback_statement(&mut self, mark: usize) {
        if self.tx_depth == 0 {
            return;
        }
        self.applying_undo = true;
        while self.tx_undo.len() > mark {
            if let Some(u) = self.tx_undo.pop() {
                self.apply_one_undo(u);
            }
        }
        self.applying_undo = false;
        self.tx_depth -= 1;
        if self.tx_depth == 0 {
            self.tx_undo.clear();
            self.tx_touched.clear();
            self.tx_touched_edges.clear();
        }
    }

    /// Replay the undo log newest-first and reset all transaction state to closed.
    fn apply_undo_and_reset(&mut self) {
        self.applying_undo = true;
        let undo = std::mem::take(&mut self.tx_undo);
        for u in undo.into_iter().rev() {
            self.apply_one_undo(u);
        }
        self.applying_undo = false;
        self.tx_depth = 0;
        self.tx_undo.clear();
        self.tx_touched.clear();
        self.tx_touched_edges.clear();
    }

    /// Apply a single inverse op. Runs with `applying_undo == true`, so the
    /// mutation methods it calls neither re-record undo nor re-note touched
    /// vertices — they only restore known-good state and keep the indexes current.
    fn apply_one_undo(&mut self, u: Undo) {
        match u {
            Undo::InsertVertex(vi) => {
                let _ = self.remove_vertex(vi, true);
            }
            Undo::InsertEdge(ei) => self.remove_edge(ei),
            Undo::VProp(vi, key, Some(v)) => self.set_vertex_prop(vi, &key, v),
            Undo::VProp(vi, key, None) => self.remove_vertex_prop(vi, &key),
            Undo::EProp(ei, key, Some(v)) => self.set_edge_prop(ei, &key, v),
            Undo::EProp(ei, key, None) => self.remove_edge_prop(ei, &key),
            Undo::VLabelAdd(vi, name) => self.remove_vertex_label(vi, &name),
            Undo::VLabelRemove(vi, name) => self.add_vertex_label(vi, &name),
            Undo::EType(ei, name) => self.add_edge_label(ei, &name),
            Undo::ETypeRemove(ei, name) => self.remove_edge_label(ei, &name),
            Undo::DeleteVertex { vi, labels } => self.untombstone_vertex(vi, &labels),
            Undo::DeleteEdge { ei, eid } => self.untombstone_edge(ei, eid),
        }
    }

    /// Re-run the built-in vertex constraints (required / type / unique) against
    /// every vertex touched during the transaction, now that all writes are
    /// staged. A vertex added then removed within the transaction is skipped.
    fn run_deferred_checks(&self) -> Result<(), TxCommitError> {
        // `tx_touched` collects one entry PER write (a K-clause `SET` on a vertex
        // pushes it K times), and each per-element check below hydrates the whole
        // property row — O(row width). So a naive pass is O(touches × width), i.e.
        // quadratic in the number of properties a wide `SET` writes. Two guards keep
        // it linear without changing observable behaviour:
        //   1. Skip a side entirely when no constraint of a kind it checks is
        //      declared — the loop body could never return `Err`, so skipping it is
        //      behaviour-identical (and the common no-constraint write pays nothing).
        //   2. De-duplicate touched ids in FIRST-SEEN order — a vertex only needs
        //      rechecking once, and keeping first-seen order means the *first*
        //      violation encountered (hence the returned error) is unchanged.
        // Validators live in `v_validators` keyed by label OR edge type, so both the
        // vertex and edge passes gate on it.
        let check_vertices = !self.v_required.is_empty()
            || !self.v_type.is_empty()
            || !self.v_unique.is_empty()
            || !self.v_cardinality.is_empty()
            || !self.v_validators.is_empty();
        if check_vertices {
            let mut seen = HashSet::with_capacity(self.tx_touched.len());
            for &vi in &self.tx_touched {
                if !seen.insert(vi) {
                    continue; // already checked this vertex in this commit
                }
                if !self.is_vertex_live(vi) {
                    continue; // added then removed within the transaction — nothing to check
                }
                let labels: Vec<String> = self.vlabels[vi as usize]
                    .iter()
                    .map(|&l| self.labels.text(l).to_string())
                    .collect();
                let props = self.vertex_props(vi);
                if self.missing_required(&labels, &props).is_some() {
                    return Err(TxCommitError::Required);
                }
                if self.type_violation(&labels, &props).is_some() {
                    return Err(TxCommitError::Type);
                }
                if self.unique_conflict(&labels, &props, Some(vi)).is_some() {
                    return Err(TxCommitError::Unique);
                }
                // Cardinality: a vertex is touched when added OR when an incident edge
                // is added/removed (either endpoint's degree changed). This commit is
                // where BOTH bounds land — max (also caught eagerly for a direct
                // addEdge on the TS side) and min (commit-time only, since a single
                // write can't satisfy a positive lower bound).
                if self.cardinality_violation(vi) {
                    return Err(TxCommitError::Cardinality);
                }
                // Custom validators (a definite-false predicate, or an evaluation fault
                // like an unknown function) — surfaced with their own carried error.
                if let Err(e) = self.check_validators_vertex(vi) {
                    return Err(TxCommitError::Validator(e));
                }
            }
        }
        // Edge constraints: re-check every edge touched during the transaction
        // against the fully-staged graph (edge analogue of the vertex loop above).
        let check_edges = !self.e_required.is_empty()
            || !self.e_type_constraints.is_empty()
            || !self.e_unique.is_empty()
            || !self.v_validators.is_empty();
        if check_edges {
            let mut seen = HashSet::with_capacity(self.tx_touched_edges.len());
            for &ei in &self.tx_touched_edges {
                if !seen.insert(ei) {
                    continue; // already checked this edge in this commit
                }
                if !self.is_edge_live(ei) {
                    continue; // added then removed within the transaction — nothing to check
                }
                let etypes = self.edge_type_names(ei);
                let props = self.edge_props_of(ei);
                if self.edge_missing_required(&etypes, &props).is_some() {
                    return Err(TxCommitError::Required);
                }
                if self.edge_type_violation(&etypes, &props).is_some() {
                    return Err(TxCommitError::Type);
                }
                if self
                    .edge_unique_conflict(&etypes, &props, Some(ei))
                    .is_some()
                {
                    return Err(TxCommitError::Unique);
                }
                if let Err(e) = self.check_validators_edge(ei) {
                    return Err(TxCommitError::Validator(e));
                }
            }
        }
        Ok(())
    }

    /// A live vertex's present properties as `(key, value)` pairs — the shape the
    /// constraint predicates consume. A stored null is present (and included).
    fn vertex_props(&self, vi: u32) -> Vec<(String, Value)> {
        let i = vi as usize;
        let mut out = Vec::new();
        for kid in 0..self.props.cols.len() as u32 {
            if self.props.is_present_id(i, kid) {
                let key = self.props.keys.text(kid).to_string();
                let val = self.props.value_id(i, kid, &self.strs);
                out.push((key, val));
            }
        }
        out
    }

    /// Reverse a vertex delete: un-tombstone the slot in place (its columns were
    /// never cleared on delete, so property values survive) and rebuild its label
    /// membership + property indexes. Adjacency is repopulated by the incident
    /// edges' own `DeleteEdge` inverses (replayed after this one).
    fn untombstone_vertex(&mut self, vi: u32, labels: &[u32]) {
        let i = vi as usize;
        if self.is_vertex_live(vi) {
            return;
        }
        self.v_live[i] = true;
        self.live_n += 1;
        self.vlabels[i] = labels.to_vec();
        for &lid in labels {
            self.by_label.entry(lid).or_default().push(vi);
        }
        if !self.vidx.is_empty() {
            for key in self.vidx.keys().cloned().collect::<Vec<_>>() {
                let val = self.props.value(i, &key, &self.strs);
                idx_apply(&mut self.vidx, &key, vi, &val, true);
            }
        }
        self.bump();
        let mut names: Vec<String> = labels
            .iter()
            .map(|&l| self.labels.text(l).to_string())
            .collect();
        for kid in 0..self.props.cols.len() as u32 {
            if self.props.is_present_id(i, kid) {
                names.push(self.props.keys.text(kid).to_string());
            }
        }
        for name in names {
            self.touch(&name);
        }
    }

    /// Reverse an edge delete: un-tombstone it in place and restore its type
    /// bucket, both endpoints' adjacency, property indexes, and external-id overlay.
    fn untombstone_edge(&mut self, ei: u32, eid: Option<Arc<str>>) {
        let i = ei as usize;
        if self.is_edge_live(ei) {
            return;
        }
        self.e_live[i] = true;
        self.live_e += 1;
        let tid = self.e_type[i];
        let (src, dst) = (self.e_src[i], self.e_dst[i]);
        self.by_etype.entry(tid).or_default().push(ei);
        self.out[src as usize].push(Adj {
            eidx: ei,
            nbr: dst,
            etype: tid,
        });
        self.in_[dst as usize].push(Adj {
            eidx: ei,
            nbr: src,
            etype: tid,
        });
        if !self.eidx.is_empty() {
            for key in self.eidx.keys().cloned().collect::<Vec<_>>() {
                let val = self.edge_props.value(i, &key, &self.strs);
                idx_apply(&mut self.eidx, &key, ei, &val, true);
            }
        }
        self.interval_idx_insert(ei);
        if let Some(arc) = eid {
            self.eid_fwd.insert(ei, arc.clone());
            self.eid_rev.insert(arc, ei);
        }
        self.bump();
        let mut names: Vec<String> = vec![self.etype.text(tid).to_string()];
        for kid in 0..self.edge_props.cols.len() as u32 {
            if self.edge_props.is_present_id(i, kid) {
                names.push(self.edge_props.keys.text(kid).to_string());
            }
        }
        for name in names {
            self.touch(&name);
        }
    }

    // --- mutation ----------------------------------------------------------

    fn fresh_id(&mut self) -> String {
        loop {
            let id = format!("_n{}", self.synth);
            self.synth += 1;
            if self.vid.get(&id).is_none() {
                return id;
            }
        }
    }

    /// Add a vertex with the given labels and properties; returns its index.
    /// Reject a graph holding a malformed label / edge type / property key (see
    /// [`validate_label`] / [`validate_prop_key`]). One cheap pass over the
    /// interned name dictionaries (distinct names, not per-element). Called at
    /// the codec ingestion boundary so loaded data can't smuggle in a name that
    /// won't round-trip through every codec.
    pub fn validate_wellformed(&self) -> CodeResult<()> {
        for name in self.labels.strings.iter().chain(self.etype.strings.iter()) {
            validate_label(name)?;
        }
        for name in self
            .props
            .keys
            .strings
            .iter()
            .chain(self.edge_props.keys.strings.iter())
        {
            validate_prop_key(name)?;
        }
        Ok(())
    }

    pub fn add_vertex(&mut self, labels: &[String], props: Vec<(String, Value)>) -> u32 {
        let id = self.fresh_id();
        self.add_vertex_with_id(&id, labels, props)
    }

    /// Append a vertex carrying an **explicit** external id (vs `add_vertex`,
    /// which mints one). The id must be fresh — a caller that might collide
    /// checks `vid.get(id)` first (bulk append / merge does). The building block
    /// for id-preserving bulk ingest into a live graph.
    pub fn add_vertex_with_id(
        &mut self,
        id: &str,
        labels: &[String],
        props: Vec<(String, Value)>,
    ) -> u32 {
        // NaN/±Infinity are not values in the LPG numeric model, and every codec
        // entry point already coerces them. This is the *computed*-write side of
        // the same contract — see `Value::finite_only`. Free when there is
        // nothing to coerce, which is every ordinary write.
        let props: Vec<(String, Value)> = props
            .into_iter()
            .map(|(k, v)| (k, v.finite_only()))
            .collect();
        let vi = self.vid.intern(id);
        debug_assert_eq!(vi as usize, self.n, "add_vertex_with_id expects a fresh id");
        self.v_live.push(true);
        self.live_n += 1;
        let lids: Vec<u32> = labels.iter().map(|l| self.labels.intern(l)).collect();
        for &lid in &lids {
            self.by_label.entry(lid).or_default().push(vi);
        }
        self.vlabels.push(lids);
        self.out.push(Vec::new());
        self.in_.push(Vec::new());
        self.props.push_element();
        for (k, v) in dedupe_props_last_wins(props) {
            if self.any_vidx_rooted_at(&k) {
                idx_apply(&mut self.vidx, &k, vi, &v, true);
            }
            self.touch(&k);
            self.props.set_value(vi as usize, &k, v, &mut self.strs);
        }
        self.n += 1;
        // Topology change: drop the CSR snapshot, bump the global version and the
        // new vertex's labels.
        self.invalidate_csr();
        self.bump();
        for l in labels {
            self.touch(l);
        }
        // Undo of an insert = tombstone the slot (detach removes any edges added
        // to it later — but on reverse replay those are already undone).
        self.record_undo(Undo::InsertVertex(vi));
        vi
    }

    /// Add an edge `from -> to` of `etype` with properties; returns its index.
    pub fn add_edge(
        &mut self,
        from: u32,
        to: u32,
        etype: &str,
        props: Vec<(String, Value)>,
    ) -> u32 {
        self.add_edge_labelled(from, to, &[etype], props)
    }

    /// Add an edge carrying SEVERAL labels — the general form.
    ///
    /// Edges are multi-label like vertices, matching the TS engine, where
    /// `Edge.labels` has always been a `Set<string>`. The first label is stored
    /// densely (every adjacency entry carries a copy, and the hot filter is one
    /// `u32` compare); the rest go in the sparse `e_extra`, so a single-label
    /// edge costs exactly what it did.
    ///
    /// An empty label list is not representable — an edge must have at least
    /// one — so it interns the empty string, which is what the single-label form
    /// did with `""` before.
    pub fn add_edge_labelled(
        &mut self,
        from: u32,
        to: u32,
        labels: &[&str],
        props: Vec<(String, Value)>,
    ) -> u32 {
        // NaN/±Infinity are not values in the LPG numeric model, and every codec
        // entry point already coerces them. This is the *computed*-write side of
        // the same contract — see `Value::finite_only`. Free when there is
        // nothing to coerce, which is every ordinary write.
        let props: Vec<(String, Value)> = props
            .into_iter()
            .map(|(k, v)| (k, v.finite_only()))
            .collect();
        let ei = self.e_src.len() as u32;
        let mut ids: Vec<u32> = Vec::with_capacity(labels.len().max(1));

        for l in labels
            .iter()
            .copied()
            .chain(labels.is_empty().then_some(""))
        {
            let id = self.etype.intern(l);

            if !ids.contains(&id) {
                ids.push(id);
            }
        }

        let tid = ids[0];

        if ids.len() > 1 {
            self.e_extra.insert(ei, ids[1..].to_vec());
            self.extra_etypes.extend(ids[1..].iter().copied());
            self.refresh_extra_mask();
        }

        // Every label buckets the edge, exactly as a vertex is bucketed under
        // each of its labels — so a bucket seed for ANY of them finds it.
        for &extra in &ids[1..] {
            self.by_etype.entry(extra).or_default().push(ei);
        }

        self.e_src.push(from);
        self.e_dst.push(to);
        self.e_type.push(tid);
        self.by_etype.entry(tid).or_default().push(ei);
        self.e_live.push(true);
        self.live_e += 1;
        self.out[from as usize].push(Adj {
            eidx: ei,
            nbr: to,
            etype: tid,
        });
        self.in_[to as usize].push(Adj {
            eidx: ei,
            nbr: from,
            etype: tid,
        });
        self.edge_props.push_element();
        for (k, v) in dedupe_props_last_wins(props) {
            if self.any_eidx_rooted_at(&k) {
                idx_apply(&mut self.eidx, &k, ei, &v, true);
            }
            self.touch(&k);
            self.edge_props
                .set_value(ei as usize, &k, v, &mut self.strs);
        }
        // Register the new edge in every interval index it has endpoints for.
        self.interval_idx_insert(ei);
        // Topology change: drop the CSR snapshot, bump the global version and type.
        self.invalidate_csr();
        self.bump();

        for l in labels
            .iter()
            .copied()
            .chain(labels.is_empty().then_some(""))
        {
            self.touch(l);
        }

        self.record_undo(Undo::InsertEdge(ei));
        // Both endpoints' degree changed — note them for the commit-time
        // cardinality recheck (no-op unless inside a transaction with a
        // cardinality constraint declared).
        self.cardinality_note_endpoints(ei);
        ei
    }

    /// Whether vertex `vi`'s `id` property IS its identity — i.e. a string `id`
    /// equal to the external id (as set by `INSERT (:P {id: 'x'})`). A numeric or
    /// absent `id`, or an external id that diverges from it, is an ordinary
    /// property and remains SET-able; a matching string `id` is fixed.
    pub fn vertex_id_is_identity(&self, vi: u32) -> bool {
        matches!(
            self.props.value(vi as usize, "id", &self.strs),
            Value::Str(s) if s.as_ref() == self.vid.text(vi)
        )
    }

    /// The edge analogue of [`Graph::vertex_id_is_identity`]: whether edge `ei`'s
    /// `id` property IS its external id (a string `id` set at INSERT), and so fixed.
    pub fn edge_id_is_identity(&self, ei: u32) -> bool {
        matches!(
            self.edge_props.value(ei as usize, "id", &self.strs),
            Value::Str(s) if s.as_ref() == self.edge_id(ei).as_ref()
        )
    }

    pub fn set_vertex_prop(&mut self, vi: u32, key: &str, v: Value) {
        // NaN/±Infinity are not values in the LPG numeric model, and every codec
        // entry point already coerces them. This is the *computed*-write side of
        // the same contract — see `Value::finite_only`. Free when there is
        // nothing to coerce, which is every ordinary write.
        let v = v.finite_only();

        if self.tx_active() {
            let prior = if self.props.is_present(vi as usize, key) {
                Some(self.props.value(vi as usize, key, &self.strs))
            } else {
                None
            };
            self.record_undo(Undo::VProp(vi, key.to_string(), prior));
        }
        if self.any_vidx_rooted_at(key) {
            let old = self.props.value(vi as usize, key, &self.strs);
            idx_apply(&mut self.vidx, key, vi, &old, false);
        }
        self.props.set_value(vi as usize, key, v, &mut self.strs);
        if self.any_vidx_rooted_at(key) {
            let new = self.props.value(vi as usize, key, &self.strs);
            idx_apply(&mut self.vidx, key, vi, &new, true);
        }
        // Value change: bump only this key (not the element's labels), so a
        // label-only/topology query isn't invalidated by an unrelated edit.
        self.bump();
        self.touch(key);
    }
    pub fn remove_vertex_prop(&mut self, vi: u32, key: &str) {
        if self.tx_active() {
            let prior = if self.props.is_present(vi as usize, key) {
                Some(self.props.value(vi as usize, key, &self.strs))
            } else {
                None
            };
            self.record_undo(Undo::VProp(vi, key.to_string(), prior));
        }
        if self.any_vidx_rooted_at(key) {
            let old = self.props.value(vi as usize, key, &self.strs);
            idx_apply(&mut self.vidx, key, vi, &old, false);
        }
        self.props.remove_value(vi as usize, key);
        self.bump();
        self.touch(key);
    }
    pub fn set_edge_prop(&mut self, ei: u32, key: &str, v: Value) {
        // NaN/±Infinity are not values in the LPG numeric model, and every codec
        // entry point already coerces them. This is the *computed*-write side of
        // the same contract — see `Value::finite_only`. Free when there is
        // nothing to coerce, which is every ordinary write.
        let v = v.finite_only();

        if self.tx_active() {
            let prior = if self.edge_props.is_present(ei as usize, key) {
                Some(self.edge_props.value(ei as usize, key, &self.strs))
            } else {
                None
            };
            self.record_undo(Undo::EProp(ei, key.to_string(), prior));
        }
        // The interval index keys on both endpoints, so a write to either moves the
        // whole interval: drop the old [lo,hi] before the write, re-insert after.
        let touch_interval = self.key_is_interval_endpoint(key);
        if touch_interval {
            self.interval_idx_remove(ei);
        }
        if self.any_eidx_rooted_at(key) {
            let old = self.edge_props.value(ei as usize, key, &self.strs);
            idx_apply(&mut self.eidx, key, ei, &old, false);
        }
        self.edge_props
            .set_value(ei as usize, key, v, &mut self.strs);
        if self.any_eidx_rooted_at(key) {
            let new = self.edge_props.value(ei as usize, key, &self.strs);
            idx_apply(&mut self.eidx, key, ei, &new, true);
        }
        if touch_interval {
            self.interval_idx_insert(ei);
        }
        self.bump();
        self.touch(key);
    }
    pub fn remove_edge_prop(&mut self, ei: u32, key: &str) {
        if self.tx_active() {
            let prior = if self.edge_props.is_present(ei as usize, key) {
                Some(self.edge_props.value(ei as usize, key, &self.strs))
            } else {
                None
            };
            self.record_undo(Undo::EProp(ei, key.to_string(), prior));
        }
        // Removing an endpoint makes the interval incomplete — drop it from the index
        // (read while both endpoints are still present).
        if self.key_is_interval_endpoint(key) {
            self.interval_idx_remove(ei);
        }
        if self.any_eidx_rooted_at(key) {
            let old = self.edge_props.value(ei as usize, key, &self.strs);
            idx_apply(&mut self.eidx, key, ei, &old, false);
        }
        self.edge_props.remove_value(ei as usize, key);
        self.bump();
        self.touch(key);
    }

    pub fn add_vertex_label(&mut self, vi: u32, name: &str) {
        let lid = self.labels.intern(name);
        if !self.vlabels[vi as usize].contains(&lid) {
            self.vlabels[vi as usize].push(lid);
            self.by_label.entry(lid).or_default().push(vi);
            self.bump();
            self.touch(name);
            self.record_undo(Undo::VLabelAdd(vi, name.to_string()));
        }
    }
    pub fn remove_vertex_label(&mut self, vi: u32, name: &str) {
        if let Some(lid) = self.labels.get(name) {
            let had = self.vlabels[vi as usize].contains(&lid);
            self.vlabels[vi as usize].retain(|&x| x != lid);
            if let Some(bucket) = self.by_label.get_mut(&lid) {
                bucket.retain(|&x| x != vi);
            }
            self.bump();
            self.touch(name);
            if had {
                self.record_undo(Undo::VLabelRemove(vi, name.to_string()));
            }
        }
    }

    /// ADD a label to an edge, keeping the ones it already has.
    ///
    /// Edges are multi-label, like vertices — this used to REPLACE the type
    /// (last wins), which is the single-label model and diverged from the TS
    /// engine, where `Edge.labels` is a `Set<string>`.
    ///
    /// The first label stays dense in `e_type` (every adjacency entry mirrors
    /// it); the rest live in the sparse `e_extra`. Adding the SECOND label to an
    /// edge is therefore the moment `has_multi_label_edges` flips on, and
    /// removing it is the moment it flips back — both directions are covered by
    /// `edge_label_transitions_keep_the_fast_path_correct`.
    pub fn add_edge_label(&mut self, ei: u32, name: &str) {
        let tid = self.etype.intern(name);

        if self.edge_has_label(ei, tid) {
            return; // already carried; adding twice is a no-op
        }

        if self.tx_active() {
            self.record_undo(Undo::ETypeRemove(ei, name.to_string()));
        }

        self.e_extra.entry(ei).or_default().push(tid);
        self.extra_etypes.insert(tid);
        self.refresh_extra_mask();

        if self.is_edge_live(ei) {
            self.by_etype.entry(tid).or_default().push(ei);
        }

        self.bump();
        self.touch(name);
    }

    /// Remove ONE label from an edge.
    ///
    /// Removing the first label promotes an extra into its place, which means
    /// rewriting `e_type` and every adjacency mirror — the same work the old
    /// replace-the-type path did. Removing the LAST remaining label leaves the
    /// edge with the empty type, which is what an unlabelled edge has always
    /// been. When the last EXTRA goes the map entry is dropped, so the
    /// single-label fast path re-arms.
    pub fn remove_edge_label(&mut self, ei: u32, name: &str) {
        let Some(tid) = self.etype.get(name) else {
            return; // a name no edge carries
        };

        if !self.edge_has_label(ei, tid) {
            return;
        }

        if self.tx_active() {
            self.record_undo(Undo::EType(ei, name.to_string()));
        }

        if self.e_type[ei as usize] == tid {
            // Promote an extra, or fall back to the empty type.
            let promoted = self
                .e_extra
                .get_mut(&ei)
                .and_then(|extra| (!extra.is_empty()).then(|| extra.remove(0)));

            let next = match promoted {
                Some(next) => next,
                None => self.etype.intern(""),
            };

            self.set_first_edge_label(ei, next);
        } else if let Some(extra) = self.e_extra.get_mut(&ei) {
            extra.retain(|&x| x != tid);
        }

        // An emptied extras list must LEAVE the map: `has_multi_label_edges`
        // reads `e_extra.is_empty()`, so a stale empty entry would keep every
        // adjacency filter on the slow path forever.
        if self.e_extra.get(&ei).is_some_and(Vec::is_empty) {
            self.e_extra.remove(&ei);
        }

        // `extra_etypes` is what keeps the adjacency filter cheap, so a label
        // that is no longer anyone's extra must LEAVE it — otherwise every query
        // for that label stays on the slow path for good. Recomputed rather than
        // refcounted: removals are rare and the map is small.
        if self.e_extra.is_empty() {
            self.extra_etypes.clear();
        } else if !self.e_extra.values().any(|v| v.contains(&tid)) {
            self.extra_etypes.remove(&tid);
        }

        self.refresh_extra_mask();

        if let Some(bucket) = self.by_etype.get_mut(&tid) {
            bucket.retain(|&x| x != ei);
        }

        self.invalidate_csr();
        self.bump();
        self.touch(name);
    }

    /// Replace an edge's FIRST label, updating both adjacency mirrors.
    fn set_first_edge_label(&mut self, ei: u32, tid: u32) {
        let i = ei as usize;

        self.e_type[i] = tid;

        let (src, dst) = (self.e_src[i] as usize, self.e_dst[i] as usize);

        for a in self.out[src].iter_mut().filter(|a| a.eidx == ei) {
            a.etype = tid;
        }

        for a in self.in_[dst].iter_mut().filter(|a| a.eidx == ei) {
            a.etype = tid;
        }

        // Only if it is not already there. This promotes an EXISTING secondary
        // type to first, and the edge was already bucketed under it — pushing
        // again duplicated it, which is invisible to any reader that walks the
        // bucket and wrong for any reader that takes its LENGTH. Both engines'
        // edge-type count shortcuts take the length.
        if self.is_edge_live(ei) {
            let bucket = self.by_etype.entry(tid).or_default();

            if !bucket.contains(&ei) {
                bucket.push(ei);
            }
        }
    }

    /// Delete an edge (tombstone + unlink from both endpoints' adjacency).
    pub fn remove_edge(&mut self, ei: u32) {
        let i = ei as usize;
        if !self.is_edge_live(ei) {
            return;
        }
        // Both endpoints' degree will drop — note them for the commit-time
        // cardinality recheck (min may now be unmet). Endpoints read from the
        // still-intact e_src/e_dst; no-op outside a transaction / rollback replay.
        self.cardinality_note_endpoints(ei);
        // Record the inverse (un-tombstone) before tombstoning: capture any
        // external-id overlay, which the removal below drops.
        if self.tx_active() {
            let eid = self.eid_fwd.get(&ei).cloned();
            self.record_undo(Undo::DeleteEdge { ei, eid });
        }
        // Drop the edge from every edge property index before tombstoning.
        if !self.eidx.is_empty() {
            for key in self.eidx.keys().cloned().collect::<Vec<_>>() {
                let val = self.edge_props.value(i, &key, &self.strs);
                idx_apply(&mut self.eidx, &key, ei, &val, false);
            }
        }
        self.interval_idx_remove(ei);
        // Invalidate the edge's type and every property key it carried.
        let mut touched: Vec<String> = vec![self.etype.text(self.e_type[i]).to_string()];
        for kid in 0..self.edge_props.cols.len() as u32 {
            // Presence, not value: a stored-null key is present and its epoch
            // must still be bumped on delete.
            if self.edge_props.is_present_id(i, kid) {
                touched.push(self.edge_props.keys.text(kid).to_string());
            }
        }
        self.e_live[i] = false;
        self.live_e -= 1;
        // EVERY type the edge carried, not just `e_type`. `by_etype` is what
        // `edges_with_etype_name` scans, so a bucket left holding a dead edge
        // makes a later constraint declare validate against it.
        for tid in self.edge_labels(ei) {
            if let Some(bucket) = self.by_etype.get_mut(&tid) {
                bucket.retain(|&x| x != ei);
            }
        }
        // Drop any external id overlay for this edge.
        if let Some(old) = self.eid_fwd.remove(&ei) {
            self.eid_rev.remove(&old);
        }
        let (src, dst) = (self.e_src[i] as usize, self.e_dst[i] as usize);
        self.out[src].retain(|a| a.eidx != ei);
        self.in_[dst].retain(|a| a.eidx != ei);
        self.invalidate_csr();
        self.bump();
        for name in touched {
            self.touch(&name);
        }
    }

    /// Delete a vertex. Without `detach`, a vertex that still has edges is an
    /// error (ISO/Cypher semantics); with `detach`, incident edges go first.
    pub fn remove_vertex(&mut self, vi: u32, detach: bool) -> CodeResult<()> {
        let i = vi as usize;
        if !self.is_vertex_live(vi) {
            return Ok(());
        }
        let incident: Vec<u32> = self.out[i]
            .iter()
            .chain(self.in_[i].iter())
            .map(|a| a.eidx)
            .collect();
        if !detach && !incident.is_empty() {
            return Err(CodeError::new(
                ErrorCode::InvalidGraphOp,
                "cannot delete a vertex that still has relationships; use DETACH DELETE",
            ));
        }
        for ei in incident {
            self.remove_edge(ei);
        }
        // Invalidate the vertex's labels and every property key it carried
        // (gathered before the columns/labels are cleared below).
        let mut touched: Vec<String> = self.vlabels[i]
            .iter()
            .map(|&l| self.labels.text(l).to_string())
            .collect();
        for kid in 0..self.props.cols.len() as u32 {
            // Presence, not value (stored null is present) — see remove_edge.
            if self.props.is_present_id(i, kid) {
                touched.push(self.props.keys.text(kid).to_string());
            }
        }
        for lid in self.vlabels[i].clone() {
            if let Some(bucket) = self.by_label.get_mut(&lid) {
                bucket.retain(|&x| x != vi);
            }
        }
        // Drop the vertex from every vertex property index.
        if !self.vidx.is_empty() {
            for key in self.vidx.keys().cloned().collect::<Vec<_>>() {
                let val = self.props.value(i, &key, &self.strs);
                idx_apply(&mut self.vidx, &key, vi, &val, false);
            }
        }
        // Capture the labels for the rollback inverse before clearing them (the
        // columns are left intact, so property values survive the tombstone).
        let undo_labels: Vec<u32> = if self.tx_active() {
            self.vlabels[i].clone()
        } else {
            Vec::new()
        };
        self.vlabels[i].clear();
        self.out[i].clear();
        self.in_[i].clear();
        self.v_live[i] = false;
        self.live_n -= 1;
        self.invalidate_csr();
        self.bump();
        for name in touched {
            self.touch(&name);
        }
        // Recorded last (after the cascade's per-edge `DeleteEdge` inverses), so a
        // reverse replay un-tombstones the vertex first, then re-adds its edges.
        self.record_undo(Undo::DeleteVertex {
            vi,
            labels: undo_labels,
        });
        Ok(())
    }
}
