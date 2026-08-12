# GQL burn-down build queue (scout findings)

Condensed implementation plans from 6 read-only scouts (2026-08). Each is grounded
in core (authoritative) + engine insertion points. Baseline after WALK/TRAIL: **419**.
Verify every feature: `cargo test --release --lib`; differential_fuzz seeds 1 & 42
(FUZZ_ITERS 2500-3000, byte-identity); `cargo clippy --all-targets`; then
`CORPUS_BASELINE=1 ... --test gql_corpus` (0 NEW). Do NOT touch lenke-core.

## DONE

- WALK / TRAIL path modes (commit 88010650) — corpus 427->419.
- Line/block comments + standalone ORDER BY before RETURN (commit e22a41c7) — 419->405.
- Edge-label disjunction `-[:A|B]->` — 407->391 (16 cases; node-label disjunction `(n:A|B)` stays deferred).
- TRIM spec-form + round(2-arg)/atan2/log10/list_sort(order,nullOrder) — 391->381 (10 cases).
- INSERT..RETURN binds created nodes (parallel worktree agent, merged 867e2bd3) — 3 cases.
- ACYCLIC / SIMPLE path modes (parallel worktree agent, cherry-picked c6e2b751) — 11 cases. VarLength `trail:bool`→`mode:PathMode`. bare_all_composes_mode / bare_path_binds_simple_cycle stay blocked (need `((..)){n}` subpath group). [Combined parallel re-baseline: 377->363.]
- ALL SHORTEST / SHORTEST 1 selectors (commit 2b480fe3) — 363->350 (13 cases; inline shortest-endpoint props also unblocked ANY-SHORTEST subscript cases). k>=2 + per-hop WHERE deferred.
- SELECT..FROM MATCH **phase 1** (constant / projection / global-agg / GROUP BY via implicit grouping) — 381->377 (4 cases). Refactored `match_body` + `project_and_page` (both now shared with SELECT). **Phase 2 (HAVING) still TODO** — 7 `select_having_*` cases; needs GROUP BY that forces a group with NO agg in the SELECT list, an `extract_aggs` expr-walker to hoist HAVING/ORDER aggregates into the group aggs, and a post-aggregation filter with slot rewriting (keys-then-aggs schema). Currently `SELECT … HAVING` errors (baselined). See queue section 3 Phase 2.

## ~~1. Edge-label disjunction~~ DONE — ~~~24 cases~~ 16 cases

Represent edge type as a LIST (empty = any); "typed but all-unknown" must short-circuit
to no-rows (NOT collapse to "any"); partially-unknown drops unknown names (core lower_labels).

- Lexer: add `Tok::Pipe`; the single-`|` error branch (gql.rs ~223-230) emits Pipe when next char != `|` (keep `||`->Concat).
- Rel: `Rel.etype: Option<String>` -> `etypes: Vec<String>`; rel() parses `:id (Pipe id)*`.
- IR: `edge_label: Option<String>` -> `Vec<String>` on Expand/OptionalExpand/IntervalExpand/VarLength/ShortestPath (ir.rs 317,333,354,373,387); 5 builder sigs take `&[String]`.
- gql.rs construction sites: 664,952-954(INSERT: guard etypes.len()==1 -> error),1147(var_length),1159,1185,1255-1259.
- exec: add `want_etypes(store,&[String]) -> Ok(vec![])any | Err(())all-unknown | Ok(ids)`; convert the 15 etype_id sites (2102..4827) preserving early-returns.
- `for_each_nbr` (exec ~2538): `want:&[u32]`; index fast path ONLY when `want.len()==1` (multi-type MUST flat-scan for byte-identity); filter `want.is_empty()||want.contains(&et)`.
- DFS walkers (varlen_count_dfs 3099/3135 + outdeg 3057, varlen_agg_dfs 3268/3303, varlen_dfs 4733/4775, shortest_path BFS 4868, try_3hop_product_count hit 4205): `want:&[u32]`, filter `!want.is_empty() && !want.contains(&et)`.
- cost.rs unknown_edge (234): all-unknown semantics. opt.rs ~20 clone-through sites compile as-is. gremlin.rs 588-670 pass `&[]`/`&[label]`.
- GOTCHA: order — never union out_typed buckets for multi-type. Node-label disjunction `(n:A|B)` is OUT of scope (5 cases stay blocked).

## 2. TRIM spec-form + scalar fns — ~10 cases — several independent easy fixes

- TRIM (tests_e trim_sql_spec_form): parser special-case in `fn call` (gql.rs ~2409) BEFORE generic args: `if name eq_ignore "trim" return parse_trim_body()`. parse_trim_body: LEADING->"ltrim"/TRAILING->"rtrim"/BOTH(default)->"trim"; then `[char] FROM src`: if eat FROM -> args=[expr]; else e1, if FROM -> args=[expr, e1] (char is 2nd); else args=[e1]. Eval fix exec.rs ~5952: route `"trim"` -> `trim_fn("btrim", args)` (honors optional char set; today it ignores arg1).
- round(x[,digits]) (tests_b round_digits/round_neg_digits): parser arity gql.rs 2433/2448 -> 1-or-2; new `"round"` arm before exec.rs 5920 replicating core: `digits=trunc(arg1) as i32 default 0; f=10f64.powi(digits); (x*f).round()/f`; null->NULL. Drop old round at 5920/6395. EXACT op order (1-ULP risk).
- atan2(y,x) (tests*b atan2*\*): parser 2-arg group gql.rs 2450; exec 2-arg numeric arm 5927-5934: `x.atan2(y)` (x=args[0],y=args[1]); null->NULL via `_`.
- log10(x) (language m_numeric_value_functions_8): parser 1-arg group 2433; exec 1-arg group 5920 + scalar_num_fn 6382 `x.log10()`.
- list_sort(list[,order][,nullOrder]) (tests_b list_sort_desc/nulls_first): parser 1..=3 args (2442); rewrite exec 6065 to read descending from args[1] ("desc"), nulls_first from args[2] ("first"/"last"), reuse the ORDER BY comparator (value.rs cmp_total + exec 1489-1531) — MUST equal ORDER BY byte-for-byte.

## 3. SELECT ... FROM MATCH — 11 cases (tests_e 78-88) — 2 phases

Pure parser desugar to MATCH+RETURN. Keywords non-reserved (no lexer change).

- Phase 0: extract gql.rs 465-520 (pattern+comma-join+publish scope+WHERE) into `fn match_body() -> Plan`.
- Phase 1 (cases 78,79,80,81): at gql.rs ~443 `if peek_kw("SELECT") return select_statement()`. Two-pass parse (save/restore self.pos, idiom at ~476): scan for depth-0 `FROM`. No FROM -> parse items over empty scope, apply_items(Plan::Row). FROM -> set pos to FROM, eat FROM, optional graph-ref ident, eat MATCH, match_body(), then rewind to items, return_items() over populated scope, apply_items + DISTINCT/ORDER/paging tail (reuse 583-616).
- Phase 2 (cases 82-88, HAVING/GROUP BY): GROUP BY forces grouping even w/o agg in SELECT; HAVING lowered input-scope, aggs extracted (new `extract_aggs` expr-walker) into group aggs, evaluated post-agg. `aggregating = has_agg_items || group_keys || having`. plan.aggregate(keys,aggs)->filter(rewritten_having)->project(item order)->order_page. HAVING null drops all (3VL filter). Final project in ITEM order (Aggregate emits keys-then-aggs).
- GOTCHA: item-order vs aggregate key-then-agg order coincide for corpus (keys before count(\*)); Phase 2 project fixes latent hazard.

## 4. ALL SHORTEST / SHORTEST 1 — DONE (commit 2b480fe3, 363->350, 13 cases; k>=2 + per-hop WHERE deferred)

<details><summary>original design (implemented)</summary> (worktree agent was blocked by a bad fork base; design below is complete & verified against current code)

Concrete plan (implement in the MAIN tree; agent a5af0ba3 could not because its worktree predated the crate):

- ir.rs: `enum ShortestSelector { Any, All }`; add `min: u32` + `selector: ShortestSelector` to `Plan::ShortestPath` (currently has input/from/dir/edge_label/max); update the `shortest_path(...)` builder signature.
- gql.rs `shortest_path_binding` + `query`: accept `ANY SHORTEST`/`ALL SHORTEST`/`SHORTEST 1[ GROUP|GROUPS]` (→Any; `1 GROUP`→All; k>=2 and k=0 → parse error). Add a BARE-selector entry (no path var) before match_body. Translate inline endpoint/seed `label`+`{props}` into Scan-label + node_prop_filters (seed below the hop, endpoint filters above it; a same-var endpoint like `->{1,3}(a)` → a `Slot(src)=Slot(end)` equality filter). Keep rejecting per-hop edge WHERE. Thread `min` from `*`(0)/`+`(1).
- exec.rs `shortest_path` (+ dispatch ~900): record ALL min-distance predecessors and enumerate the shortest-path DAG (`enumerate_shortest_paths`), emitting one row per distinct shortest path so endpoint multiplicity is right WITH OR WITHOUT lineage (do NOT gate multiplicity on `track`). Endpoints = nodes with `dist >= min`, so `*`(min 0) emits the seed at dist 0 (zero-length-to-self). `Any` keeps only the FIRST predecessor → one row per endpoint (existing 4 unit tests stay green at min=1). Mirror core all_shortest_walk / shortest_ends (crates/lenke-core/src/gql/eval/pathfind.rs). Seed-cycle re-emission is NOT needed for any required case.
- opt.rs: 4 `Plan::ShortestPath { .. }` match arms (rewrite ~202/211, pushdown ~671/681/697) need `min, selector` added to destructure + reconstruction. ALSO cost.rs estimate arm.
- Fixes all 8 all*shortest*_/shortest*1*_ cases + incidental ANY-SHORTEST inline-props cases. DEFER SHORTEST k>=2 + shortest_k_per_hop_pred (parse error).

## 4-OLD. (scout summary, superseded by the design above)

- TRACTABLE: ALL SHORTEST (all_shortest_tied_lengths, all_shortest_endpoint_multiplicity, +others) and SHORTEST 1[/GROUP] (reduces to ANY/ALL). Prereq: shortest_path_binding (gql.rs 624-666) currently REJECTS inline props/labels on endpoints — must translate to Scan-label + node_prop_filters (both ALL SHORTEST corpus cases use `(a:N {id:'a'})`).
- IR: add `selector {Any,All,K{k,group}}` to Plan::ShortestPath (ir.rs 383-389) + builder 650-664 + dispatch exec 900-912.
- Parser: generalize shortest_path_binding to accept ANY/ALL SHORTEST + SHORTEST k [GROUP|GROUPS]; add BARE-selector path before pattern() (gql.rs 465) for `MATCH ALL SHORTEST (...)` w/o path var.
- Exec ALL (exec shortest_path 4812): replace single pred map with `preds: Vec<(prev,edge)>` per vertex appended at equal distance (mirror core pathfind.rs all_shortest_walk 1128-1211); emit one row per shortest path (enumerate DAG when tracked, DP-count when untracked — multiplicity must NOT gate on lineage). Mirror seed-cycle/zero-length (pathfind 970-988,1146-1197).
- DEFER: SHORTEST k>=2 (needs per-trail length-ordered enumeration the engine lacks), shortest_k_per_hop_pred (also per-hop WHERE, deferred). Row order NOT pinned (multiset compare) = big de-risk.

## 5. Small batch

- Line/block comments (language m_line_comment_double_dash) EASY: in lex() before `match c` (gql.rs ~204): `--` and `//` -> skip to \n; `/* */` -> skip to close (err if unterminated). Must precede `-` (232) and `/` (220) arms. `--` is UNCONDITIONALLY a comment (GQL undirected uses `~`, no `--` edge).
- Standalone ORDER BY before RETURN (tests_f standalone_page_1/2/3/4) EASY-MED: in query_tail, after WITH/MATCH chain loop, before SET/REMOVE/RETURN dispatch (gql.rs ~547): if peek ORDER/OFFSET/LIMIT (NOT bare SKIP): parse ORDER BY keys over bindings (new standalone_sort_keys, simpler — no hidden cols), OFFSET/SKIP, LIMIT -> plan.order_page; fall through to RETURN. OFFSET 99 -> 0 rows.
- INSERT..RETURN binding (language m_insert_return_binds_created_node, tests_a insert_node_then_return, insert_multi_label_node) HARDEST: new `Plan::InsertReturn{nodes,edges,tail}` + executor arm (seed 1-row Batch of Col::Nodes[created id] per slot, pull_body(tail)). insert() (gql.rs 915): if peek RETURN, set scope=var_to_idx, slots=nodes.len(), tail=query_tail(Plan::Row). `&` multi-label (insert_multi_label_node) needs `Tok::Amp` lexer + insert_node `&`-label loop (gql.rs 977). pull_body only supports Row/Project/Filter/Expand/VarLength — restrict INSERT tail to projections. First write-then-return path in engine.

## 6. ACYCLIC / SIMPLE path modes — 8 cases — node-uniqueness

Core PathMode default=Trail (matches engine). ACYCLIC=no repeated NODE, SIMPLE=no repeated node except close start==end.

- IR: replace VarLength `trail: bool` with `mode: PathMode{Walk,Trail,Simple,Acyclic}` (or add `node_unique: bool`) — touches ~15 sites (builder, dispatch, streaming rebuild, count/agg drivers, opt.rs field-name rebuilds, gremlin, tests).
- Parser: extend the WALK/TRAIL block (gql.rs, already added) to accept ACYCLIC/SIMPLE (currently errors).
- Exec (varlen_dfs 4733, varlen_count_dfs 3099, varlen_agg_dfs 3268): mirror core pathfind reachable_each_unit — node_unique reuses `used` as NODE stack, mark start once, skip if visited (Simple allows close nbr==from). Gate mode-blind fast paths: algebraic count 3050 (only Trail), count(DISTINCT endpoint) BFS 3193-3204 (only Walk|Trail).
- Note: quantified subpath-group cases (vqs_18, mea_acyclic) ALSO need the `((..)){n}` group feature — stay blocked on that.

## Round 2 — tractable clusters found after the original queue was cleared (baseline 343)

Re-categorized the remaining 342 (26 intentional "core rejects but engine accepts" numeric-model/error-parity + 317 engine!=core). Most of the 317 are the DEFERRED hard features: parenthesized subpath-group `((..)){n}` (~85, the biggest), per-hop WHERE on var-length (~17), FOR..IN/WITH OFFSET (~12), SHORTEST k>=2/bounded (~5). But these NEWER clusters ARE tractable:

- **Simple CASE** — DONE (commit 2393abca, 343->337).
- ~~Simple CASE orig~~ `CASE expr WHEN val THEN … [ELSE …] END` (~6: m_simple_case_over_integers_1-4, case_simple_form, m_simple_case_null_subject_never_matches). The engine has searched CASE (`CASE WHEN cond THEN`); add the simple form = desugar `CASE x WHEN v1 THEN r1 … ELSE e` to searched `CASE WHEN x=v1 THEN r1 …` (NULL subject matches nothing — 3VL equality). Parser-only, in `primary`'s CASE handling.
- **String escapes** — DONE (commit 5c202b0c, 334->325, 9 cases incl. resolving the malformed-reject case).
- ~~String escapes orig~~ (~8: tck*literals6*\_ — `\n` `\t` `\uXXXX` etc., plus malformed-escape rejection which is error-parity/intentional). Lexer: decode escapes inside single/double-quoted string literals (mirror core lexer.rs escape table). Beware: `h\__`/`tck*\*\_malformed*\*` (`\uH`, overflowing) are core-REJECTS = intentional, leave baselined.
- **NULLS FIRST / LAST** — DONE (commit 71b0ba9f, 337->334).
- ~~NULLS orig~~ (~4: m_nulls_first_overrides_ascending_default, m_nulls_last_overrides_descending_default, …). `SortKey.nulls_first` already exists; parse the optional `NULLS FIRST|LAST` after ASC/DESC in `order_keys`/`sort_keys`/`standalone_sort_keys` and set it (default already matches core: nulls last).
- **IS [NOT] LABELED** — DONE (commit c27592d7, 325->320, 5 cases). Broader label ALGEBRA still open (see below).
- ~~IS LABELED orig~~ (~14: m*is_labeled_tests_element*\*, is_labeled, m_is_not_labeled_negates, m_label_conjunction `(n:Person&Software)`, m_label_negation `(n:!Software)`, m_label_wildcard `(n:%)`, m_is_as_label_introducer `(n IS Person)`, m_edge_label_negation `-[:!CREATED]->`, m_colon_label_predicate `x:Person|Software`). `IS LABELED L` / `IS NOT LABELED L` predicate is the tractable core (desugar to the existing label-predicate `x:L` → In(labels(x))). The full label ALGEBRA (&/|/!/%) in node/rel position is bigger; `&` conjunction and `!` negation are the next steps. Node-label disjunction `(n:A|B)` was already noted deferred but composes here.

Still DEFERRED (hard / out of scope): `((..)){n}` subpath groups (~85 — the largest single lever left, but a real feature), per-hop var-length WHERE, FOR..IN list unwind, SHORTEST k>=2, node-label full algebra. Intentional (leave baselined): oversized-int/overflow literals, arith-on-non-numeric, reserved-word-as-identifier, malformed escapes, cross-type-compare errors, avg(duration).

## Round 3 candidates (baseline 320)

- **Node label algebra** — DONE (commit d9f28891, 320->310, 10 cases). Edge negation `-[:!T]->` + landing-node labels still deferred.
- ~~Label algebra orig~~ (~9): `(n:A&B)` conjunction (AND of label filters — seed Scan on first, filter the rest), `(n:!A)` negation, `(n:%)` wildcard (any label), `(n IS Person)` IS-introducer (= `:Person`), edge `-[:!T]->` negation, label disjunction in a WHERE predicate (`x:A|B`). Medium; the `&` conjunction and `IS` introducer are the most tractable.
- **THE BIG LEVER: parenthesized subpath-group quantification `((..)){n,m}`** (~85 cases, tests_d/tests_b). A genuine feature: a parenthesized path segment repeated n..m times, binding the group's variables per repetition. This is the single largest remaining cluster but a substantial implementation (new IR for a repeated sub-pattern + per-rep variable scoping + interaction with path modes). Assess carefully before committing; may warrant its own scouting pass. If not taken, it + the other hard-deferred features (per-hop var-length WHERE, FOR..IN, SHORTEST k>=2) + intentional divergences are the floor — STOP with a final summary.

## Round 4 — DONE so far: landing-node labels (commit 4b9dcf59, 310->289, 21 cases — endpoint labels in COUNT{}/EXISTS{} subqueries + landing label-disjunction). NEW deferred: reverse-correlated subquery where the outer var is the hop's LANDING (`COUNT { (m)-[:R]->(n) }`, `(m)<-[:R]-(n)`, `(m:Target)-[:R]->(n)`) — needs the body to traverse from the bound endpoint backward (~6 cases).

## Round 4 (cont) — tractable tail found at baseline 310 (NOT yet exhausted — keep going, do not stop)

Re-categorized the 309 remaining: 25 intentional "core rejects" + 285 engine!=core. The 285 = ((..)){n} subpath group (~85, the big lever), per-hop var-length WHERE (~17), FOR..IN/WITH OFFSET (~12), SHORTEST k>=2 (~5) — all hard-deferred — PLUS this tractable tail worth investigating (each cluster: dump a case, run engine vs core, find the divergence):

- **COUNT{} / EXISTS{} subquery variants** (~15: count_subquery_degree_3..12, exists_count_semi_join_1..5): endpoint LABELS inside the subquery (`(n)-[:R]->(:Target)`), INCOMING direction (`(m)<-[:R]-(n)`, `(m)-[:R]->(n)`), `WHERE true` bodies, and `count(DISTINCT a)`. Likely the correlated-subquery body doesn't handle an endpoint label / reversed hop / the start not being the outer var. HIGH VALUE if a common root cause.
- **CAST edge cases** (~3: cast_bool `CAST('yes' AS BOOL)`, cast_list `CAST('ab' AS LIST)`, cast_int_null `CAST('nope' AS INT)`): string→bool/list/int coercions; check value::cast vs core.
- **Total order** (~4: order_total_order_asc/desc, min_total_order, max_total_order): a sort/min/max total-order nuance (likely NaN/null placement or mixed-type) — compare vs core's cmp_total.
- **Implicit grouping + ORDER BY** (~2: m_implicit_grouping, group_by_aggregate — `RETURN s.name, count(*) ORDER BY s.name`): investigate why the grouped projection + ORDER BY over a group key diverges (row order? the ORDER BY hidden-column path with an aggregate?).
- **Temporal compares** (~few: temporal_timestamp_ge, temporal_duration_unordered_count_zero): DURATION/DATETIME/TIMESTAMP literal comparisons.
- **count(\*) arithmetic** (count_star_shortcut_plus1 `count(*) + 1`), order_by_letin_over_output_column (LET in ORDER BY).

STOP only once THIS tail is worked and only ((..)){n} + per-hop-WHERE + FOR..IN + SHORTEST-k>=2 + edge-label-negation + intentional divergences remain.


## Round 4 diagnosis notes (baseline 285)
- **CAST string->bool/list/int = INTENTIONAL divergence, LEAVE BASELINED.** Core desugars `CAST(x AS BOOL)`->`to_boolean(x)` / `AS LIST`->`to_list(x)` (lenient, NULL on failure); the engine's CAST THROWS E_INVALID_VALUE by deliberate design (documented in ir.rs: "a failed conversion throws ... unlike CAST FUNCTIONS which return NULL"). Changing it is a value-contract change — do not.
- **Total order = FIXED** (commit 3b16d5c6, 289->285): `rank()` now Num<Str<Bool<Temporal (was Bool<Num<Str) matching core type_rank. Was a latent bug the fuzzer never hit.
- **NEXT tractable: aggregate in a RETURN expression** (~2+: count_star_shortcut_plus1 `RETURN count(*) + 1`, m_implicit_grouping / group_by_aggregate `RETURN s.name, count(*) ORDER BY s.name`). The RETURN-level analog of the HAVING `extract_aggs` already built: an aggregate nested inside a projection expression (`count(*)+1`) or an ORDER BY over a group-key expression needs the agg hoisted into the Aggregate and the surrounding expr rewritten to a slot. Reuse the HAVING machinery (hoist_having_agg / rewrite_group_keys) generalized to the RETURN item list. MEDIUM.
- Reverse-correlated subquery (~6, `COUNT { (m)-[:R]->(n) }` outer var is landing) and temporal literal compares (temporal_timestamp_ge etc.) — still open, medium.
