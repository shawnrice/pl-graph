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
- **Grouped ORDER BY over a group-key expression** — DONE (commit ba1a631d, 285->283, 2 cases: m_implicit_grouping, group_by_aggregate). order_keys now matches an ORDER BY expr against the group keys.
- **STILL OPEN: aggregate NESTED in a projection expression** (~2+: count_star_shortcut_plus1 `RETURN count(*) + 1`, m_implicit_grouping / group_by_aggregate `RETURN s.name, count(*) ORDER BY s.name`). The RETURN-level analog of the HAVING `extract_aggs` already built: an aggregate nested inside a projection expression (`count(*)+1`) or an ORDER BY over a group-key expression needs the agg hoisted into the Aggregate and the surrounding expr rewritten to a slot. Reuse the HAVING machinery (hoist_having_agg / rewrite_group_keys) generalized to the RETURN item list. MEDIUM.
- Reverse-correlated subquery (~6, `COUNT { (m)-[:R]->(n) }` outer var is landing) and temporal literal compares (temporal_timestamp_ge etc.) — still open, medium.

## Round 4 (cont) — diagnoses at baseline 282

- **TIMESTAMP literal = DONE** (commit 18fbf82a, 283->282): `TIMESTAMP '…'` is core's DATETIME alias; added to temporal_tag.
- **reverse-correlated subquery (~6) — DIAGNOSED, DEFER (slot-management risk).** correlated_subquery_body requires the FIRST subquery node to be the bound outer var; the failing cases have the outer var as the LANDING (`COUNT { (m)-[:R]->(n) }`). Forward body = `Expand{input:Row, from:<outer slot>}` and only counts matches. The catch: extend_chain binds the landing at scope-slot `outer_width+1` while the runtime column lands at `outer_width` (a gap that's harmless ONLY because forward cases never filter the landing). A reverse case with a local-node LABEL (`(m:Target)-[:R]->(n)`) needs a filter at the correct landing slot — get the offset wrong and it silently miscounts. Needs careful slot reconciliation (verify against the EXISTS/CountSubquery eval's row-width contract) before implementing. MEDIUM.
- **aggregate nested in a projection expr `count(*)+1` (~1) — remaining.** return_items parses a leading aggregate as a bare RetItem::Agg then chokes on the trailing `+ 1`. Needs generalizing apply_items: parse each RETURN item as an expression with aggregate hoisting (reuse hoist_having_agg), keys = items with no aggregate, project each item's (rewritten) expr over the post-agg schema. Touches apply_items (shared by RETURN + WITH) — verify WITH still works. MEDIUM.
- temporal_duration_unordered_count_zero — likely already-passing-or-intentional (fixture has `d` not `dur`, so `n.dur` is missing -> NULL > DURATION -> UNKNOWN -> count 0); re-check.

## CAMPAIGN COMPLETE — baseline 694 -> 281 (this run: from 427). STOPPED.

The tractable tail is worked. Remaining 281 = 25 intentional "core rejects" (numeric-model + error-parity)

- 257 hard-deferred, dominated by:

* parenthesized subpath-group `((..)){n,m}` (~85+) — a genuine feature: repeated sub-pattern IR + per-rep var scoping. The single largest remaining lever; warrants its own scouting pass.
* per-hop var-length WHERE `-[e WHERE ..]->{..}` (~17), FOR..IN / WITH OFFSET|ORDINALITY list unwind (~12), SHORTEST k>=2 / bounded shortest (~5).
* reverse-correlated subquery (~6) — DIAGNOSED, deferred: needs verified landing-slot reconciliation in the EXISTS/CountSubquery body (silent-miscount risk).
* aggregate nested in a projection expr `count(*)+1` (1) — deferred: generalizing the shared RETURN/WITH apply_items pipeline for one case is not worth the regression risk.
* edge-label negation `-[:!T]->` + node-label disjunction in a WHERE predicate — the remaining label-algebra corners.
  INTENTIONAL (leave forever): CAST throws (engine design; core is lenient), f64 numeric model (oversized ints / arith-on-non-numeric -> NULL), cross-type-compare OPERATOR throw, reserved-word-as-identifier, malformed literals.

## ROUND 5 — FINISH EVERYTHING (loop restarted at baseline 281; drive known_gaps toward 0)

User directive: fix all remaining deferred items (they were deferred for effort, not principle). Priority order (tractable -> hard); DIAGNOSE each before coding (inline engine-dialect ndjson fixture, engine-vs-core):

1. count(\*)+1 (aggregate nested in a projection/ORDER-BY expr, 1+) — reuse hoist_having_agg; generalize return_items/apply_items; keep RETURN + WITH byte-identical.
2. SAFE error-parity "core rejects" (make the engine ALSO reject, matching core — these do NOT change the f64 value contract): reserved-word-as-identifier (m*reserved_word*_), CALL config validation (call*config*_), aggregate-type faults (avg*duration/sum_date/sum_mix*_/faulting*aggregate — sum/avg over temporal/mixed throws), date_part*_ (unknown fn / rejects string|number), range_bounded_2/3.
3. Edge-label negation `-[:!T]->` + node-label disjunction/expr inside a WHERE predicate (`x:A|B`).
4. SHORTEST k>=2 / bounded shortest (~5) — per-trail length-ordered enumeration.
5. reverse-correlated subquery (~6) — FIRST verify the EXISTS/CountSubquery landing-slot contract (forward binds landing at scope-slot outer_width+1 vs runtime column outer_width); then start from the bound endpoint with reversed dir + landing filter at the correct slot.
6. FOR..IN / WITH OFFSET|ORDINALITY list unwind (~12).
7. per-hop var-length WHERE `-[e:R WHERE ..]->{..}` / `(()-[..]->()){..}` (~17).
8. THE BIG ONE: parenthesized subpath-group quantification `((..)){n,m}` (~85+) — do a READ-ONLY scout first (design the repeated-sub-pattern IR + per-rep variable scoping + interaction with path modes), then implement.
9. NUMERIC-MODEL value-contract cases (oversized/overflow int literals, arith-on-non-numeric str/bool+num -> NULL vs core throw): these were the user's established f64/postgres-style choice. Attempt ONLY if it holds byte-identity for existing cases; if fixing requires breaking the f64 model or the fuzzer, SURFACE to the user rather than silently changing the value contract.
   Verify EVERY iteration: cargo test --release --lib; differential_fuzz seeds 1 & 42 (byte-identity); clippy 0; re-baseline CORPUS_BASELINE=1 (0 NEW); commit; update this file.

### Round-5 progress + reclassification (baseline 280)

- count(\*)+1 aggregate-in-projection-expr — DONE (commit c538ee7d, 281->280).
- RECLASSIFIED: **aggregate-type faults (avg/sum over temporal/mixed/list) are VALUE-CONTRACT, not safe error-parity.** fold_grouped deliberately POISONS a non-numeric aggregate group to NULL (explicit code comment), the SAME postgres-style non-numeric->NULL model as arith `'abc'+1`->NULL. Core THROWS. Changing sum/avg to throw = changing the value contract the user chose. SURFACE, do not silently flip. Same bucket: oversized/overflow int literals, str/bool + num arith. => user decision.
- Genuinely SAFE error-parity (fix, no value-contract change): reserved-word-as-identifier (parser rejects a reserved keyword used as a bare name), CALL config validation (call*config*_). date*part*_ needs checking (may be non-date->throw = value-contract, or a missing-fn = safe).
- Next FEATURE work (unambiguous, the bulk): edge-label negation -[:!T]->, node-label expr in WHERE, SHORTEST k>=2, reverse-correlated subquery, FOR..IN, per-hop var-length WHERE, and THE BIG ONE ((..)){n,m} (~85, scout first).

### Round-5 progress (baseline 265)

- label expression in a WHERE predicate x:A|B (commit e8c8f56f) — 280->279.
- reverse-correlated COUNT/EXISTS subquery (commit 9f153740) — 279->273 (6). Forward landing-slot contract validated it.
- Subpath-group `((..)){n,m}` INCREMENT 1 (single-edge, endpoint-only -> var_length desugar) (commit bd883674) — 273->265 (8). Parser hook: is_subpath_group_start / parse_subpath_group in gql.rs; unanchored `((` seeds Scan{None} in pattern().

### Subpath-group scout plan (real gap = 48 cases, staged):

- INC 1 DONE (single-edge endpoint-only, 8 cases).
- INC 2 (multi-hop body, endpoint-only ~5: mea_trail, qsp_multi_ends_1/2; mea_acyclic/simple DEFER — path mode on multi-hop): bounded-unroll `{n,m}` into r\*k Expands UNION'd; RISK: unroll is a WALK, not Trail — only safe where no edge repeats in reach or `{n,n}` exact. Check each fixture.
- INC 3 (group-var-as-list in RETURN, ~21: qsp*group_vars*\_, gv\_\_, vqs_7/9/11/13/14/15/19/21/22/23, unanchored_path_len/nodes_size): NEEDS (a) new `Plan::RepeatGroup{body,min,max,mode,group_slots,...}` in ir.rs + a DFS repeater in exec that materializes one Value::List per group slot (columnar analogue of core bind_group_vars_flat pathfind.rs:163), AND (b) `Expr::Index{base,idx}` — subscript `x[i]` — parse postfix `[` in field_chain (gql.rs), eval next to Expr::Field. Expr::Index is a HARD PREREQ (y[size(y)-1], x[0].id). LARGE.
- INC 4 DEFER: nested `((..){a,b}){n,m}` list-of-lists (bind_unit recursion), per-rep WHERE with per-rep vars (e.amt<=x.bal — the per-hop-WHERE feature).
  Row order NOT a hazard (multiset compare; ordered:true cases carry ORDER BY).

### ROUND 6 (loop cont.) — baseline 265 -> 202

- **Expr::Index subscript `x[i]` — DONE** (commit c69d0b1c, 265->257, 8 cases). ISO 0-based
  subscript over list literals, records, maps, AND `nodes(p)[i]`/`edges(p)[i]` path lists.
  The path case emits a typed Col::Nodes/Col::Edges (u32::MAX sentinel for out-of-range) so a
  following `.prop` resolves the node/edge property (`edges(p)[0].w`). Parser routes
  list/record/map/call primaries through field_chain so a subscript can follow. This was
  the INC3 part (a) prereq.
- **Multi-label edges — DONE** (commit 32982b57, 257->202, 55 cases). Was the single biggest
  cluster (all tests_f walked_frame/streamed_count/every_count). An edge's type is its FIRST
  label; the rest are secondary. Store gains sparse `edge_extra: HashMap<eid,Vec<u32>>` +
  `edge_has_label`/`has_multi_label_edges`, mirroring core's e_type/e_extra. One shared
  `edge_carries_wanted` predicate feeds for_each_nbr, the var-length DFS/count/fold, the
  shortest BFS, and the degree-product/3-hop count shortcuts. Type-index fast path skipped on
  a multi-label graph. ndjson reads edge `"labels":[…]` (first=type, rest=extras); harness
  passes the whole array through instead of first(). type(edge) still = first label.

### Remaining 202 — reclassified next targets (tractable -> hard)

- **`bare_path_*` / walked_frame_33 (~4): named path over a NON-shortest var-length**
  (`MATCH p = (a)-[:R]->{1,3}(x) RETURN path_length(p)`). Engine rejects with "a named path
  requires a shortest-path selector"; core accepts and binds the (walk) path. NEXT, tractable:
  lift that parser restriction for a var-length body and materialize the lineage path. Verify
  path mode = WALK matches core.
- **per-hop WHERE in a subpath group `((x)-[e:R]->(y) WHERE pred){n,m}` (~10: qsp*per_hop*_,
  vqs*8, subpath_where*_)** — where the WHERE only FILTERS and the RETURN needs no group-var
  list (just t.id). Per-rep predicate over the rep's own x/e/y. MEDIUM.
- **THE BIG ONE: group-variable-as-list `Plan::RepeatGroup` (~21: qsp*group_vars*\_, gv\_\_,
  vqs_7/9/11/…)** — each group var (x,e,y) binds to a Value::List across reps; `size(e)`,
  `x[0].id`, `y[size(y)-1].id`, `WITH e AS hops`. Needs the new operator + DFS repeater
  materializing one list per group slot (columnar analogue of core bind_group_vars_flat).
  Expr::Index (done) was the prereq. LARGE — its own iteration.
- Smaller: zero*limit* (4), order*alias* (4), distinct*nan* (3), group_by_bound (3),
  value_subquery_aggregate (3), num_string_overflow (3, likely value-contract), exists_multi_match (4).
- VALUE-CONTRACT (surface, don't flip): num_string_overflow, sum/avg-over-temporal faults,
  oversized-int, CAST-throws. Left baselined by design.

### ROUND 6 (cont.) — 202 -> 190

- **Named path over a non-shortest var-length — DONE** (commit 0095a5cc, 202->194, 8 cases).
  Lifted the "named path requires a shortest selector" parser restriction; var_length now
  builds a Lineage sidecar (node/edge chain stack + push_path, shared with shortest_path) so
  path_length(p)/edges(p)/nodes(p) resolve over a plain var-length body. New PathBufs struct.
  bare_path_binds_simple_cycle stays baselined (needs repeated-pattern-variable `(a)…(a)`
  equality join — a separate feature).
- **LIMIT 0 short-circuits before projection — DONE** (commit 10d32af0, 194->190, 4 cases).
  OrderPage with limit==0 returns empty without pulling its input, so a faulting projection
  (`RETURN 1/0 AS x LIMIT 0`) yields empty (matches core) instead of erroring.

### NEXT (baseline 190), tractable -> hard:

- **repeated-pattern-variable equality join `(a)…(a)` / `(x)-[]->(x)`** (bare_path_binds_simple_cycle
  - likely others): a variable used twice in a pattern pins the two slots equal (post-filter
    new_slot == existing_slot). Check how node() binds a name already in scope. SMALL-MED.
- **`->{0,0}` zero-bound quantifier** (zero_bound_3): min=max=0 var-length = stay at source,
  count = source count. Likely a var_length min==max==0 edge case. SMALL.
- **per-hop WHERE in a subpath group `((x)-[e:R]->(y) WHERE pred){n,m}`** (~10: qsp*per_hop*_,
  vqs*8, subpath_where*_) where the WHERE only filters. MEDIUM.
- **THE BIG ONE: group-variable-as-list `Plan::RepeatGroup`** (~21: qsp*group_vars*\_, gv\_\_,
  vqs_7/9/11/…). Its own iteration. Expr::Index (done) + var_length lineage (done) are prereqs.
- Smaller: order_alias (4, NULLS FIRST / DISTINCT+ORDER-BY-underlying-expr), distinct_nan (3),
  group_by_bound (3, LET/WITH bound name + GROUP BY), exists_multi_match (4, EXISTS{ MATCH MATCH }).
- VALUE-CONTRACT (surface, don't flip): num_string_overflow, sum/avg-over-temporal, oversized-int, CAST-throws.

### ROUND 6 (cont.) — 190 -> 169

- **GROUP BY after RETURN — DONE** (commit 4c5c5186, 190->183, 7 cases). project_and_page now
  consumes an explicit GROUP BY (was SELECT-only); non-agg items are the implicit keys.
- **ORDER BY alias before NULLS + by a projected expr — DONE** (commit 22fe08fb, 183->181, 2).
  Added NULLS to the bare-alias terminator set; an ORDER BY expr equal to a projected item's
  expr sorts by that output column (composes with DISTINCT).
- **LET binding clause — DONE** (commit 53c20335, 181->178, 3). ISO additive-binding clause
  `LET name = expr [,…]` (distinct from the LET…IN…END expression): projects existing bindings
  forward + adds new ones.
- **Unquantified subpath group `(( pattern WHERE ))` — DONE** (commit 2539be58, 178->169, 9).
  Balanced-paren lookahead splits quantified (var_length) from unquantified (scoping paren →
  inline inner pattern + trailing WHERE). Named path over an unquantified group rejected (core does).

### NEXT (baseline 169), tractable -> hard:

- **per-hop WHERE in a QUANTIFIED subpath group `((x)-[e:R]->(y) WHERE pred){n,m}`** (qsp*per_hop*\*,
  vqs_8, nested_per_rep) — the WHERE filters each repetition. Now that unquantified groups + the
  balanced-paren split exist, this extends the quantified path with a per-rep predicate in var_length. MEDIUM.
- **THE BIG LEVER: group-variable-as-list `Plan::RepeatGroup`** (~21: qsp*group_vars*\_, gv\_\_,
  vqs_7/9/11/…) — each group var (x,e,y) binds to a Value::List across reps. Its own iteration.
  Expr::Index + var_length lineage (both done) are prereqs.
- **uncorrelated multi-pattern EXISTS `EXISTS { MATCH (x:N) MATCH (y:M) }`** (exists_multi_match, 4) —
  needs pull_body to support a Scan/cross-join body (currently Row/Expand/VarLength/Filter/Project
  only), or a constant-EXISTS eval (body independent of the outer row → run once, broadcast). MEDIUM.
- Smaller: shortest\_ / shortest_per_hop (SHORTEST k>=2), value_subquery_aggregate (3), multiseg_u (6),
  distinct_nan (string->NaN, murky/value-contract), num_string_overflow (value-contract, leave).
- group*by_bound_1 done; order_by_letin_over_output_column = LET-IN-END \_expression* in ORDER BY (separate).

### ROUND 6 (cont.) — 169 -> 165 (session total 265 -> 165, 100 cases)

- **PROPERTY_EXISTS on edges + null element — DONE** (commit 83104a79, 169->167). Handles a node
  OR edge slot; a non-element (u32::MAX sentinel or computed value) -> NULL (matches core).
- **FILTER clause + repeated statement-level ORDER BY/LIMIT — DONE** (commit 01b8a9fb, 167->165).
  ISO FILTER statement in the query-tail loop; the standalone order/page clause now loops (page then re-page).

### NEXT (baseline 165) — remaining is dominated by a few LARGE features:

- **THE BIG LEVER: group-variable-as-list `RepeatGroup`** (~21: vqs* 12, qsp_group_vars 4, gv*\*).
  DESIGN CONFIRMED: core binds each group var to a LIST across reps (pathfind.rs bind_group_vars_flat:
  for a k=1 single-hop group, source var x = [verts[rep] for rep], edge e = [edges[rep]], target y =
  [verts[rep+1]]). The engine's var_length lineage (node_stack/edge_stack per emitted path, already
  built) captures exactly this — so materialize x/e/y list columns AT EMIT from the stacks. Cleanest as
  an extension to var_length (append list columns for the named group vars after the endpoint) OR a
  follow-on operator reading the group segment. Scope to SINGLE-HOP groups first; multi-hop (gv_bind_each_rep_2hop)
  and nested (nested_paren_varying, list-of-lists) defer. Its own iteration.
- **per-hop edge WHERE `-[e:R WHERE pred]->` in var-length AND shortest** (qsp_per_hop 3, per_hop_inline 2,
  shortest_per_hop 3, nested_per_hop 2) — thread a captured predicate into the varlen_dfs / shortest BFS
  adjacency step, evaluating with the hop's edge (and endpoints) bound. Recurring; unblocks ~10.
- **SHORTEST k (k>=2)** (shortest_2_keeps_two, \_group_all, \_groups_synonym, shortest_k_clamps = 4;
  shortest_k_per_hop_pred also needs per-hop WHERE). ALGORITHM CONFIRMED (core shortest_k_walk): enumerate
  ALL trails per endpoint, sort by (length, discovery), keep first k (plain) or all paths in the k smallest
  distinct lengths (GROUP/GROUPS). Needs trail enumeration (var_length DFS) + per-endpoint selection + lineage.
- **VALUE scalar subquery / uncorrelated multi-pattern EXISTS** (value*subquery*_ 5, exists_multi_match 4) —
  VALUE { … RETURN count(_) } maps to CountSubquery for the correlated count case; general VALUE + uncorrelated
  EXISTS need pull_body to run a Scan/cross-join body or a constant-subquery eval.
- **multiseg_u (6)** — dual-anchor correlated multi-segment EXISTS (ReBAC). HARD.
- VALUE-CONTRACT (leave/surface): range_bounded_2/3 (core caps range size — check if safe error-parity or
  value-contract), num_string_overflow, distinct_nan (string->NaN), sum/avg-over-temporal, CAST-throws.

### ROUND 6 (cont.) — 165 -> 150 (session total 265 -> 150, 115 cases)

- **group-variable-as-list `Plan::RepeatGroup` — DONE** (commit ee5d2e06, 165->150, 15 cases).
  New IR operator lowered when a quantified single-hop group names inner vars; appends one list
  column per group var, materialized AT EMIT from the DFS node/edge stack (reuses the var_length
  lineage machinery). size(x)/size(e) = hop count; typed subscript `x[i].prop` via new
  Expr::Index.elem (ElemKind Node/Edge) + parser group-slot tracking. Optional/anon endpoint;
  bare variable routes through field_chain so `x[0]` parses at expr top level. SINGLE-HOP only.

### NEXT (baseline 150), remaining group-var + other levers:

- **per-rep WHERE in a group `((x)-[e:R]->(y) WHERE pred){n,m}`** (qsp*group_vars_where_scalar, vqs_8,
  qsp_per_hop*\*, nested_per_rep) — per-repetition predicate over x/e/y. Extend RepeatGroup/var_length
  DFS with a captured predicate evaluated per hop (bind x=current,e=edge,y=next). Overlaps per-hop edge WHERE.
- **multi-hop group unit `((x)-[e1]->(m)-[e2]->(y)){n}`** (gv_bind_each_rep_2hop) — k>1: the flat stride
  becomes verts[rep*k+p]; generalize push_group_cols with the unit's hop count k + per-position slots.
- **WITH-carry of a group list** (gv_carry_through_with: `WITH e AS hops … hops[1].amt`) — the group list
  survives a WITH projection; needs the elem-kind (node/edge) to carry through the WITH rebind.
- **nested groups** (vqs_16, nested_paren_varying, nested_outer_gv) — list-of-lists; DEFER (bind_unit recursion).
- **SHORTEST k>=2** (4: shortest*2*\*, shortest_k_clamps) — core shortest_k_walk (enumerate trails, sort by
  length, keep first k / k length-groups). **per-hop edge WHERE** (~8) — thread predicate into varlen/shortest.
- **VALUE scalar subquery / uncorrelated multi-pattern EXISTS** (9). **multiseg_u** (6, dual-anchor, HARD).
- VALUE-CONTRACT (leave/surface): range_bounded, num_string_overflow, distinct_nan, sum/avg-over-temporal, CAST.

### ROUND 6 (cont.) — 150 -> 146 (session total 265 -> 146, 119 cases)

- **SHORTEST k (k>=2) selector — DONE** (commit 49e17341, 150->146, 4 cases). New
  ShortestSelector::ShortestK{k,group}; shortest_k_path enumerates all TRAILS per source (collect_trails,
  edge-dedup bounds depth), groups by endpoint, sorts by (length,discovery), keeps first k or k distinct
  length-groups (mirrors core shortest_k_walk). Endpoint filter stays a Filter above.

### NEXT (baseline 146) — the remaining are the hard/nested tail:

- **per-rep WHERE in a group `((x)-[e]->(y) WHERE pred){n,m}`** (qsp*group_vars_where_scalar,
  qsp_per_hop*\*, vqs_8, nested_per_rep) — per-repetition predicate over the rep's SCALAR x/e/y; needs
  per-rep scalar slots + eval a captured predicate per hop in the RepeatGroup/var_length DFS (build a
  1-row batch with x=Col::Nodes([v]), e=Col::Edges([eid]), y=Col::Nodes([nbr]) and eval). MEDIUM-HARD.
- **inline edge props on a plain var-length `-[:R {amt:20.0}]->{n,m}`** (per_hop_inline_from_a/b) — each
  edge must match; needs an edge-prop filter threaded into var_length (VarLength has no such field — add
  one or a per-hop predicate). Overlaps per-rep WHERE (edge-only case). 2 cases.
- **multi-hop group unit k>1** (gv_bind_each_rep_2hop), **nested groups** (vqs_16, nested_paren_varying,
  nested_outer_gv, nested_quant_ends), **WITH-carry of a group list** (gv_carry_through_with) — DEFER-hard.
- **VALUE scalar subquery / uncorrelated multi-pattern EXISTS** (value*subquery*_ 5, exists_multi_match 4)
  — VALUE {…RETURN count(_)} ~ CountSubquery for the correlated case; general VALUE + uncorrelated EXISTS
  need pull_body to run a Scan/cross-join body or a constant-subquery eval. MEDIUM.
- **multiseg_u (6)** — dual-anchor correlated multi-segment EXISTS (ReBAC). HARD.
- VALUE-CONTRACT (leave/surface): range_bounded_2/3, num_string_overflow, distinct_nan (string->NaN),
  sum/avg-over-temporal, CAST-throws, m_reserved_word (reserved-word-as-identifier = intentional).

### ROUND 6 (cont.) — 146 -> 140 (session total 265 -> 140, 125 cases)

- **per-repetition WHERE in a subpath group — DONE** (commit fe539a72, 146->140, 6 cases).
  RepeatGroup gains per_rep_pred (Option<Box<Expr>>); parser parses the group WHERE against a fixed
  SCALAR mini-scope (source=0, edge=1, target=2), var_length DFS evals it per hop over a 1-row batch
  [Nodes([v]), Edges([eid]), Nodes([nbr])], pruning failing hops. Composes with group lists + optional endpoint.

### NEXT (baseline 140):

- **inline edge props on a PLAIN var-length `-[:R {amt:20.0}]->{n,m}`** (per_hop_inline_from_a/b, 2) —
  NOW EASY: desugar to a RepeatGroup with per_rep_pred = (e.amt == 20.0) at scalar slot 1, empty group_binds.
  In extend_chain's quant+bind path, when rel has ONLY inline props (no var, no where), build RepeatGroup.
- **VALUE scalar subquery / uncorrelated multi-pattern EXISTS** (9: value*subquery*_, exists_multi_match) —
  VALUE{…RETURN count(_)} ~ CountSubquery correlated; general VALUE + uncorrelated EXISTS need pull_body
  to run a Scan/cross-join body or a constant-subquery eval. MEDIUM.
- **multi-hop group unit k>1** (gv_bind_each_rep_2hop), **nested groups** (vqs_16, nested_paren_varying,
  nested_outer_gv, nested_quant_ends, nested_per_hop, nested_per_rep), **WITH-carry** (gv_carry_through_with) — hard.
- **multiseg_u (6)** dual-anchor correlated multi-segment EXISTS (ReBAC). HARD.
- VALUE-CONTRACT (leave/surface): range_bounded_2/3, num_string_overflow, distinct_nan, sum/avg-over-temporal,
  CAST-throws, m_reserved_word (reserved-word-as-identifier = intentional).

### ROUND 6 (cont.) — 140 -> 138 (session total 265 -> 138, 127 cases)

- **inline edge props on a var-length hop — DONE** (commit f227f2a5, 140->138). `-[:R {k:v}]->{n,m}`
  desugars to a RepeatGroup with per_rep_pred = AND(edge.k==v) (edge at scalar slot 1), no group binds.

### REMAINING 138 landscape (baseline 138):

- **~25 value-contract "core rejects but engine accepts"** — LEAVE baselined (f64 model, CAST-throws,
  range_bounded core-caps, num_string_overflow, distinct_nan string->NaN, m_reserved_word, is_typed_closed,
  inline_constraint, h_rejects_literal, graph_pred_all). These are the intentional divergences; do NOT flip.
- **subquery cluster (~13): VALUE scalar subquery + uncorrelated/multi-pattern EXISTS** (value*subquery*_,
  exists_multi_match, exists_bound_a). VALUE{…RETURN count(_)} correlated ~ CountSubquery (2 easy). General
  VALUE (scalar b.name) + uncorrelated EXISTS { MATCH .. MATCH .. } need pull_body to run a Scan/cross-join
  body or a constant-subquery eval (pull_body currently: Row/Expand/VarLength/Filter/Project only). MEDIUM.
- **shortest_per_hop (3): per-hop edge WHERE in a shortest path** `-[e:R WHERE e.w>5]->*` — thread an edge
  predicate into shortest_path BFS + shortest_k collect_trails (reuse the per-rep eval-per-edge mechanism). MEDIUM.
- **nested groups (~13): nested_per_rep, nested_quant_ends, nested_per_hop, nested_paren_varying,
  nested_outer_gv** — list-of-lists group vars, a group WRAPPING a var-length/group (bind_unit recursion). HARD.
- **multi-hop group unit k>1 (qsp_multi_ends/cross ~4, gv_bind_each_rep_2hop)** — generalize push_group_cols
  with the unit hop count k (verts[rep*k+p]). MEDIUM-HARD.
- **multiseg_u (6)** dual-anchor correlated multi-segment EXISTS (ReBAC). HARD.
- WITH-carry of a group list (gv_carry_through_with) — elem-kind must survive the WITH rebind.

Infra now in place for reuse: RepeatGroup (group_binds + per_rep_pred), Expr::Index.elem (typed subscripts),
ShortestK + collect_trails (trail enumeration), var_length lineage (node/edge stacks), multi-label edges.

### ROUND 6 (cont.) — 138 -> 134 (session total 265 -> 134, 131 cases)

- **per-hop edge WHERE in a shortest path — DONE** (commit 23dc817e, 138->134, 4 cases). ShortestPath
  gains edge_pred (Option<Box<Expr>>); parser parses the edge WHERE against a scalar mini-scope (edge at
  slot 0); BFS + collect_trails skip an edge failing it via edge_pred_ok (eval over [Col::Edges([eid])]).
  Also cleared shortest_k_per_hop_pred (SHORTEST k + per-hop combined).

### NEXT (baseline 134) — still-tractable first, then hard:

- **VALUE{…RETURN count(\*)} correlated ~ CountSubquery** (value_subquery_aggregate_dave_deg/carol_zero, 2) —
  parse `VALUE { MATCH <corr-body> RETURN count(*) }`, detect the count(\*) RETURN, emit CountSubquery
  (reuse correlated_subquery_body). EASY-MEDIUM. The general VALUE (scalar b.name) + uncorrelated forms are harder.
- **multi-hop group unit k>1 `((x)-[e1]->(m)-[e2]->(y)){n}`** (gv_bind_each_rep_2hop, qsp_multi_ends/cross ~4)
  — the group body has k>1 hops; generalize push_group_cols to stride verts[rep*k+p]/edges[rep*k+p] and the
  parser to bind per-position vars. Also needs the var_length DFS to only emit at multiples of k reps. MEDIUM-HARD.
- **~25 value-contract "core rejects"** — LEAVE baselined (see prior entry).
- **nested groups (~13)**, **multiseg_u (6)**, **general VALUE / uncorrelated EXISTS (~11)**, **WITH-carry (1)** — HARD.

Infra in place: RepeatGroup (group_binds + per_rep_pred), Expr::Index.elem, ShortestK + collect_trails +
edge_pred, var_length lineage, multi-label edges. When only value-contract + genuinely-hard remain, STOP with a summary.

### ROUND 6 (cont.) — 134 -> 132 (session total 265 -> 132, 133 cases)

- **VALUE { … RETURN count(\*) } correlated subquery — DONE** (commit 014d0fd0, 134->132, 2 cases).
  correlated_subquery_body gained a count_return flag; VALUE consumes the trailing RETURN count(\*) and
  lowers to CountSubquery (same as COUNT { … }). General VALUE (scalar RETURN / uncorrelated / constant) deferred.

### NEXT (baseline 132) — remaining tractable-ish, then hard:

- **multi-hop group unit k>1 `((x)-[e1]->(m)-[e2]->(y)){n}`** (gv_bind_each_rep_2hop, qsp_multi_ends/cross ~4) —
  parser must accept a multi-hop group body (currently rejects), bind per-position vars, and the executor
  must materialize with stride k (verts[rep*k+p]) and only emit endpoints at rep boundaries (len % k == 0).
  RepeatGroup is single-hop; this needs a k field + DFS emit-gating. MEDIUM-HARD.
- **general VALUE scalar subquery / uncorrelated multi-pattern EXISTS** (value_subquery_correlated_scalar,
  \_global, \_constant, \_where_narrows, exists_multi_match ~4, exists_bound_a) — need a scalar-subquery
  evaluator returning the body's single value, AND pull_body to run a Scan/cross-join / multi-MATCH body
  (multi-MATCH is broken even at top level: "bound variable y cannot be re-labeled in a continuing MATCH"). HARD.
- **~25 value-contract "core rejects"** — LEAVE baselined.
- **nested groups (~13)**, **multiseg_u (6, dual-anchor ReBAC)**, **WITH-carry of a group list (1)** — HARD.

STOP CRITERION getting close: after the multi-hop group unit, the non-value-contract remainder is
nested-groups + general-subqueries + multiseg + WITH-carry, all genuinely hard. Reassess then.

### ROUND 6 — FINAL (session 265 -> 125, 140 cases cleared; LOOP STOPPED)

- **multi-hop subpath group unit k>1 — DONE** (commit 82d71122, 132->125, 7 cases). GroupPos became
  NodeAt(p)/EdgeAt(p); RepeatGroup carries k; min/max lowered to hops (reps\*k); DFS emits only at a rep
  boundary (len.is_multiple_of(k)); push_group_cols strides by k. Cleared gv_bind_each_rep_2hop,
  qsp_multi_ends_1/2, qsp_multi_gv, mea_trail/simple/acyclic.

## LOOP STOPPED at baseline 125 — remaining are value-contract + genuinely-hard.

Session cleared 140 corpus gaps (694 -> ... -> 265 earlier campaign, then 265 -> 125 this run) across 19
features. The remaining 125 are NOT tractable without either changing the value contract (forbidden) or a
large multi-file mechanism for 1-6 cases each:

REMAINING, categorized:

- ~25 VALUE-CONTRACT "core rejects" — INTENTIONAL, leave forever: f64 model (num_string_overflow,
  oversized ints), CAST-throws, range_bounded core-caps, distinct_nan string->NaN, m_reserved_word
  (reserved word as identifier), is_typed_closed RECORD schema, inline_constraint, h_rejects_literal,
  graph_pred_all, date_part_rejects, sum/avg-over-temporal.
- NESTED groups (~15): nested_per_rep, nested_quant_ends, nested_per_hop, nested_paren_varying,
  nested_outer_gv, vqs_16 — a group/var-length WRAPPING a group (list-of-lists group vars, bind_unit
  recursion). Needs a recursive RepeatGroup + nested Value::List materialization. HARD.
- SUBQUERIES (~11): exists_multi_match, exists_bound_a (uncorrelated / multi-MATCH EXISTS — pull_body
  supports only Row/Expand/VarLength/Filter/Project, no Scan/cross-join; multi-MATCH is broken even at
  top level), value_subquery_correlated_scalar/\_where/\_constant (general scalar subquery returning a
  value, not a count). Needs a scalar-subquery evaluator + cross-join pull_body. HARD.
- MULTISEG (6): multiseg*u — dual-anchor correlated multi-segment EXISTS (ReBAC: (u)-[:MEMBER]->*(s)-[…]->
  (gr)-[:PARENT]->\_(t) correlated on BOTH u and t). Needs a two-sided correlated reachability body. HARD.
- qsp_multi_cross (2): per-rep WHERE on a MULTI-hop unit (references e1 AND e2 of the rep) — needs a
  per-rep-BOUNDARY eval with a multi-slot mini-scope, distinct from the per-hop mechanism. MEDIUM.
- vqs_22 (1): a group followed by OPTIONAL MATCH. WITH-carry of a group list (1). MEDIUM.

Infra shipped this session (reusable for the above): multi-label edges; Expr::Index subscript +
ElemKind typed elements; named-path var_length lineage; RepeatGroup (group_binds NodeAt/EdgeAt + k +
per_rep_pred); ShortestK + collect_trails + edge_pred; VALUE/COUNT correlated subqueries; LET clause;
FILTER + composed paging; GROUP BY after RETURN; ORDER BY alias/NULLS; unquantified subpath group.

### ROUND 7 — "work on the hard features" (125 -> 73; session 265 -> 73, 192 cases)

The "genuinely-hard" clusters flagged at the round-6 stop turned out largely tractable, plus a much
larger set of feature gaps surfaced (64 non-nested engine!=core). Shipped, each gate-green + byte-identical:

- per-rep WHERE on a multi-hop unit (rep-boundary eval, 2) - OPTIONAL MATCH binding an edge var (1)
- correlated scalar VALUE subquery (ScalarSubquery, 3) - uncorrelated multi-pattern EXISTS (4)
- uncorrelated VALUE scalar subquery (2) - repeated-variable equality join (9 — incl ALL multiseg!)
- single-rep + fixed-inner nested groups (2) - FOR..IN list unwind (Plan::Unwind, 10)
- bare ALL/ANY selectors (6) - graph-element predicates IS DIRECTED/SOURCE OF/ALL_DIFFERENT/SAME (5)
- per-hop edge WHERE on a plain var-length (3) - REPEATABLE ELEMENTS/DIFFERENT EDGES modes + labels(edge) (4)
- GROUP BY without aggregate = DISTINCT (1)

### ROUND 8 — "do the 14 and the 27" (73 -> 55; session 265 -> 55, 210 cases cleared)

Cleared 15 specialized features, each gate-green + fuzz-byte-identical (seeds 1 & 42):

- uncorrelated COUNT subquery (`COUNT { MATCH … MATCH … }`) + repeated-var on a FIXED hop (self-loop
  `(u)-[r]->(u)`; cycle-closing comma pattern folds)
- GROUP BY a non-returned key = hidden grouping key + schema-aware ORDER-BY-alias resolution
- leading OPTIONAL MATCH pads one null row when empty (`Plan::NullPadIfEmpty`, tck_null1)
- per-hop edge WHERE may reference the hop SOURCE variable (`(a)-[e WHERE a.k=…]->{…}`)
- interval-overlap hop compares TEMPORAL bounds, not just numeric (contains_window — a real correctness
  bug: date-interval queries silently returned empty)
- group-variable list typing survives a WITH rename (`WITH e AS hops … hops[i].amt`)
- ORDER BY expression can reference an output alias (`ORDER BY (LET x=a IN x END)`)
- IS TYPED RECORD closed schema (`{a::INTEGER, b::STRING [NOT NULL], geo::RECORD {…}}`, 2)
- uncorrelated CALL () {…} empty-scope subquery (cross-join; outer refs isolated to NULL, 2)
- correlated CALL with a COUNT aggregate (LEFT semantics, count 0 for empty)
- edge-label negation `-[:!T]->` / `-[:!(A|B)]->` (complement id set in want_etypes)

### ROUND 9 — extend the fuzzers to emit the hard shapes, then fix (55 -> 55 corpus; 3 fuzzer-found fixes)

The byte-identity safety net that gated the nested cluster now EXISTS. Shared generator
`tests/support/gql_shapes.rs` (quantified var-length, subpath GROUPS with group vars, per-rep WHERE,
shortest, and — behind `Caps::nested` — NESTED groups + group-over-var-length), reducing every binding
to SCALARS so results compare as a multiset. Reused by BOTH `differential_fuzz.rs` (correctness) and
`perf_fuzz.rs` (perf) via `#[path]`. `FUZZ_HARD=supported` (default, CI-green) / `all` (nested driver:
engine parse-error on a shape core supports = hard failure with repro) / `off`. Per-query 2s timeout
guard + `FUZZ_TRACE=1`.

Building it immediately found + fixed 3 real byte-identity bugs in ALREADY-shipped constructs:

- ANY SHORTEST `->+(t)` now admits the SOURCE at the shortest CYCLE length (collect cycle-closing edges
  in the BFS). This IS any_shortest_plus_seed_cycle_len — now cleared (the count already reflects it).
- shortest `->*` and group `*`/`{0,…}` over an UNKNOWN edge type now still emit the zero-rep source
  (fall through with a never-matching set, not the empty "any" set / early empty()).

REMAINING (55 = ~30 value-contract intentional + ~24 feature). The nested cluster is now DRIVABLE:
`FUZZ_HARD=all` red-fails with a concrete repro (first: `( (x)-[e:R]->{0,1}(y) ){1,2} … size(e[1])`).
Parser entry point: `gql.rs::parse_subpath_group` (~2441/2473 bail-outs). Needs a recursive/variable-
stride RepeatGroup materializing nested `Value::List`.

- NESTED-RECURSIVE (14): list-of-lists group vars (nested_paren_lol/varying_1/2, nested_quant_gv_vectorize),
  variable-inner group vars (nested_outer_gv_2), multi-rep decomposition (nested_quant_ends_2/3),
  nested per-rep WHERE (nested_per_rep_where_1..4), nested per-hop edge (nested_per_hop_edge_1/2), vqs_16.
  Needs core's recursive bind_unit + nested Value::List materialization with byte-identical enumeration
  order. The big coherent HARD chunk; verified data model captured (x[outer][inner] depth = enclosing groups).
  NOW fuzzer-driven (`FUZZ_HARD=all`).
- for_drives_batch_optional_match (1) — FOR-driven fresh-var OPTIONAL MATCH `(p:Person {name: name})`: needs
  BOTH a correlated inline-prop EXPRESSION (props() only takes literals today) AND a left-outer correlated
  node scan (no such plan node). Two new capabilities.
- VALUE-CONTRACT (~30, leave baselined by principle): num_string_overflow, distinct_nan, sum/avg-over-temporal
  - mixed, CAST-throws (bool/list/int*null), range caps, m_reserved_word, inline_constraint, hardening (bool\*num
    / str+num / oversized-int / overflow-exponent), date_part strict, faulting_aggregate, call_config*\*\_error,
    zero_bound_3 (core treats {0,0} as {0,1} — engine's 4 is ISO-correct), temporal_duration (core lacks
    duration ordering — engine's compare is correct).

### ROUND 10 — nested subpath groups implemented, fuzzer-driven (55 -> 42; 13 of 14 nested cleared)

`Plan::NestedGroup` over a recursive `GUnit`/`GElem` IR (`exec.rs::nested_group`): a 2-level (single-Sub
inner) double-DFS emitting `levels`-tagged `StepRec`s, materialized into nested `Value::List`s by the
recursive `bind_nested` (a port of core's pathfind model — the fuzzer compares as a multiset, so only the
SET of (endpoint + nested group-var lists) must match, not core's enumeration order). Parser:
`subpath_group_is_nested()` routes to `parse_nested_group`; `group_var_depth` + `field_chain` type
`x[i][j].prop` at the right depth. Each step fuzzer-driven (`FUZZ_HARD=all`), each gate-green:

- basic list-of-lists group vars (7): nested_outer_gv_2, nested_paren_lol, nested_paren_varying_1/2,
  nested_quant_ends_2, nested_quant_gv_vectorize, vqs_16
- per-hop edge WHERE / inline props on the inner hop (2): nested_per_hop_edge_1/2 (`GElem::Hop.edge_pred`)
- per-rep WHERE (4): nested_per_rep_where_1/2/3/4 (per-rep view = one level shallower, `bind_nested`
  key_start=1; parser decrements group_var_depth while parsing the WHERE)

Nested is now DEFAULT fuzz coverage (`Caps::supported() == all`). Fuzzer-found + fixed 3 real bugs in
shipped constructs along the way (round 9): shortest `->+(t)` source-cycle (cleared
any_shortest_plus_seed_cycle_len), and the unknown-edge-type zero-rep source for shortest/group.

REMAINING FEATURE (2, each its own structural extension):

- nested_quant_ends_3 — a MULTI-SEGMENT inner unit (`( ()-[:R]->()-[:R]->{1,2}() ){1}`): the outer body is
  [Hop, Sub], not a single Sub. Needs a general multi-element GUnit matcher (match_seq over outer elems).
- for_drives_batch_optional_match — FOR-driven fresh-var OPTIONAL MATCH: correlated inline-prop EXPRESSIONs
  (props() takes only literals) + a left-outer correlated node scan.
  Plus the deferred empty-inner-rep `{0,n}` epsilon-closure (fuzzer's nested inner is min>=1).
