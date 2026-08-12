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

## 4. ALL SHORTEST / SHORTEST 1 — READY TO IMPLEMENT (worktree agent was blocked by a bad fork base; design below is complete & verified against current code)

Concrete plan (implement in the MAIN tree; agent a5af0ba3 could not because its worktree predated the crate):
- ir.rs: `enum ShortestSelector { Any, All }`; add `min: u32` + `selector: ShortestSelector` to `Plan::ShortestPath` (currently has input/from/dir/edge_label/max); update the `shortest_path(...)` builder signature.
- gql.rs `shortest_path_binding` + `query`: accept `ANY SHORTEST`/`ALL SHORTEST`/`SHORTEST 1[ GROUP|GROUPS]` (→Any; `1 GROUP`→All; k>=2 and k=0 → parse error). Add a BARE-selector entry (no path var) before match_body. Translate inline endpoint/seed `label`+`{props}` into Scan-label + node_prop_filters (seed below the hop, endpoint filters above it; a same-var endpoint like `->{1,3}(a)` → a `Slot(src)=Slot(end)` equality filter). Keep rejecting per-hop edge WHERE. Thread `min` from `*`(0)/`+`(1).
- exec.rs `shortest_path` (+ dispatch ~900): record ALL min-distance predecessors and enumerate the shortest-path DAG (`enumerate_shortest_paths`), emitting one row per distinct shortest path so endpoint multiplicity is right WITH OR WITHOUT lineage (do NOT gate multiplicity on `track`). Endpoints = nodes with `dist >= min`, so `*`(min 0) emits the seed at dist 0 (zero-length-to-self). `Any` keeps only the FIRST predecessor → one row per endpoint (existing 4 unit tests stay green at min=1). Mirror core all_shortest_walk / shortest_ends (crates/lenke-core/src/gql/eval/pathfind.rs). Seed-cycle re-emission is NOT needed for any required case.
- opt.rs: 4 `Plan::ShortestPath { .. }` match arms (rewrite ~202/211, pushdown ~671/681/697) need `min, selector` added to destructure + reconstruction. ALSO cost.rs estimate arm.
- Fixes all 8 all_shortest_*/shortest_1_* cases + incidental ANY-SHORTEST inline-props cases. DEFER SHORTEST k>=2 + shortest_k_per_hop_pred (parse error).

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
