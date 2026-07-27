//! Recursive-descent parser: token stream → `Query` AST. Port of TS `parser.ts`.
//! ISO GQL precedence (loosest→tightest): OR/XOR, AND, NOT, IS/IN predicates,
//! comparison, `||`, +/-, *///%, unary, primary. Label expressions: `|` < `&` < `!`.

use super::ast::*;
use super::lexer::{err, is_reserved, tokenize, SyntaxError, Token, Tt};

/// Map an ISO GQL `CAST` target type name to the conversion function it
/// desugars to. Integer/float/string/bool/list families are representable;
/// anything else (temporal, bytes, record, …) has no home in this value model
/// and returns `None` (a loud CAST error). Mirrors the TS `castTargetFn`.
fn cast_target_fn(type_name: &str) -> Option<&'static str> {
    Some(match type_name.to_ascii_lowercase().as_str() {
        "int" | "integer" | "int8" | "int16" | "int32" | "int64" | "int128" | "int256" | "uint"
        | "uint8" | "uint16" | "uint32" | "uint64" | "uint128" | "uint256" | "bigint"
        | "ubigint" | "smallint" | "usmallint" | "signed" | "unsigned" => "to_integer",
        "float" | "float32" | "float64" | "double" | "decimal" | "real" | "number" | "numeric" => {
            "to_float"
        }
        "string" | "text" | "varchar" | "char" => "to_string",
        "bool" | "boolean" => "to_boolean",
        "list" | "array" => "to_list",
        // Temporal CAST targets desugar to the matching temporal constructor
        // function; `timestamp` is a DATETIME alias. Mirrors the TS `CAST_TEMPORAL`.
        "date" => "date",
        "datetime" | "timestamp" => "datetime",
        "local_datetime" => "local_datetime",
        "local_time" => "local_time",
        "zoned_time" => "zoned_time",
        "zoned_datetime" => "zoned_datetime",
        "duration" => "duration",
        _ => return None,
    })
}

/// Map a type name to the normalized `IS TYPED` category. Reuses the `CAST`
/// vocabulary and adds the temporal kinds + `null`/`nothing`/`any`. Numeric split
/// (`integer` vs `float`) is resolved at eval by boundary inference (lenke has one
/// f64 numeric type). Mirrors the TS `typeTestCategory`.
fn type_test_category(type_name: &str) -> Option<&'static str> {
    Some(match type_name.to_ascii_lowercase().as_str() {
        "int" | "integer" | "int8" | "int16" | "int32" | "int64" | "int128" | "int256" | "uint"
        | "uint8" | "uint16" | "uint32" | "uint64" | "uint128" | "uint256" | "bigint"
        | "ubigint" | "smallint" | "usmallint" | "signed" | "unsigned" => "integer",
        "float" | "float32" | "float64" | "double" | "decimal" | "real" | "number" | "numeric" => {
            "float"
        }
        "string" | "text" | "varchar" | "char" => "string",
        "bool" | "boolean" => "bool",
        "list" | "array" => "list",
        "date" => "date",
        "local_time" => "local_time",
        "local_datetime" | "datetime" | "timestamp" => "local_datetime",
        "zoned_time" => "zoned_time",
        "zoned_datetime" => "zoned_datetime",
        "duration" => "duration",
        "null" | "nothing" => "null",
        "any" => "any",
        _ => return None,
    })
}

/// AND a parenthesized-subpath `WHERE` into the last element the subpath binds —
/// the final segment's node, or the start node when the subpath is a single node.
/// That element binds last, so its inline `WHERE` sees every variable in the
/// subpath; evaluating there is exact for a non-quantified subpath.
fn attach_subpath_where(pat: &mut PathPattern, cond: Expr) {
    let target = match pat.segments.last_mut() {
        Some(seg) => &mut seg.node.where_,
        None => &mut pat.start.where_,
    };
    *target = Some(match target.take() {
        Some(existing) => Expr::And(vec![existing, cond]),
        None => cond,
    });
}

/// The single, consistent reserved-word rejection used in every binding
/// position. `what` names the role (a label name, a variable, …). The message
/// names backticks explicitly and echoes the user's ORIGINAL casing in both the
/// name and the suggested delimited form — `Keyword` tokens lowercase `value`,
/// so `raw` carries the exact spelling (`` `Order` ``, never `` `order` ``).
fn reserved_error(tok: &Token, what: &str) -> SyntaxError {
    let original = tok.raw.as_deref().unwrap_or(&tok.value);
    SyntaxError {
        message: format!(
            "`{original}` is a reserved word and can't be used bare as {what}; \
             quote it as a delimited identifier with backticks: `{original}`"
        ),
        pos: tok.pos,
    }
}

/// Parse a top-level [`Statement`]: either a linear query or an ISO GQL
/// transaction-control command (`START TRANSACTION`/`COMMIT`/`ROLLBACK`). The FFI
/// query path (`lnk_query_rows`/`lnk_query_arrow`) dispatches on the returned
/// variant. For the query grammar alone, [`parse_with_dialect`] returns a bare
/// [`Query`].
pub fn parse(src: &str) -> Result<Statement, SyntaxError> {
    parse_with_max_chain(src, DEFAULT_MAX_CHAIN)
}

/// Like [`parse`] but with a caller-supplied operator-chain ceiling (see
/// [`DEFAULT_MAX_CHAIN`]). The FFI query path passes the graph's configured value
/// (`Graph::max_operator_chain`); everything else uses [`parse`]'s default.
pub fn parse_with_max_chain(src: &str, max_chain: usize) -> Result<Statement, SyntaxError> {
    let tokens = tokenize(src)?;
    let mut p = Parser {
        tokens,
        pos: 0,
        depth: 0,
        in_operator_enabled: true,
        dialect: Dialect::Lenke,
        max_chain,
    };
    p.parse_statement()
}

/// Parse a bare boolean predicate — a `WHERE`-clause expression — into its `Expr`
/// AST. This is the compiler surface a declarative VALIDATOR constraint needs
/// (`create_validator("User", "u", "u.age >= 0")`): it runs the *same* ISO
/// expression grammar as a real `WHERE`, then asserts the whole predicate was
/// consumed. Returns a `SyntaxError` (→ `E_SYNTAX` at the FFI boundary) on an
/// unparseable predicate or one with trailing input. Byte-identical predicate
/// semantics with the TS `parsePredicate`.
pub fn parse_predicate(src: &str) -> Result<Expr, SyntaxError> {
    let tokens = tokenize(src)?;
    let mut p = Parser {
        tokens,
        pos: 0,
        depth: 0,
        in_operator_enabled: true,
        dialect: Dialect::Lenke,
        max_chain: DEFAULT_MAX_CHAIN,
    };
    let e = p.parse_expr()?;
    if !p.at_end() {
        let t = p.peek();
        return err(
            format!(
                "Unexpected trailing input '{}' in validator predicate",
                t.value
            ),
            t.pos,
        );
    }
    Ok(e)
}

/// Parse under an explicit [`Dialect`]. `IsoStrict` treats sigil extensions
/// (`_MERGE`) as ordinary identifiers, so an extension clause is a syntax error —
/// the differential/conformance harness parses under it to prove the ISO surface
/// stays self-contained. See docs/design/gql-extensions.md §1.
pub fn parse_with_dialect(src: &str, dialect: Dialect) -> Result<Query, SyntaxError> {
    let tokens = tokenize(src)?;
    let mut p = Parser {
        tokens,
        pos: 0,
        depth: 0,
        in_operator_enabled: true,
        dialect,
        max_chain: DEFAULT_MAX_CHAIN,
    };
    p.parse_query()
}

/// Parse the bare query grammar (Lenke dialect) under a caller-supplied
/// operator-chain ceiling. The prepared-statement path uses this so a prepared
/// query honours the same `maxOperatorChain` option as a one-shot query.
pub fn parse_query_with_max_chain(src: &str, max_chain: usize) -> Result<Query, SyntaxError> {
    let tokens = tokenize(src)?;
    let mut p = Parser {
        tokens,
        pos: 0,
        depth: 0,
        in_operator_enabled: true,
        dialect: Dialect::Lenke,
        max_chain,
    };
    p.parse_query()
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    /// Current recursive-descent nesting depth (see [`Parser::descend`]).
    depth: u32,
    /// Parse dialect — gates sigil extensions like `_MERGE` (see [`Parser::check_ext`]).
    dialect: Dialect,
    /// Operator-chain ceiling for this parse (see [`DEFAULT_MAX_CHAIN`]).
    max_chain: usize,
    /// Whether a bare `IN` is the membership operator (default) or a structural
    /// keyword. Suppressed only while parsing a `LET … IN … END` binding's RHS so
    /// the binding's value expression ends at the structural `IN`; re-enabled
    /// inside any grouped sub-expression (see [`Parser::parse_primary`]).
    in_operator_enabled: bool,
}

/// Recursion-depth ceiling. Recursive descent over deeply nested input
/// (`((((…))))`, `NOT NOT NOT …`, `!!!…`, nested lists / subqueries) would
/// otherwise overflow the native stack and *abort the process* — uncatchable.
/// Past this bound the recursive entry points return a `SyntaxError` instead.
///
/// Set well below the TS limit (500): a debug-build stack frame for the parser
/// chain is large, and Rust threads default to a 2 MiB stack, so the guard must
/// fire with margin to spare. 128 levels of nesting is far beyond any real
/// query yet leaves the descent comfortably within a 2 MiB stack.
const MAX_DEPTH: u32 = 128;

/// Default operator-chain sanity ceiling. The associative operator nodes are
/// n-ary (a flat `Vec`, see `ast::Expr`), so a long chain like `true AND true
/// AND … (500k)` is *not* a chain-deep tree and every walk over it (eval, compile,
/// drop, the analysis passes) is a loop bounded by real nesting depth — no
/// stack-overflow risk regardless of chain length or thread-stack size. This bound
/// is therefore not crash-safety; it's a pure anti-resource-abuse guard (each
/// operand is an allocation + an eval step), set far beyond any legitimate hand-
/// or machine-generated predicate. Configurable per graph
/// (`createEmptyGraph(backend, { maxOperatorChain })`), mirrored by the TS parser.
pub(crate) const DEFAULT_MAX_CHAIN: usize = 10_000;

type R<T> = Result<T, SyntaxError>;

impl Parser {
    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }
    /// One token past `peek` (clamped to EOF), for two-token lookahead.
    fn peek2(&self) -> &Token {
        let i = (self.pos + 1).min(self.tokens.len() - 1);
        &self.tokens[i]
    }
    fn at_end(&self) -> bool {
        self.peek().tt == Tt::Eof
    }
    fn advance(&mut self) -> Token {
        let t = self.tokens[self.pos].clone();
        self.pos += 1;
        t
    }
    fn check(&self, tt: Tt) -> bool {
        self.peek().tt == tt
    }
    fn check_kw(&self, kw: &str) -> bool {
        let t = self.peek();
        t.tt == Tt::Keyword && t.value == kw
    }
    /// A non-reserved keyword that arrives as an `ident` (LABELED, FIRST, LAST).
    fn check_soft(&self, word: &str) -> bool {
        let t = self.peek();
        t.tt == Tt::Ident && t.value.eq_ignore_ascii_case(word)
    }
    /// A sigil extension keyword (`_MERGE`, `_ON_CREATE`, …) — lexes as an ident,
    /// matched case-insensitively, never when backtick-delimited, and ONLY under
    /// the `Lenke` dialect, so it can never shrink the ISO identifier namespace.
    /// Mirrors the TS `checkExtIdent`.
    fn check_ext(&self, word: &str) -> bool {
        let t = self.peek();
        self.dialect == Dialect::Lenke
            && t.tt == Tt::Ident
            && !t.delimited
            && t.value.eq_ignore_ascii_case(word)
    }
    fn expect(&mut self, tt: Tt, what: &str) -> R<Token> {
        if !self.check(tt) {
            let t = self.peek();
            let got = if t.value.is_empty() {
                format!("{:?}", t.tt)
            } else {
                t.value.clone()
            };
            return err(format!("Expected {what}, got '{got}'"), t.pos);
        }
        Ok(self.advance())
    }
    fn expect_kw(&mut self, kw: &str) -> R<Token> {
        if !self.check_kw(kw) {
            let t = self.peek();
            let got = if t.value.is_empty() {
                format!("{:?}", t.tt)
            } else {
                t.value.clone()
            };
            return err(
                format!("Expected '{}', got '{got}'", kw.to_uppercase()),
                t.pos,
            );
        }
        Ok(self.advance())
    }

    /// Consume a soft keyword (a reserved word that lexes as an ident, e.g.
    /// `SELECT`/`FROM`/`HAVING`) or error. The counterpart of [`expect_kw`] for
    /// the reserved-but-not-structural words.
    fn expect_soft(&mut self, word: &str) -> R<Token> {
        if !self.check_soft(word) {
            let t = self.peek();
            let got = if t.value.is_empty() {
                format!("{:?}", t.tt)
            } else {
                t.value.clone()
            };
            return err(
                format!("Expected '{}', got '{got}'", word.to_uppercase()),
                t.pos,
            );
        }
        Ok(self.advance())
    }

    /// Run `body` one level deeper, guarding against unbounded recursion. Used to
    /// wrap the recursive entry points so deep nesting yields a `SyntaxError`
    /// rather than a stack-overflow abort.
    fn descend<T>(&mut self, body: impl FnOnce(&mut Self) -> R<T>) -> R<T> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            self.depth -= 1;
            return err("Query nested too deeply", self.peek().pos);
        }
        let r = body(self);
        self.depth -= 1;
        r
    }

    /// Consume a `Number` token already known to be present and require it to be a
    /// non-negative integer — for SKIP/LIMIT/OFFSET and quantifier bounds, where
    /// a float, NaN, or out-of-range value is never valid.
    fn read_count(&mut self, what: &str) -> R<u32> {
        let t = self.advance();
        let n = t.num.unwrap_or(f64::NAN);
        if !n.is_finite() || n.fract() != 0.0 || n < 0.0 || n > u32::MAX as f64 {
            return err(
                format!("{what} must be a non-negative integer, got '{}'", t.value),
                t.pos,
            );
        }
        Ok(n as u32)
    }

    fn expect_count(&mut self, what: &str) -> R<usize> {
        if !self.check(Tt::Number) {
            let t = self.peek();
            let got = if t.value.is_empty() {
                format!("{:?}", t.tt)
            } else {
                t.value.clone()
            };
            return err(format!("Expected {what}, got '{got}'"), t.pos);
        }
        Ok(self.read_count(what)? as usize)
    }

    /// A `LIMIT` / `OFFSET` bound: an integer literal, or — when `allow_param` — a
    /// dynamic `$param` (ISO `nonNegativeIntegerSpecification`, opengql:2268). The
    /// param's bound value is validated to be a non-negative integer at execution.
    /// `SKIP` passes `allow_param=false`, so `SKIP $x` stays rejected (Cypher-only).
    fn expect_count_bound(&mut self, what: &str, allow_param: bool) -> R<CountBound> {
        if allow_param && self.check(Tt::Param) {
            return Ok(CountBound::Param(self.advance().value));
        }
        Ok(CountBound::Lit(self.expect_count(what)?))
    }

    /// Consume an identifier in a binding position (variable, label, key, alias).
    /// A bare reserved word is rejected; a delimited identifier may be any word.
    /// Both token classes that can't be a bare name here are caught up front so
    /// the rejection is uniform: a structural `Keyword` token (`Order`, `Count`,
    /// `Match`, `Set`, …) — which would otherwise fail `expect(Ident)` with a
    /// generic, casing-losing message — and a reserved-but-not-structural `Ident`
    /// (`Group`, `Product`).
    fn bind_name(&mut self, what: &str) -> R<String> {
        let t = self.peek();
        if t.tt == Tt::Keyword || (t.tt == Tt::Ident && !t.delimited && is_reserved(&t.value)) {
            return Err(reserved_error(t, what));
        }
        Ok(self.expect(Tt::Ident, what)?.value)
    }

    // --- patterns ----------------------------------------------------------

    fn parse_property_map(&mut self) -> R<Vec<PropertyConstraint>> {
        self.expect(Tt::LBrace, "'{'")?;
        let mut props = Vec::new();
        if !self.check(Tt::RBrace) {
            loop {
                let key = self.bind_name("a property name")?;
                self.expect(Tt::Colon, "':'")?;
                props.push(PropertyConstraint {
                    key,
                    value: self.parse_expr()?,
                });
                if self.check(Tt::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(Tt::RBrace, "'}'")?;
        Ok(props)
    }

    fn parse_predicate(&mut self) -> R<(Vec<PropertyConstraint>, Option<Expr>)> {
        let props = if self.check(Tt::LBrace) {
            self.parse_property_map()?
        } else {
            Vec::new()
        };
        let where_ = if self.check_kw("where") {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };
        Ok((props, where_))
    }

    fn parse_node(&mut self) -> R<NodePattern> {
        self.expect(Tt::LParen, "'('")?;
        let variable = if self.check(Tt::Ident) {
            Some(self.bind_name("a variable")?)
        } else {
            None
        };
        let label = if self.check(Tt::Colon) || self.check_kw("is") {
            self.advance();
            Some(self.parse_label_expr()?)
        } else {
            None
        };
        let (props, where_) = self.parse_predicate()?;
        self.expect(Tt::RParen, "')'")?;
        Ok(NodePattern {
            variable,
            label,
            props,
            where_,
        })
    }

    fn parse_label_expr(&mut self) -> R<LabelExpr> {
        self.descend(|p| p.parse_label_or())
    }
    fn parse_label_or(&mut self) -> R<LabelExpr> {
        let mut left = self.parse_label_and()?;
        while self.check(Tt::Pipe) {
            self.advance();
            left = LabelExpr::Or(Box::new(left), Box::new(self.parse_label_and()?));
        }
        Ok(left)
    }
    fn parse_label_and(&mut self) -> R<LabelExpr> {
        let mut left = self.parse_label_not()?;
        while self.check(Tt::Amp) {
            self.advance();
            left = LabelExpr::And(Box::new(left), Box::new(self.parse_label_not()?));
        }
        Ok(left)
    }
    fn parse_label_not(&mut self) -> R<LabelExpr> {
        self.descend(|p| {
            if p.check(Tt::Bang) {
                p.advance();
                return Ok(LabelExpr::Not(Box::new(p.parse_label_not()?)));
            }
            p.parse_label_primary()
        })
    }
    fn parse_label_primary(&mut self) -> R<LabelExpr> {
        if self.check(Tt::Percent) {
            self.advance();
            return Ok(LabelExpr::Wildcard);
        }
        if self.check(Tt::LParen) {
            self.advance();
            let inner = self.parse_label_expr()?;
            self.expect(Tt::RParen, "')' to close a label expression")?;
            return Ok(inner);
        }
        Ok(LabelExpr::Label(self.bind_name("a label name")?))
    }

    #[allow(
        clippy::type_complexity,
        reason = "ad-hoc rel-detail tuple (variable, label, property constraints, WHERE) consumed once by the caller"
    )]
    fn parse_rel_detail(
        &mut self,
    ) -> R<(
        Option<String>,
        Option<LabelExpr>,
        Vec<PropertyConstraint>,
        Option<Expr>,
    )> {
        self.expect(Tt::LBracket, "'['")?;
        let variable = if self.check(Tt::Ident) {
            Some(self.bind_name("a variable")?)
        } else {
            None
        };
        let label = if self.check(Tt::Colon) || self.check_kw("is") {
            self.advance();
            Some(self.parse_label_expr()?)
        } else {
            None
        };
        let (props, where_) = self.parse_predicate()?;
        self.expect(Tt::RBracket, "']'")?;
        Ok((variable, label, props, where_))
    }

    fn parse_rel(&mut self) -> R<RelPattern> {
        // Pure abbreviated forms first: `->`, `<->`, `~>`.
        let abbrev = match self.peek().tt {
            Tt::RArrow => Some(Direction::Out),
            Tt::LRArrow => Some(Direction::Both),
            Tt::TildeR => Some(Direction::Both),
            _ => None,
        };
        if let Some(direction) = abbrev {
            self.advance();
            return Ok(RelPattern {
                variable: None,
                label: None,
                direction,
                props: Vec::new(),
                where_: None,
                quantifier: None,
            });
        }

        // Left marker.
        let mut left_arrow = false;
        if self.check(Tt::LArrow) {
            self.advance();
            left_arrow = true;
        } else if self.check(Tt::Dash) || self.check(Tt::Tilde) || self.check(Tt::LTilde) {
            self.advance();
        } else {
            let t = self.peek();
            let got = if t.value.is_empty() {
                format!("{:?}", t.tt)
            } else {
                t.value.clone()
            };
            return err(
                format!("Expected a relationship (e.g. -[:T]->, <-[:T]-, ~[:T]~, ->), got '{got}'"),
                t.pos,
            );
        }

        // No bracket → abbreviated edge.
        if !self.check(Tt::LBracket) {
            return Ok(RelPattern {
                variable: None,
                label: None,
                direction: if left_arrow {
                    Direction::In
                } else {
                    Direction::Both
                },
                props: Vec::new(),
                where_: None,
                quantifier: None,
            });
        }

        let (variable, label, props, where_) = self.parse_rel_detail()?;

        // Closing marker.
        let mut right_arrow = false;
        if self.check(Tt::RArrow) {
            self.advance();
            right_arrow = true;
        } else if self.check(Tt::Dash) || self.check(Tt::Tilde) || self.check(Tt::TildeR) {
            self.advance();
        } else {
            let t = self.peek();
            let got = if t.value.is_empty() {
                format!("{:?}", t.tt)
            } else {
                t.value.clone()
            };
            return err(
                format!("Expected ']->', ']-' or ']~' to close a relationship, got '{got}'"),
                t.pos,
            );
        }

        let direction = if right_arrow && !left_arrow {
            Direction::Out
        } else if left_arrow && !right_arrow {
            Direction::In
        } else {
            Direction::Both
        };
        Ok(RelPattern {
            variable,
            label,
            direction,
            props,
            where_,
            quantifier: None,
        })
    }

    fn starts_relationship(&self) -> bool {
        matches!(
            self.peek().tt,
            Tt::Dash | Tt::LArrow | Tt::RArrow | Tt::LRArrow | Tt::Tilde | Tt::LTilde | Tt::TildeR
        )
    }

    /// ISO quantified parenthesized subpath `( (x)-[e]->(y) [WHERE cond] ) {n,m}`
    /// (Phase 1: a single-edge repetition unit). The per-repetition `cond` may
    /// reference the hop's source `x`, edge `e`, and target `y`; `y` is also the
    /// landing/endpoint node. The opening `((` is at the cursor.
    fn parse_quantified_subpath(&mut self) -> R<Segment> {
        let open = self.peek().pos;
        self.expect(Tt::LParen, "'(' to open a quantified subpath")?;
        let hop_from = self.parse_node()?; // inner (x)
        if !self.starts_relationship() {
            return err(
                "a quantified subpath must contain a relationship, e.g. `((x)-[e]->(y)){1,3}`",
                self.peek().pos,
            );
        }
        let mut rel = self.parse_rel()?;
        let node = self.parse_node()?; // inner (y) = the landing node
        if self.starts_relationship() {
            return err(
                "a quantified subpath with more than one relationship is not yet supported",
                self.peek().pos,
            );
        }
        // Phase 1: the inner nodes carry only a variable — put any per-hop label /
        // property test in the subpath `WHERE` (`((x)-[e]->(y) WHERE x:Account AND
        // e.amt > 0){1,3}`), so nothing is silently dropped.
        for n in [&hop_from, &node] {
            if n.label.is_some() || !n.props.is_empty() || n.where_.is_some() {
                return err(
                    "a label or property on a quantified-subpath node isn't supported yet — \
                     put the per-hop test in the subpath `WHERE` instead",
                    open,
                );
            }
        }
        // The subpath `WHERE` (over x/e/y) becomes the per-repetition predicate,
        // AND-ed with any inline `-[e {…} WHERE …]->` predicate.
        if self.check_kw("where") {
            self.advance();
            let cond = self.parse_expr()?;
            rel.where_ = Some(match rel.where_.take() {
                Some(w) => Expr::And(vec![w, cond]),
                None => cond,
            });
        }
        self.expect(Tt::RParen, "')' to close a quantified subpath")?;
        let Some(q) = self.parse_quantifier()? else {
            return err(
                "a parenthesized subpath must be quantified (`( … ){n,m}`)",
                open,
            );
        };
        rel.quantifier = Some(q);
        Ok(Segment {
            rel,
            node,
            hop_from: Some(hop_from),
        })
    }

    fn parse_quantifier(&mut self) -> R<Option<Quantifier>> {
        if self.check(Tt::Star) {
            self.advance();
            return Ok(Some(Quantifier { min: 0, max: None }));
        }
        if self.check(Tt::Plus) {
            self.advance();
            return Ok(Some(Quantifier { min: 1, max: None }));
        }
        if self.check(Tt::LBrace) {
            let open = self.advance();
            let min = if self.check(Tt::Number) {
                self.read_count("a quantifier bound")?
            } else {
                0
            };
            let mut max = Some(min);
            if self.check(Tt::Comma) {
                self.advance();
                max = if self.check(Tt::Number) {
                    Some(self.read_count("a quantifier bound")?)
                } else {
                    None
                };
            }
            self.expect(Tt::RBrace, "'}' to close a quantifier")?;
            if let Some(m) = max {
                if m < min {
                    return err(
                        format!("Quantifier upper bound {m} is less than lower bound {min}"),
                        open.pos,
                    );
                }
            }
            return Ok(Some(Quantifier { min, max }));
        }
        Ok(None)
    }

    fn parse_path_pattern(&mut self) -> R<PathPattern> {
        self.descend(|p| {
            // Optional path variable: `p = …`. At the start of a pattern an
            // identifier followed by `=` can only be a path-variable binding (a
            // node pattern always opens with `(`).
            let sel_pos = p.peek().pos;
            let path_var = if p.check(Tt::Ident)
                && matches!(p.tokens.get(p.pos + 1), Some(n) if n.tt == Tt::Eq)
            {
                let name = p.advance().value;
                p.advance(); // '='
                Some(name)
            } else {
                None
            };

            // ISO `<parenthesized path pattern expression>`: `( <path> [WHERE
            // cond] )` — a subpath grouping whose WHERE is scoped to (and applied
            // as part of) the subpath. Detected by a paren that immediately opens
            // another paren: a node pattern never nests a `(`, so `((` can only
            // begin a subpath. The inner WHERE is a *pattern* predicate, distinct
            // from the clause-level `WHERE` that follows the whole MATCH.
            if p.check(Tt::LParen) && p.peek2().tt == Tt::LParen {
                if path_var.is_some() {
                    return err(
                        "a path variable on a parenthesized subpath (`p = ( … )`) is not yet supported",
                        sel_pos,
                    );
                }
                p.advance(); // outer '('
                let mut inner = p.parse_path_pattern()?;
                if p.check_kw("where") {
                    p.advance();
                    let cond = p.parse_expr()?;
                    attach_subpath_where(&mut inner, cond);
                }
                p.expect(Tt::RParen, "')' to close a parenthesized subpath")?;
                // A quantifier on the whole subpath (`( … )+`) would make the WHERE
                // a per-iteration predicate on a variable-length path — a different
                // matcher. Reject loudly rather than silently mishandle it.
                if p.check(Tt::Star) || p.check(Tt::Plus) || p.check(Tt::LBrace) {
                    return err(
                        "a quantifier on a parenthesized subpath (`( … )+`) is not yet supported",
                        p.peek().pos,
                    );
                }
                return Ok(inner);
            }

            // Optional path selector (`ANY SHORTEST`) then optional mode (`TRAIL`).
            let selector = p.parse_path_selector()?;
            let mode = p.parse_path_mode();

            let start = p.parse_node()?;
            let mut segments = Vec::new();
            loop {
                // A quantified PARENTHESIZED subpath `((x)-[e]->(y) WHERE …){n,m}`
                // (ISO `<parenthesized path pattern expression> <quantifier>`): the
                // per-repetition predicate may reference the hop's SOURCE node `x`,
                // edge `e`, and TARGET node `y`. A `((` can only open a subpath (a
                // node pattern never nests a `(`).
                if p.check(Tt::LParen) && p.peek2().tt == Tt::LParen {
                    segments.push(p.parse_quantified_subpath()?);
                    continue;
                }
                if !p.starts_relationship() {
                    break;
                }
                let mut rel = p.parse_rel()?;
                rel.quantifier = p.parse_quantifier()?;
                // A quantified segment may carry a *per-hop* predicate — inline
                // properties or a `WHERE` — applied to every edge of the walk
                // (`(a)-[e:R WHERE e.amt > $t]->{1,5}(b)`). The optional edge
                // variable is scoped to that predicate (it names each hop's edge in
                // turn); it is not yet a group/list variable exposed to the outer
                // query, so referencing it outside the segment reads as null.
                let node = p.parse_node()?;
                segments.push(Segment {
                    rel,
                    node,
                    hop_from: None,
                });
            }

            // Every selector (and a bare path variable) works over exactly one
            // variable-length segment. `ANY SHORTEST`/`ALL SHORTEST` additionally
            // need the `*`/`+` (min ≤ 1) shortest shape and reject a per-hop
            // predicate (their BFS drivers don't filter); `ANY`/`SHORTEST k` and
            // bare path binding enumerate trails, so they accept any bound and a
            // per-hop predicate.
            let single_varlen = segments.len() == 1 && segments[0].rel.quantifier.is_some();
            match selector {
                PathSelector::AnyShortest | PathSelector::AllShortest => {
                    let ok = single_varlen
                        && segments[0].rel.quantifier.is_some_and(|q| q.min <= 1);
                    if !ok {
                        return err(
                            "ANY SHORTEST / ALL SHORTEST currently support a single variable-length segment with a `*` or `+` (min ≤ 1) quantifier, e.g. `(a)-[]->*(b)`",
                            sel_pos,
                        );
                    }
                    let seg = &segments[0].rel;
                    if !seg.props.is_empty() || seg.where_.is_some() {
                        return err(
                            "a per-hop edge predicate on a variable-length segment is not yet supported together with a path selector (ANY/ALL SHORTEST)",
                            sel_pos,
                        );
                    }
                }
                PathSelector::Any | PathSelector::ShortestK { .. } => {
                    if !single_varlen {
                        return err(
                            "ANY / SHORTEST k currently support a single variable-length segment, e.g. `(a)-[]->{1,5}(b)`",
                            sel_pos,
                        );
                    }
                }
                PathSelector::Walk if path_var.is_some() => {
                    // Bare path binding (`p = (a)-[:R]->{m,n}(b)`) enumerates a single
                    // variable-length segment; fixed-length / multi-segment path
                    // binding is not yet wired.
                    if !single_varlen {
                        return err(
                            "a named path variable currently requires either a path selector (e.g. `p = ANY SHORTEST …`) or a single variable-length segment (e.g. `p = (a)-[:R]->{1,5}(b)`)",
                            sel_pos,
                        );
                    }
                }
                PathSelector::Walk => {}
            }

            Ok(PathPattern {
                start,
                segments,
                path_var,
                selector,
                mode,
            })
        })
    }

    /// Parse an optional ISO path mode after the selector, before the pattern
    /// (`WALK`/`TRAIL`/`SIMPLE`/`ACYCLIC`). Contextual (non-reserved) — recognized
    /// only here, so `walk`/`simple`/… stay usable as ordinary identifiers. The
    /// default is `TRAIL` (matching Neo4j/Fabric; bare `WALK` is unusable
    /// unbounded), which is also the engine's existing var-length behaviour.
    fn parse_path_mode(&mut self) -> PathMode {
        let mode = if self.check_soft("walk") {
            PathMode::Walk
        } else if self.check_soft("trail") {
            PathMode::Trail
        } else if self.check_soft("simple") {
            PathMode::Simple
        } else if self.check_soft("acyclic") {
            PathMode::Acyclic
        } else {
            return PathMode::Trail;
        };
        self.advance();
        mode
    }

    /// Parse an optional ISO path selector prefixing a pattern: `ANY [SHORTEST]`,
    /// `ALL [SHORTEST]`, `SHORTEST k [GROUP[S]]`. Bare `ALL` folds into `Walk` (the
    /// enumerate-all default).
    fn parse_path_selector(&mut self) -> R<PathSelector> {
        let pos = self.peek().pos;
        if self.check_kw("any") {
            self.advance();
            if self.check_kw("shortest") {
                self.advance();

                return Ok(PathSelector::AnyShortest);
            }

            // Bare `ANY` — one arbitrary path per endpoint.
            return Ok(PathSelector::Any);
        }
        if self.check_kw("all") {
            self.advance();
            if self.check_kw("shortest") {
                self.advance();

                return Ok(PathSelector::AllShortest);
            }

            // Bare `ALL` is the ISO default selector — every matching path. That is
            // exactly the enumerate-all behaviour of `Walk` (no selector), so it
            // maps there and needs no downstream change.
            return Ok(PathSelector::Walk);
        }
        if self.check_kw("shortest") {
            self.advance();
            // `SHORTEST k [GROUP[S]]`: k shortest paths per endpoint, or (GROUP)
            // every path in the k smallest length-groups.
            if !self.check(Tt::Number) {
                return err(
                    "SHORTEST must be followed by a count (e.g. `SHORTEST 3`) or written as `ANY SHORTEST`",
                    pos,
                );
            }
            let k = self.read_count("SHORTEST k")?;
            if k == 0 {
                return err("SHORTEST k requires k >= 1", pos);
            }
            // `GROUP`/`GROUPS` is a soft keyword (arrives as an identifier).
            let group = self.check_soft("group") || self.check_soft("groups");
            if group {
                self.advance();
            }

            return Ok(PathSelector::ShortestK { k, group });
        }

        Ok(PathSelector::Walk)
    }

    /// Is the token right after the current one the keyword `kw`? (Used to tell
    /// `OPTIONAL CALL` from `OPTIONAL MATCH` without consuming.)
    fn kw_after(&self, kw: &str) -> bool {
        matches!(self.tokens.get(self.pos + 1), Some(t) if t.tt == Tt::Keyword && t.value == kw)
    }

    /// `[OPTIONAL] CALL name(args) [YIELD col [AS alias], …]`. The inline-subquery
    /// form (`CALL { … }` / `CALL (…) { … }`) is not yet supported — it's rejected
    /// with a pointed message rather than mis-parsed as a named call.
    fn parse_call_clause(&mut self) -> R<Clause> {
        let optional = self.check_kw("optional") && {
            self.advance();
            true
        };
        self.expect_kw("call")?;

        // Inline subquery form: `CALL { … }` or `CALL (scope) { … }`.
        if self.check(Tt::LBrace) || self.check(Tt::LParen) {
            return self.parse_inline_call(optional);
        }

        // Procedure reference: a (possibly dotted) name.
        let mut name = self.bind_name("a procedure name")?;
        while self.check(Tt::Dot) {
            self.advance();
            name.push('.');
            name.push_str(&self.bind_name("a procedure name segment")?);
        }

        self.expect(Tt::LParen, "'(' after a procedure name")?;
        // The procedure's config, as a `{key: value}` map argument (or nothing).
        let config = if self.check(Tt::LBrace) {
            self.parse_property_map()?
        } else {
            Vec::new()
        };
        self.expect(Tt::RParen, "')' to close procedure arguments")?;

        let yields = if self.check_kw("yield") {
            self.advance();
            let mut items = vec![self.parse_yield_item()?];
            while self.check(Tt::Comma) {
                self.advance();
                items.push(self.parse_yield_item()?);
            }
            Some(items)
        } else {
            None
        };

        Ok(Clause::CallNamed(CallNamed {
            optional,
            name,
            config,
            yields,
        }))
    }

    /// `[OPTIONAL] CALL (scope) { <linear query> }` — an inline subquery.
    fn parse_inline_call(&mut self, optional: bool) -> R<Clause> {
        let mut scope = Vec::new();
        if self.check(Tt::LParen) {
            self.advance();
            if !self.check(Tt::RParen) {
                scope.push(self.bind_name("a scoped variable")?);
                while self.check(Tt::Comma) {
                    self.advance();
                    scope.push(self.bind_name("a scoped variable")?);
                }
            }
            self.expect(Tt::RParen, "')' to close the variable scope")?;
        }

        self.expect(Tt::LBrace, "'{' to open an inline subquery")?;
        let body = self.parse_set_op_query()?;
        self.expect(Tt::RBrace, "'}' to close an inline subquery")?;

        Ok(Clause::CallInline(CallInline {
            optional,
            scope,
            body,
        }))
    }

    fn parse_yield_item(&mut self) -> R<YieldItem> {
        let name = self.bind_name("a YIELD column name")?;
        let alias = if self.check_kw("as") {
            self.advance();
            Some(self.bind_name("a YIELD alias")?)
        } else {
            None
        };

        Ok(YieldItem { name, alias })
    }

    fn parse_match_clause(&mut self) -> R<MatchClause> {
        let optional = if self.check_kw("optional") {
            self.advance();
            true
        } else {
            false
        };
        self.expect_kw("match")?;
        // Optional ISO match mode — `REPEATABLE ELEMENTS` / `DIFFERENT EDGES` —
        // applies to every path in this MATCH.
        let match_mode = self.parse_match_mode();
        let mut patterns = vec![self.parse_path_pattern()?];
        while self.check(Tt::Comma) {
            self.advance();
            patterns.push(self.parse_path_pattern()?);
        }
        if let Some(mode) = match_mode {
            for p in &mut patterns {
                p.mode = mode;
            }
        }
        let where_ = if self.check_kw("where") {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };
        Ok(MatchClause {
            optional,
            patterns,
            where_,
        })
    }

    /// ISO `<match mode>`: `REPEATABLE (ELEMENT [BINDINGS] | ELEMENTS)` → `Walk`
    /// (elements may repeat); `DIFFERENT (EDGE [BINDINGS] | EDGES)` → `Trail`
    /// (no edge repeated — lenke's default). `None` if no mode is present.
    fn parse_match_mode(&mut self) -> Option<PathMode> {
        let soft = |p: &Self, w: &str| p.check_kw(w) || p.check_soft(w);
        if soft(self, "repeatable") {
            self.advance();
            if soft(self, "elements") {
                self.advance();
            } else if soft(self, "element") {
                self.advance();
                if soft(self, "bindings") {
                    self.advance();
                }
            }
            return Some(PathMode::Walk);
        }
        if soft(self, "different") {
            self.advance();
            if soft(self, "edges") {
                self.advance();
            } else if soft(self, "edge") {
                self.advance();
                if soft(self, "bindings") {
                    self.advance();
                }
            }
            return Some(PathMode::Trail);
        }
        None
    }

    // --- expressions -------------------------------------------------------

    fn parse_expr(&mut self) -> R<Expr> {
        self.descend(|p| p.parse_or_xor())
    }

    /// Bound a left-associative operator chain (see [`MAX_CHAIN`]). Called once
    /// per iteration of an iterative binary-operator loop so an over-long chain
    /// becomes a `SyntaxError` before it can build a stack-overflowing AST.
    fn chain_limit(&self, count: usize) -> R<()> {
        if count > self.max_chain {
            return err("Operator chain too long", self.peek().pos);
        }
        Ok(())
    }

    fn parse_or_xor(&mut self) -> R<Expr> {
        let mut left = self.parse_and()?;
        let mut chain = 0;
        // Flatten a maximal run of the SAME operator into one n-ary node; nest on
        // an operator switch (both are left-associative at one precedence level),
        // so `a OR b OR c XOR d` is `Xor([Or([a,b,c]), d])`.
        while self.check_kw("or") || self.check_kw("xor") {
            chain += 1;
            self.chain_limit(chain)?;
            let is_or = self.advance().value == "or";
            let mut items = vec![left, self.parse_and()?];
            while (is_or && self.check_kw("or")) || (!is_or && self.check_kw("xor")) {
                chain += 1;
                self.chain_limit(chain)?;
                self.advance();
                items.push(self.parse_and()?);
            }
            left = if is_or {
                Expr::Or(items)
            } else {
                Expr::Xor(items)
            };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> R<Expr> {
        let first = self.parse_not()?;
        if !self.check_kw("and") {
            return Ok(first);
        }
        let mut items = vec![first];
        let mut chain = 0;
        while self.check_kw("and") {
            chain += 1;
            self.chain_limit(chain)?;
            self.advance();
            items.push(self.parse_not()?);
        }
        Ok(Expr::And(items))
    }

    fn parse_not(&mut self) -> R<Expr> {
        self.descend(|p| {
            if p.check_kw("not") {
                p.advance();
                return Ok(Expr::Not(Box::new(p.parse_not()?)));
            }
            p.parse_postfix_predicate()
        })
    }

    fn parse_postfix_predicate(&mut self) -> R<Expr> {
        let e = self.parse_comparison()?;
        // ISO `<labeled predicate>` COLON form: `n:Label` on a bare element variable
        // reference (opengql:2078 — `elementVariableReference COLON labelExpression`).
        // Desugars to the same `IsLabeled` node as `IS LABELED`, reusing the pattern
        // parser's label-expression grammar (so `n:A|B`, `n:A&B`, `n:!A` all work).
        if self.check(Tt::Colon) && matches!(e, Expr::Var(_)) {
            self.advance();
            return Ok(Expr::IsLabeled {
                expr: Box::new(e),
                label: self.parse_label_expr()?,
                negated: false,
            });
        }
        if self.check_kw("is") {
            self.advance();
            let negated = if self.check_kw("not") {
                self.advance();
                true
            } else {
                false
            };
            if self.check_kw("null") {
                self.advance();
                return Ok(Expr::IsNull {
                    expr: Box::new(e),
                    negated,
                });
            }
            if self.check_kw("true") {
                self.advance();
                return Ok(Expr::IsTruth {
                    expr: Box::new(e),
                    truth: Some(true),
                    negated,
                });
            }
            if self.check_kw("false") {
                self.advance();
                return Ok(Expr::IsTruth {
                    expr: Box::new(e),
                    truth: Some(false),
                    negated,
                });
            }
            if self.check_kw("unknown") {
                self.advance();
                return Ok(Expr::IsTruth {
                    expr: Box::new(e),
                    truth: None,
                    negated,
                });
            }
            if self.check_soft("labeled") {
                self.advance();
                return Ok(Expr::IsLabeled {
                    expr: Box::new(e),
                    label: self.parse_label_expr()?,
                    negated,
                });
            }
            // `IS [NOT] TYPED <value type> [NOT NULL]` — the ISO value-type predicate.
            if self.check_soft("typed") || self.check_kw("typed") {
                self.advance();
                let ty = self.parse_value_type_test()?;
                // Optional `NOT NULL` type modifier (distinct from the `IS NOT`
                // predicate negation): excludes null from the type's value set.
                let not_null = if self.check_kw("not") {
                    self.advance();
                    self.expect_kw("null")?;
                    true
                } else {
                    false
                };
                return Ok(Expr::IsTyped {
                    expr: Box::new(e),
                    ty,
                    not_null,
                    negated,
                });
            }
            // `<edge> IS [NOT] DIRECTED`.
            if self.check_soft("directed") || self.check_kw("directed") {
                self.advance();
                return Ok(Expr::GraphPred {
                    kind: GraphPredKind::Directed,
                    args: vec![e],
                    negated,
                });
            }
            // `<node> IS [NOT] SOURCE OF <edge>` / `DESTINATION OF <edge>`.
            if self.check_soft("source") || self.check_kw("source") {
                self.advance();
                self.expect_of()?;
                let edge = self.parse_primary()?;
                return Ok(Expr::GraphPred {
                    kind: GraphPredKind::SourceOf,
                    args: vec![e, edge],
                    negated,
                });
            }
            if self.check_soft("destination") || self.check_kw("destination") {
                self.advance();
                self.expect_of()?;
                let edge = self.parse_primary()?;
                return Ok(Expr::GraphPred {
                    kind: GraphPredKind::DestOf,
                    args: vec![e, edge],
                    negated,
                });
            }
            return err(
                "Expected NULL, TRUE, FALSE, UNKNOWN, LABELED, TYPED, DIRECTED, or \
                 SOURCE/DESTINATION OF after IS",
                self.peek().pos,
            );
        }
        if self.in_operator_enabled && self.check_kw("in") {
            self.advance();
            return Ok(Expr::In {
                expr: Box::new(e),
                list: Box::new(self.parse_unary()?),
                negated: false,
            });
        }
        if self.in_operator_enabled && self.check_kw("not") {
            self.advance();
            self.expect_kw("in")?;
            return Ok(Expr::In {
                expr: Box::new(e),
                list: Box::new(self.parse_unary()?),
                negated: true,
            });
        }
        // ISO string-matching predicates. `contains`/`starts`/`ends` are
        // non-reserved words (arrive as idents); they desugar to the equivalent
        // BOOL functions. `STARTS`/`ENDS` require a following `WITH`.
        if self.check_soft("contains") {
            self.advance();
            return Ok(Expr::Func {
                name: "contains".into(),
                args: vec![e, self.parse_concat()?],
                distinct: false,
                star: false,
            });
        }
        if self.check_soft("starts") {
            self.advance();
            self.expect_kw("with")?;
            return Ok(Expr::Func {
                name: "starts_with".into(),
                args: vec![e, self.parse_concat()?],
                distinct: false,
                star: false,
            });
        }
        if self.check_soft("ends") {
            self.advance();
            self.expect_kw("with")?;
            return Ok(Expr::Func {
                name: "ends_with".into(),
                args: vec![e, self.parse_concat()?],
                distinct: false,
                star: false,
            });
        }
        Ok(e)
    }

    fn parse_comparison(&mut self) -> R<Expr> {
        let left = self.parse_concat()?;
        let op = match self.peek().tt {
            Tt::Eq => Some(CompareOp::Eq),
            Tt::Neq => Some(CompareOp::Ne),
            Tt::Lt => Some(CompareOp::Lt),
            Tt::Gt => Some(CompareOp::Gt),
            Tt::Lte => Some(CompareOp::Le),
            Tt::Gte => Some(CompareOp::Ge),
            _ => None,
        };
        if let Some(op) = op {
            self.advance();
            return Ok(Expr::Compare {
                op,
                left: Box::new(left),
                right: Box::new(self.parse_concat()?),
            });
        }
        Ok(left)
    }

    fn parse_concat(&mut self) -> R<Expr> {
        let first = self.parse_additive()?;
        if !self.check(Tt::Concat) {
            return Ok(first);
        }
        let mut items = vec![first];
        let mut chain = 0;
        while self.check(Tt::Concat) {
            chain += 1;
            self.chain_limit(chain)?;
            self.advance();
            items.push(self.parse_additive()?);
        }
        Ok(Expr::Concat(items))
    }

    fn parse_additive(&mut self) -> R<Expr> {
        let head = self.parse_multiplicative()?;
        let mut tail: Vec<(ArithOp, Expr)> = Vec::new();
        let mut chain = 0;
        loop {
            let op = match self.peek().tt {
                Tt::Plus => ArithOp::Add,
                Tt::Dash => ArithOp::Sub,
                _ => break,
            };
            chain += 1;
            self.chain_limit(chain)?;
            self.advance();
            tail.push((op, self.parse_multiplicative()?));
        }
        if tail.is_empty() {
            return Ok(head);
        }
        Ok(Expr::Arith {
            head: Box::new(head),
            tail,
        })
    }

    fn parse_multiplicative(&mut self) -> R<Expr> {
        let head = self.parse_unary()?;
        let mut tail: Vec<(ArithOp, Expr)> = Vec::new();
        let mut chain = 0;
        loop {
            let op = match self.peek().tt {
                Tt::Star => ArithOp::Mul,
                Tt::Slash => ArithOp::Div,
                Tt::Percent => ArithOp::Mod,
                _ => break,
            };
            chain += 1;
            self.chain_limit(chain)?;
            self.advance();
            tail.push((op, self.parse_unary()?));
        }
        if tail.is_empty() {
            return Ok(head);
        }
        Ok(Expr::Arith {
            head: Box::new(head),
            tail,
        })
    }

    fn parse_unary(&mut self) -> R<Expr> {
        self.descend(|p| {
            if p.check(Tt::Dash) {
                p.advance();
                return Ok(Expr::Neg(Box::new(p.parse_unary()?)));
            }
            if p.check(Tt::Plus) {
                p.advance();
                return p.parse_unary();
            }
            // ISO `!` unary-not — a TIGHT unary operator (`expressionAtom` level),
            // so it binds harder than the loose `NOT` keyword. Reuses `Not`.
            if p.check(Tt::Bang) {
                p.advance();
                return Ok(Expr::Not(Box::new(p.parse_unary()?)));
            }
            // Postfix subscript `base[index]` and property access `base.key`,
            // chained left to right (`a[0].amount`, `head(rels).x`, `a.b[0].c`). A
            // leading `[` is still a list literal, and a bare `variable.key` is
            // parsed as `Prop` in `parse_primary`; this loop only fires *after* a
            // primary, extending it with further `[…]`/`.key` steps.
            let mut e = p.parse_primary()?;
            loop {
                if p.check(Tt::LBracket) {
                    p.advance();
                    let index = p.parse_expr()?;
                    p.expect(Tt::RBracket, "']' to close a list subscript")?;
                    e = Expr::Index {
                        base: Box::new(e),
                        index: Box::new(index),
                    };
                } else if p.check(Tt::Dot) {
                    p.advance();
                    let key = p.bind_name("a property name")?;
                    e = Expr::Field {
                        base: Box::new(e),
                        key,
                    };
                } else {
                    break;
                }
            }
            Ok(e)
        })
    }

    /// The braced subquery inside `EXISTS { … }` / `COUNT { … }`. Accepts one or
    /// more `MATCH` blocks (ISO GQL: `EXISTS { MATCH (a) [WHERE …] MATCH (b) …}`).
    /// Multiple `MATCH`es are a conjunction, so they flatten to one pattern list +
    /// the AND of their `WHERE`s — the existing single-block eval handles it.
    fn parse_braced_subquery(&mut self) -> R<(Vec<PathPattern>, Option<Expr>)> {
        self.expect(Tt::LBrace, "'{'")?;
        let mut patterns = Vec::new();
        let mut wheres: Vec<Expr> = Vec::new();
        loop {
            // A leading `MATCH` is optional on the first block, required to start
            // each subsequent one.
            if self.check_kw("match") {
                self.advance();
            }
            patterns.push(self.parse_path_pattern()?);
            while self.check(Tt::Comma) {
                self.advance();
                patterns.push(self.parse_path_pattern()?);
            }
            if self.check_kw("where") {
                self.advance();
                wheres.push(self.parse_expr()?);
            }
            if !self.check_kw("match") {
                break;
            }
        }
        self.expect(Tt::RBrace, "'}'")?;
        let where_ = match wheres.len() {
            0 => None,
            1 => wheres.pop(),
            _ => Some(Expr::And(wheres)),
        };
        Ok((patterns, where_))
    }

    /// `VALUE { [MATCH p1, … [WHERE pred]] RETURN <expr> }` — a scalar subquery.
    /// The `VALUE` token has been consumed; parse the braced query specification.
    /// v1 scope: an optional single MATCH block (comma-joined patterns + WHERE)
    /// then a single-expression RETURN. The `{` is required.
    fn parse_value_subquery(&mut self) -> R<Expr> {
        self.expect(Tt::LBrace, "'{' after VALUE")?;
        let mut patterns = Vec::new();
        let mut where_ = None;
        // Patterns are optional (`VALUE { RETURN 1 }` is a constant); their
        // presence is signalled by anything other than a leading RETURN.
        if !self.check_kw("return") {
            if self.check_kw("match") {
                self.advance();
            }
            patterns.push(self.parse_path_pattern()?);
            while self.check(Tt::Comma) {
                self.advance();
                patterns.push(self.parse_path_pattern()?);
            }
            if self.check_kw("where") {
                self.advance();
                where_ = Some(Box::new(self.parse_expr()?));
            }
        }
        self.expect_kw("return")?;
        let ret = Box::new(self.parse_expr()?);
        self.expect(Tt::RBrace, "'}' to close VALUE")?;
        Ok(Expr::ValueSubquery {
            patterns,
            where_,
            ret,
        })
    }

    fn parse_call_args(&mut self) -> R<(Vec<Expr>, bool, bool)> {
        self.expect(Tt::LParen, "'(' to open a function call")?;
        let mut star = false;
        let mut distinct = false;
        let mut args = Vec::new();
        if self.check(Tt::Star) {
            self.advance();
            star = true;
        } else if !self.check(Tt::RParen) {
            if self.check_kw("distinct") {
                self.advance();
                distinct = true;
            }
            args.push(self.parse_expr()?);
            while self.check(Tt::Comma) {
                self.advance();
                args.push(self.parse_expr()?);
            }
        }
        self.expect(Tt::RParen, "')' to close a function call")?;
        Ok((args, distinct, star))
    }

    /// `TRIM([[LEADING|TRAILING|BOTH] [char] FROM] src)` — the SQL trim form.
    /// Desugars to `ltrim`/`rtrim`/`trim` (BOTH), each taking the optional trim
    /// character set. The leading `trim` ident is consumed; current token is `(`.
    fn parse_trim(&mut self) -> R<Expr> {
        self.expect(Tt::LParen, "'(' after TRIM")?;
        let soft = |p: &mut Self, w: &str| p.check_kw(w) || p.check_soft(w);
        let fn_name = if soft(self, "leading") {
            self.advance();
            "ltrim"
        } else if soft(self, "trailing") {
            self.advance();
            "rtrim"
        } else if soft(self, "both") {
            self.advance();
            "trim"
        } else {
            "trim" // BOTH is the default
        };
        // `[ [char] FROM ] src`.
        let (src, char_arg) = if soft(self, "from") {
            self.advance();
            (self.parse_expr()?, None)
        } else {
            let e1 = self.parse_expr()?;
            if soft(self, "from") {
                self.advance();
                (self.parse_expr()?, Some(e1)) // e1 was the trim character
            } else {
                (e1, None) // simple `TRIM(src)`
            }
        };
        self.expect(Tt::RParen, "')' to close TRIM")?;
        let mut args = vec![src];
        args.extend(char_arg);
        Ok(Expr::Func {
            name: fn_name.to_string(),
            args,
            distinct: false,
            star: false,
        })
    }

    /// `ALL_DIFFERENT(a, b, …)` / `SAME(a, b, …)` — parse the ≥2-element operand
    /// list (the leading name is consumed; the current token is `(`).
    fn parse_element_identity(&mut self, kind: GraphPredKind) -> R<Expr> {
        let (args, _distinct, _star) = self.parse_call_args()?;
        if args.len() < 2 {
            return err(
                "ALL_DIFFERENT/SAME require at least two element arguments",
                self.peek().pos,
            );
        }
        Ok(Expr::GraphPred {
            kind,
            args,
            negated: false,
        })
    }

    /// Consume the `OF` keyword (in `IS SOURCE/DESTINATION OF`).
    fn expect_of(&mut self) -> R<()> {
        if self.check_kw("of") || self.check_soft("of") {
            self.advance();
            Ok(())
        } else {
            err("expected OF after SOURCE/DESTINATION", self.peek().pos)
        }
    }

    /// Read a type-name token, joining the two-word `LOCAL`/`ZONED` temporal
    /// forms (`LOCAL DATETIME` → `local_datetime`). Shared by `CAST` and the
    /// `IS TYPED` value-type predicate.
    /// Parse a `<value type>` for the `IS TYPED` predicate: `[ANY] RECORD [{…}]`
    /// (open or closed record), or a scalar type name. Recursive — a record field
    /// is itself a `<value type>`.
    fn parse_value_type_test(&mut self) -> R<TypeTest> {
        let is_record_word = |p: &Self| p.check_soft("record") || p.check_kw("record");
        if is_record_word(self) {
            self.advance();
            return self.parse_record_type_body();
        }
        // `ANY RECORD` → open record; a bare `ANY` → the any-type test.
        if self.check_kw("any") || self.check_soft("any") {
            self.advance();
            if is_record_word(self) {
                self.advance();
                return self.parse_record_type_body();
            }
            return Ok(TypeTest::Scalar("any".to_string()));
        }
        let type_name = self.read_type_name("in IS TYPED")?;
        match type_test_category(&type_name) {
            Some(c) => Ok(TypeTest::Scalar(c.to_string())),
            None => err(
                format!("unsupported type '{type_name}' in IS TYPED"),
                self.peek().pos,
            ),
        }
    }

    /// `RECORD` already consumed → parse an optional `{ field ::type [NOT NULL], … }`.
    /// No brace = the OPEN record (`ANY RECORD` / bare `RECORD`).
    fn parse_record_type_body(&mut self) -> R<TypeTest> {
        if !self.check(Tt::LBrace) {
            return Ok(TypeTest::AnyRecord);
        }
        self.advance(); // {
        let mut fields: Vec<(String, TypeTest, bool)> = Vec::new();
        if !self.check(Tt::RBrace) {
            loop {
                let name = self.bind_name("a record field name")?;
                self.expect(Tt::Colon, "':' or '::' after a record field name")?;
                if self.check(Tt::Colon) {
                    self.advance(); // optional second colon (`::`)
                }
                let ty = self.parse_value_type_test()?;
                let not_null = if self.check_kw("not") {
                    self.advance();
                    self.expect_kw("null")?;
                    true
                } else {
                    false
                };
                // Sorted insert, duplicate last-wins — canonical, so a value's
                // (canonical, sorted) map keys align field-for-field at eval.
                match fields.binary_search_by(|(k, _, _)| k.as_str().cmp(name.as_str())) {
                    Ok(i) => fields[i] = (name, ty, not_null),
                    Err(i) => fields.insert(i, (name, ty, not_null)),
                }
                if self.check(Tt::Comma) {
                    self.advance();
                    continue;
                }
                break;
            }
        }
        self.expect(Tt::RBrace, "'}' to close a record type")?;
        Ok(TypeTest::Record(fields))
    }

    fn read_type_name(&mut self, ctx: &str) -> R<String> {
        let tok = self.peek().clone();
        if tok.tt != Tt::Ident && tok.tt != Tt::Keyword {
            return err(
                format!("expected a type name {ctx}, got '{}'", tok.value),
                tok.pos,
            );
        }
        self.advance();
        let mut type_name = tok.value.clone();
        let lead = type_name.to_ascii_lowercase();
        if lead == "local" || lead == "zoned" {
            let next = self.peek().clone();
            if (next.tt == Tt::Ident || next.tt == Tt::Keyword)
                && matches!(
                    next.value.to_ascii_lowercase().as_str(),
                    "datetime" | "time"
                )
            {
                self.advance();
                type_name = format!("{lead}_{}", next.value.to_ascii_lowercase());
            }
        }
        Ok(type_name)
    }

    /// `CAST(value AS type)` — the leading `cast` ident is already consumed and
    /// the current token is `(`. Desugars to the conversion function matching
    /// the target type (`to_integer`/`to_float`/`to_string`/`to_boolean`/
    /// `to_list`). An unrepresentable target type (temporal/bytes/record/…) is a
    /// loud syntax error, never a silent NULL.
    fn parse_cast(&mut self) -> R<Expr> {
        self.expect(Tt::LParen, "'(' after CAST")?;
        let value = self.parse_expr()?;
        self.expect_kw("as")?;
        let type_tok = self.peek().clone();
        if type_tok.tt != Tt::Ident && type_tok.tt != Tt::Keyword {
            return err(
                format!("expected a type name in CAST, got '{}'", type_tok.value),
                type_tok.pos,
            );
        }
        self.advance();
        // Two-word temporal type names: `LOCAL DATETIME` / `LOCAL TIME` /
        // `ZONED TIME` / `ZONED DATETIME`. The compact `LOCAL_DATETIME` keyword
        // forms are single tokens and fall through unchanged. Mirrors TS `parseCast`.
        let mut type_name = type_tok.value.clone();
        let lead = type_name.to_ascii_lowercase();
        if lead == "local" || lead == "zoned" {
            let next = self.peek().clone();
            if next.tt == Tt::Ident || next.tt == Tt::Keyword {
                let word = next.value.to_ascii_lowercase();
                if word == "datetime" || word == "time" {
                    self.advance();
                    type_name = format!("{lead}_{word}");
                }
            }
        }
        let fn_name = match cast_target_fn(&type_name) {
            Some(f) => f,
            None => {
                return err(
                    format!("CAST to unsupported type '{type_name}'"),
                    type_tok.pos,
                )
            }
        };
        self.expect(Tt::RParen, "')' to close CAST")?;
        Ok(Expr::Func {
            name: fn_name.into(),
            args: vec![value],
            distinct: false,
            star: false,
        })
    }

    fn parse_case(&mut self) -> R<Expr> {
        self.expect_kw("case")?;
        let subject = if self.check_kw("when") {
            None
        } else {
            Some(Box::new(self.parse_expr()?))
        };
        let mut whens = Vec::new();
        while self.check_kw("when") {
            self.advance();
            let when = self.parse_expr()?;
            self.expect_kw("then")?;
            whens.push((when, self.parse_expr()?));
        }
        if whens.is_empty() {
            return err("CASE requires at least one WHEN ... THEN", self.peek().pos);
        }
        let else_ = if self.check_kw("else") {
            self.advance();
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };
        self.expect_kw("end")?;
        Ok(Expr::Case {
            subject,
            whens,
            else_,
        })
    }

    /// Is the current token a temporal type keyword (`date`/`datetime`/
    /// `timestamp`/`duration`, non-delimited) immediately followed by a string?
    fn temporal_literal_ahead(&self) -> bool {
        let t = self.peek();
        if t.delimited {
            return false;
        }
        let is_kw = matches!(
            t.value.to_ascii_lowercase().as_str(),
            "date" | "datetime" | "timestamp" | "duration"
        );
        is_kw && matches!(self.tokens.get(self.pos + 1), Some(n) if n.tt == Tt::Str)
    }

    fn parse_temporal_literal(&mut self) -> R<Expr> {
        let kw = self.advance();
        let str_tok = self.advance();
        let tag = match kw.value.to_ascii_lowercase().as_str() {
            "date" => "date",
            "datetime" | "timestamp" => "datetime",
            "duration" => "duration",
            _ => unreachable!("guarded by temporal_literal_ahead"),
        };
        match crate::temporal::Temporal::parse(tag, &str_tok.value) {
            Ok(t) => Ok(Expr::Lit(Lit::Temporal(t))),
            Err(e) => err(
                format!("invalid {} literal: {e}", kw.value.to_uppercase()),
                str_tok.pos,
            ),
        }
    }

    /// A bare now-function keyword: `Some(true)` = `current_date` (a DATE),
    /// `Some(false)` = `current_timestamp`/`local_timestamp` (a LOCAL DATETIME —
    /// ISO's zoned `CURRENT_TIMESTAMP` is treated as local here, we're zone-less).
    fn now_function(&self) -> Option<bool> {
        let t = self.peek();
        if t.tt != Tt::Ident || t.delimited {
            return None;
        }
        match t.value.to_ascii_lowercase().as_str() {
            "current_date" => Some(true),
            "current_timestamp" | "local_timestamp" => Some(false),
            _ => None,
        }
    }

    /// A primary is a grouping boundary: any sub-expression it parses (parens,
    /// list, call args, subscript, CASE, subquery bodies) is a fresh value
    /// expression, so re-enable the `IN` operator for its extent. This restores
    /// the default inside `(…)` even while a `LET` binding's top level suppresses
    /// a bare `IN` — so `LET x = (a IN [1]) IN body END` parses as intended.
    fn parse_primary(&mut self) -> R<Expr> {
        let saved = self.in_operator_enabled;
        self.in_operator_enabled = true;
        let r = self.parse_primary_inner();
        self.in_operator_enabled = saved;
        r
    }

    fn parse_primary_inner(&mut self) -> R<Expr> {
        let t = self.peek().clone();
        match t.tt {
            Tt::Number => {
                self.advance();
                Ok(Expr::Lit(Lit::Num(t.num.unwrap())))
            }
            Tt::Str => {
                self.advance();
                Ok(Expr::Lit(Lit::Str(t.value)))
            }
            // ISO typed temporal literal: `DATE '2020-01-01'`, `DATETIME '…'`,
            // `TIMESTAMP '…'`, `DURATION 'P…'` — a soft-keyword ident directly
            // before a string. (The `date(…)` constructor-function form falls
            // through to the normal function-call path.)
            Tt::Ident if self.temporal_literal_ahead() => self.parse_temporal_literal(),
            // Bare now-functions `current_date` / `current_timestamp` /
            // `local_timestamp` desugar to a reserved `$__now` DATETIME param the
            // host supplies — the engine stays pure (never reads the clock), which
            // is what keeps the two engines byte-identical and deterministic.
            // `current_date` truncates via `date(...)`; the datetime forms wrap in
            // `local_datetime(...)` so the result is DATETIME-kind regardless of
            // what kind `$__now` was supplied as (a DATE `$__now` coerces to
            // midnight rather than leaking a DATE out of `current_timestamp`).
            Tt::Ident if self.now_function().is_some() => {
                let is_date = self.now_function() == Some(true);
                self.advance();
                if self.check(Tt::LParen) {
                    self.advance();
                    self.expect(Tt::RParen, "')' to close a now-function")?;
                }
                let now = Expr::Param("__now".to_string());
                Ok(Expr::Func {
                    name: if is_date { "date" } else { "local_datetime" }.to_string(),
                    args: vec![now],
                    distinct: false,
                    star: false,
                })
            }
            Tt::Param => {
                self.advance();
                Ok(Expr::Param(t.value))
            }
            Tt::Keyword if t.value == "true" || t.value == "false" => {
                self.advance();
                Ok(Expr::Lit(Lit::Bool(t.value == "true")))
            }
            Tt::Keyword if t.value == "null" => {
                self.advance();
                Ok(Expr::Lit(Lit::Null))
            }
            Tt::Keyword if t.value == "case" => self.parse_case(),
            // ISO `<let value expression>`: `LET x = e, … IN <body> END`. In
            // expression position `LET` is always the let-value-expression (the
            // LET *statement* only appears at clause position).
            Tt::Keyword if t.value == "let" => self.parse_let_in(),
            Tt::Keyword if t.value == "exists" => {
                self.advance();
                let (patterns, where_) = self.parse_braced_subquery()?;
                Ok(Expr::Exists {
                    patterns,
                    where_: where_.map(Box::new),
                })
            }
            Tt::Keyword if t.value == "count" => {
                self.advance();
                if self.check(Tt::LBrace) {
                    let (patterns, where_) = self.parse_braced_subquery()?;
                    Ok(Expr::CountSubquery {
                        patterns,
                        where_: where_.map(Box::new),
                    })
                } else {
                    let (args, distinct, star) = self.parse_call_args()?;
                    Ok(Expr::Func {
                        name: "count".into(),
                        args,
                        distinct,
                        star,
                    })
                }
            }
            Tt::LParen => {
                self.advance();
                let inner = self.parse_expr()?;
                self.expect(Tt::RParen, "')'")?;
                Ok(inner)
            }
            Tt::LBracket => {
                self.advance();
                let mut items = Vec::new();
                if !self.check(Tt::RBracket) {
                    loop {
                        items.push(self.parse_expr()?);
                        if self.check(Tt::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(Tt::RBracket, "']' to close a list")?;
                Ok(Expr::List(items))
            }
            // ISO `<record constructor>`: `{ field: expr, … }`. A bare `{` in
            // expression position is unambiguous — VALUE/EXISTS/COUNT lead with a
            // keyword, and property maps / subqueries live in their own positions.
            Tt::LBrace => self.parse_record(),
            Tt::Ident => {
                // ISO `VALUE { … RETURN e }` — scalar (single-value) subquery.
                // `value` isn't a reserved word, so disambiguate on the following
                // `{` (an identifier is never followed by a brace). A backtick-
                // delimited `` `value` `` stays a plain identifier.
                if !t.delimited
                    && t.value.eq_ignore_ascii_case("value")
                    && self.peek2().tt == Tt::LBrace
                {
                    self.advance();
                    return self.parse_value_subquery();
                }
                self.advance();
                // Function call: the name may be a reserved word (UPPER, SUM, ABS).
                if self.check(Tt::LParen) {
                    // `CAST(value AS type)` is a keyword-shaped call; desugar it
                    // to the matching conversion function (to_integer/…).
                    if !t.delimited && t.value.eq_ignore_ascii_case("cast") {
                        return self.parse_cast();
                    }
                    // `PROPERTY_EXISTS(n, key)` — presence predicate; the second
                    // arg is a bare property NAME, not an expression, so it can't
                    // ride the generic function-call path.
                    if !t.delimited && t.value.eq_ignore_ascii_case("property_exists") {
                        self.expect(Tt::LParen, "'('")?;
                        let variable = self.bind_name("an element variable")?;
                        self.expect(Tt::Comma, "','")?;
                        let key = self.bind_name("a property name")?;
                        self.expect(Tt::RParen, "')'")?;
                        return Ok(Expr::PropertyExists { variable, key });
                    }
                    // `ALL_DIFFERENT(a, b, …)` / `SAME(a, b, …)` — element-identity
                    // predicates over ≥2 element operands.
                    if !t.delimited && t.value.eq_ignore_ascii_case("all_different") {
                        return self.parse_element_identity(GraphPredKind::AllDifferent);
                    }
                    if !t.delimited && t.value.eq_ignore_ascii_case("same") {
                        return self.parse_element_identity(GraphPredKind::Same);
                    }
                    // `TRIM([[spec] [char] FROM] src)` — the SQL trim form; desugars
                    // to trim/ltrim/rtrim (the comma-char forms use btrim/ltrim/rtrim).
                    if !t.delimited && t.value.eq_ignore_ascii_case("trim") {
                        return self.parse_trim();
                    }
                    let (args, distinct, star) = self.parse_call_args()?;
                    return Ok(Expr::Func {
                        name: t.value.to_ascii_lowercase(),
                        args,
                        distinct,
                        star,
                    });
                }
                if !t.delimited && is_reserved(&t.value) {
                    return Err(reserved_error(&t, "a variable"));
                }
                if self.check(Tt::Dot) {
                    self.advance();
                    let key = self.bind_name("a property name")?;
                    return Ok(Expr::Prop {
                        variable: t.value,
                        key,
                    });
                }
                Ok(Expr::Var(t.value))
            }
            _ => {
                let got = if t.value.is_empty() {
                    format!("{:?}", t.tt)
                } else {
                    t.value
                };
                err(format!("Unexpected '{got}' in expression"), t.pos)
            }
        }
    }

    // --- return / projection ----------------------------------------------

    fn parse_return_item(&mut self) -> R<ReturnItem> {
        let expr = self.parse_expr()?;
        let alias = if self.check_kw("as") {
            self.advance();
            Some(self.bind_name("an alias name")?)
        } else {
            None
        };
        Ok(ReturnItem { expr, alias })
    }

    fn parse_sort_item(&mut self) -> R<SortItem> {
        let expr = self.parse_expr()?;
        let mut descending = false;
        if self.check_kw("desc") || self.check_kw("descending") {
            self.advance();
            descending = true;
        } else if self.check_kw("asc") || self.check_kw("ascending") {
            self.advance();
        }
        let mut nulls_first = None;
        if self.check_kw("nulls") {
            self.advance();
            if self.check_soft("first") {
                self.advance();
                nulls_first = Some(true);
            } else if self.check_soft("last") {
                self.advance();
                nulls_first = Some(false);
            } else {
                return err("Expected FIRST or LAST after NULLS", self.peek().pos);
            }
        }
        Ok(SortItem {
            expr,
            descending,
            nulls_first,
        })
    }

    fn parse_projection(&mut self) -> R<Projection> {
        let distinct = if self.check_kw("distinct") {
            self.advance();
            true
        } else {
            false
        };
        let mut star = false;
        let mut items = Vec::new();
        if self.check(Tt::Star) {
            self.advance();
            star = true;
        } else {
            items.push(self.parse_return_item()?);
            while self.check(Tt::Comma) {
                self.advance();
                items.push(self.parse_return_item()?);
            }
        }
        // ISO `GROUP BY <grouping element list>` — comes after the items, before
        // ORDER BY. Drives the grouping (see `Projection::group_by`).
        let mut group_by = Vec::new();
        if self.check_kw("group") || self.check_soft("group") {
            self.advance();
            self.expect_kw("by")?;
            group_by.push(self.parse_expr()?);
            while self.check(Tt::Comma) {
                self.advance();
                group_by.push(self.parse_expr()?);
            }
        }
        let mut order_by = Vec::new();
        if self.check_kw("order") {
            self.advance();
            self.expect_kw("by")?;
            order_by.push(self.parse_sort_item()?);
            while self.check(Tt::Comma) {
                self.advance();
                order_by.push(self.parse_sort_item()?);
            }
        }
        let mut skip = None;
        if self.check_kw("skip") || self.check_kw("offset") {
            // OFFSET is the ISO spelling and accepts a dynamic `$param`; SKIP is
            // the Cypher synonym and stays literal-only.
            let allow_param = self.check_kw("offset");
            self.advance();
            skip = Some(
                self.expect_count_bound("a non-negative integer after SKIP/OFFSET", allow_param)?,
            );
        }
        let mut limit = None;
        if self.check_kw("limit") {
            self.advance();
            limit = Some(self.expect_count_bound("a non-negative integer after LIMIT", true)?);
        }
        Ok(Projection {
            star,
            items,
            distinct,
            group_by,
            having: None,
            order_by,
            skip,
            limit,
        })
    }

    /// ISO `<select statement>`: the SQL-shaped query form —
    /// `SELECT [DISTINCT] {* | items} FROM [<graph>] MATCH pattern[, …]
    ///  [WHERE c] [GROUP BY g] [HAVING h] [ORDER BY o] [OFFSET n] [LIMIT n]`.
    /// A parser-only desugar to the linear pipeline: an optional `MATCH` (from the
    /// `FROM MATCH` body, carrying the WHERE) plus a terminal `RETURN` projection
    /// that carries GROUP BY / HAVING / ORDER BY / paging. HAVING is the one piece
    /// the linear `RETURN` can't express, so it rides the projection. `SELECT`,
    /// `FROM`, `HAVING`, `GROUP` are reserved idents (soft keywords).
    fn parse_select(&mut self) -> R<Vec<Clause>> {
        self.expect_soft("select")?;
        let distinct = if self.check_kw("distinct") {
            self.advance();
            true
        } else {
            false
        };
        let mut star = false;
        let mut items = Vec::new();
        if self.check(Tt::Star) {
            self.advance();
            star = true;
        } else {
            items.push(self.parse_return_item()?);
            while self.check(Tt::Comma) {
                self.advance();
                items.push(self.parse_return_item()?);
            }
        }

        // `FROM [<graph>] MATCH pattern[, …] [WHERE c]`. The body is optional
        // (`SELECT 1 + 1` is a one-row constant); GROUP BY / HAVING require it.
        let mut match_clause = None;
        let mut where_ = None;
        if self.check_soft("from") {
            self.advance();
            // Optional graph expression. lenke is single-graph, so accept only a
            // reference to the current/home graph (consumed and ignored); a named
            // graph is rejected by falling through to the required MATCH.
            if !self.check_kw("match")
                && (self.check_soft("current_graph")
                    || self.check_soft("home_graph")
                    || self.check_soft("current_property_graph")
                    || self.check_soft("home_property_graph"))
            {
                self.advance();
            }
            self.expect_kw("match")?;
            let match_mode = self.parse_match_mode();
            let mut patterns = vec![self.parse_path_pattern()?];
            while self.check(Tt::Comma) {
                self.advance();
                patterns.push(self.parse_path_pattern()?);
            }
            if let Some(mode) = match_mode {
                for p in &mut patterns {
                    p.mode = mode;
                }
            }
            if self.check_kw("where") {
                self.advance();
                where_ = Some(self.parse_expr()?);
            }
            match_clause = Some(Clause::Match(MatchClause {
                optional: false,
                patterns,
                where_,
            }));
        }

        // `GROUP BY g` — before HAVING (SQL order), unlike RETURN's after-items.
        let mut group_by = Vec::new();
        if self.check_soft("group") || self.check_kw("group") {
            self.advance();
            self.expect_kw("by")?;
            group_by.push(self.parse_expr()?);
            while self.check(Tt::Comma) {
                self.advance();
                group_by.push(self.parse_expr()?);
            }
        }

        // `HAVING h` — the post-aggregation group filter (SELECT-statement only).
        let having = if self.check_soft("having") {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        let mut order_by = Vec::new();
        if self.check_kw("order") {
            self.advance();
            self.expect_kw("by")?;
            order_by.push(self.parse_sort_item()?);
            while self.check(Tt::Comma) {
                self.advance();
                order_by.push(self.parse_sort_item()?);
            }
        }
        let mut skip = None;
        if self.check_kw("skip") || self.check_kw("offset") {
            let allow_param = self.check_kw("offset");
            self.advance();
            skip = Some(
                self.expect_count_bound("a non-negative integer after SKIP/OFFSET", allow_param)?,
            );
        }
        let mut limit = None;
        if self.check_kw("limit") {
            self.advance();
            limit = Some(self.expect_count_bound("a non-negative integer after LIMIT", true)?);
        }

        let projection = Projection {
            star,
            items,
            distinct,
            group_by,
            having,
            order_by,
            skip,
            limit,
        };
        let mut clauses = Vec::new();
        if let Some(m) = match_clause {
            clauses.push(m);
        }
        clauses.push(Clause::Return(projection));
        Ok(clauses)
    }

    fn parse_with_clause(&mut self) -> R<WithClause> {
        self.expect_kw("with")?;
        let projection = self.parse_projection()?;
        let where_ = if self.check_kw("where") {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };
        Ok(WithClause { projection, where_ })
    }

    /// `FILTER [WHERE] <condition>` — ISO GQL §14.6. The `WHERE` is optional
    /// noise; the condition is a full boolean expression over the current scope.
    fn parse_filter_clause(&mut self) -> R<Expr> {
        self.expect_kw("filter")?;
        if self.check_kw("where") {
            self.advance();
        }
        self.parse_expr()
    }

    /// `LET x = <expr> [, y = <expr>]*` — ISO GQL §14.7. Comma-separated bindings,
    /// evaluated left-to-right (a later item may reference an earlier one).
    /// `LET x = e1, y = e2 IN <body> END` — the inline let-value expression.
    /// Each binding's RHS is parsed with the bare `IN` operator suppressed so the
    /// value expression ends at the structural `IN` (a parenthesized `IN`
    /// predicate still works — `parse_primary` re-enables it inside the parens).
    /// ISO `<record constructor>`: `{ field: expr [, field: expr]… }` (or `{}`).
    /// Field names are identifiers (a reserved word must be backtick-quoted, like
    /// any binding name). The `{` has NOT been consumed.
    fn parse_record(&mut self) -> R<Expr> {
        self.expect(Tt::LBrace, "'{' to open a record")?;
        let mut fields = Vec::new();
        if !self.check(Tt::RBrace) {
            loop {
                let key = self.bind_name("a record field name")?;
                self.expect(Tt::Colon, "':' after a record field name")?;
                let value = self.parse_expr()?;
                fields.push((key, value));
                if self.check(Tt::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(Tt::RBrace, "'}' to close a record")?;
        Ok(Expr::Record(fields))
    }

    fn parse_let_in(&mut self) -> R<Expr> {
        self.expect_kw("let")?;
        let mut bindings = Vec::new();
        loop {
            let var = self.bind_name("a LET variable")?;
            self.expect(Tt::Eq, "'=' after a LET variable")?;
            let saved = self.in_operator_enabled;
            self.in_operator_enabled = false;
            let expr = self.parse_expr();
            self.in_operator_enabled = saved;
            bindings.push(LetItem { var, expr: expr? });
            if self.check(Tt::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        self.expect_kw("in")?;
        let body = Box::new(self.parse_expr()?);
        self.expect_kw("end")?;
        Ok(Expr::LetIn { bindings, body })
    }

    fn parse_let_clause(&mut self) -> R<Vec<LetItem>> {
        self.expect_kw("let")?;
        let mut items = Vec::new();
        loop {
            let var = self.bind_name("a LET variable")?;
            self.expect(Tt::Eq, "'=' after a LET variable")?;
            let expr = self.parse_expr()?;
            items.push(LetItem { var, expr });
            if self.check(Tt::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        Ok(items)
    }

    /// `FOR <alias> IN <list> [WITH (ORDINALITY|OFFSET) <var>]` — ISO GQL list
    /// unwind. The list is parsed as a full expression (it may reference any
    /// prior binding); `IN` is consumed as a keyword up front, so it is not
    /// mistaken for the `IN` membership operator inside the list expression.
    fn parse_for_clause(&mut self) -> R<ForClause> {
        self.expect_kw("for")?;
        let alias = self.bind_name("a FOR variable")?;
        self.expect_kw("in")?;
        let list = self.parse_expr()?;
        // `WITH ORDINALITY|OFFSET <var>` is a FOR modifier ONLY when ORDINALITY
        // or OFFSET follows WITH; a bare WITH here begins the next clause and
        // must be left for the clause loop.
        let ordinal = if self.check_kw("with") && self.for_modifier_ahead() {
            self.advance(); // WITH
            let kind = if self.check_kw("offset") {
                self.advance();
                OrdKind::Offset
            } else {
                self.advance(); // ORDINALITY (a soft keyword — lexes as an ident)
                OrdKind::Ordinality
            };
            let var = self.bind_name("an ORDINALITY/OFFSET variable")?;
            Some(ForOrdinal { kind, var })
        } else {
            None
        };
        Ok(ForClause {
            alias,
            list,
            ordinal,
        })
    }

    /// Is the token after `WITH` an ORDINALITY/OFFSET modifier (vs the start of a
    /// new WITH clause)? `ORDINALITY` is a soft keyword (arrives as an ident).
    fn for_modifier_ahead(&self) -> bool {
        match self.tokens.get(self.pos + 1) {
            Some(t) => {
                (t.tt == Tt::Keyword && t.value == "offset")
                    || (t.tt == Tt::Ident && t.value.eq_ignore_ascii_case("ordinality"))
            }
            None => false,
        }
    }

    // --- write clauses -----------------------------------------------------

    fn parse_insert_clause(&mut self) -> R<Vec<PathPattern>> {
        self.expect_kw("insert")?;
        let mut patterns = vec![self.parse_path_pattern()?];
        while self.check(Tt::Comma) {
            self.advance();
            patterns.push(self.parse_path_pattern()?);
        }
        Ok(patterns)
    }

    fn parse_set_item(&mut self) -> R<SetItem> {
        let variable = self.bind_name("a variable")?;
        if self.check(Tt::Colon) || self.check_kw("is") {
            self.advance();
            return Ok(SetItem::Label {
                variable,
                label: self.bind_name("a label name")?,
            });
        }
        self.expect(Tt::Dot, "'.' or ':'")?;
        let key = self.bind_name("a property name")?;
        self.expect(Tt::Eq, "'='")?;
        Ok(SetItem::Prop {
            variable,
            key,
            value: self.parse_expr()?,
        })
    }

    fn parse_set_list(&mut self) -> R<Vec<SetItem>> {
        let mut items = vec![self.parse_set_item()?];
        while self.check(Tt::Comma) {
            self.advance();
            items.push(self.parse_set_item()?);
        }
        Ok(items)
    }

    fn parse_set_clause(&mut self) -> R<Vec<SetItem>> {
        self.expect_kw("set")?;
        self.parse_set_list()
    }

    // `_MERGE pattern [_ON_CREATE SET …] [_ON_UPDATE SET … [WHERE p] |
    // _ON_UPDATE_NOTHING]` — the lenke keyed-upsert extension (sigil-marked; see
    // docs/design/gql-extensions.md §2). Branches may appear in any order, each at
    // most once; an explicit _ON_UPDATE excludes _ON_UPDATE_NOTHING.
    fn parse_merge_clause(&mut self) -> R<MergeClause> {
        self.advance(); // _MERGE
        let pattern = self.parse_path_pattern()?;
        let mut on_create: Option<Vec<SetItem>> = None;
        let mut on_update: Option<MergeUpdate> = None;
        loop {
            if self.check_ext("_on_create") {
                self.advance();
                if on_create.is_some() {
                    return err(
                        "duplicate _ON_CREATE in _MERGE".to_string(),
                        self.peek().pos,
                    );
                }
                self.expect_kw("set")?;
                on_create = Some(self.parse_set_list()?);
            } else if self.check_ext("_on_update_nothing") {
                self.advance();
                if on_update.is_some() {
                    return err(
                        "conflicting update disposition in _MERGE".to_string(),
                        self.peek().pos,
                    );
                }
                on_update = Some(MergeUpdate::Nothing);
            } else if self.check_ext("_on_update") {
                self.advance();
                if on_update.is_some() {
                    return err(
                        "conflicting update disposition in _MERGE".to_string(),
                        self.peek().pos,
                    );
                }
                self.expect_kw("set")?;
                let items = self.parse_set_list()?;
                let where_ = if self.check_kw("where") {
                    self.advance();
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                on_update = Some(MergeUpdate::Set { items, where_ });
            } else {
                break;
            }
        }
        Ok(MergeClause {
            pattern,
            on_create,
            on_update,
        })
    }

    fn parse_remove_item(&mut self) -> R<RemoveItem> {
        let variable = self.bind_name("a variable")?;
        if self.check(Tt::Colon) || self.check_kw("is") {
            self.advance();
            return Ok(RemoveItem::Label {
                variable,
                label: self.bind_name("a label name")?,
            });
        }
        self.expect(Tt::Dot, "'.' or ':'")?;
        Ok(RemoveItem::Prop {
            variable,
            key: self.bind_name("a property name")?,
        })
    }

    fn parse_remove_clause(&mut self) -> R<Vec<RemoveItem>> {
        self.expect_kw("remove")?;
        let mut items = vec![self.parse_remove_item()?];
        while self.check(Tt::Comma) {
            self.advance();
            items.push(self.parse_remove_item()?);
        }
        Ok(items)
    }

    fn parse_delete_clause(&mut self) -> R<Clause> {
        let detach = if self.check_kw("detach") {
            self.advance();
            true
        } else {
            false
        };
        if !detach && self.check_kw("nodetach") {
            self.advance();
        }
        self.expect_kw("delete")?;
        let mut targets = vec![self.parse_expr()?];
        while self.check(Tt::Comma) {
            self.advance();
            targets.push(self.parse_expr()?);
        }
        Ok(Clause::Delete { detach, targets })
    }

    /// A bare (non-delimited) identifier matching `word` case-insensitively — the
    /// contextual match used for the transaction keywords (START/TRANSACTION/
    /// COMMIT/ROLLBACK/WORK/READ/ONLY/WRITE), which are recognized here at
    /// statement start rather than promoted to lexer keywords or reserved words, so
    /// they stay usable as ordinary identifiers everywhere else.
    fn is_word(&self, word: &str) -> bool {
        let t = self.peek();
        t.tt == Tt::Ident && !t.delimited && t.value.eq_ignore_ascii_case(word)
    }

    /// A statement is transaction control iff its first token is a bare `START`,
    /// `COMMIT`, or `ROLLBACK` identifier — a linear query can never begin with
    /// one, so there is no ambiguity, and a delimited `` `start` `` stays an ident.
    fn starts_tx_control(&self) -> bool {
        self.is_word("start") || self.is_word("commit") || self.is_word("rollback")
    }

    /// Parse a top-level statement: a transaction-control command when it starts
    /// with one, else a linear query (possibly joined by set operators).
    fn parse_statement(&mut self) -> R<Statement> {
        if self.starts_tx_control() {
            let tx = self.parse_tx_control()?;
            if !self.at_end() {
                let t = self.peek();
                let got = if t.value.is_empty() {
                    format!("{:?}", t.tt)
                } else {
                    t.value.clone()
                };
                return err(format!("Unexpected trailing input '{got}'"), t.pos);
            }
            return Ok(Statement::Tx(tx));
        }
        Ok(Statement::Query(self.parse_query()?))
    }

    /// `READ ONLY | READ WRITE` — the single optional access mode after
    /// `START TRANSACTION`. ISO also allows a comma-separated mode list; v1 takes
    /// one mode (a second mode / trailing comma falls to the top-level trailing-
    /// input check). `READ` not followed by `ONLY`/`WRITE` is a syntax error.
    fn parse_access_mode(&mut self) -> R<Option<AccessMode>> {
        if !self.is_word("read") {
            return Ok(None);
        }
        self.advance(); // READ
        if self.is_word("only") {
            self.advance();
            return Ok(Some(AccessMode::ReadOnly));
        }
        if self.is_word("write") {
            self.advance();
            return Ok(Some(AccessMode::ReadWrite));
        }
        let t = self.peek();
        err("Expected ONLY or WRITE after READ".to_string(), t.pos)
    }

    fn parse_tx_control(&mut self) -> R<TxControl> {
        let kw = self.advance().value.to_ascii_lowercase(); // start | commit | rollback
        if kw == "start" {
            if !self.is_word("transaction") {
                let t = self.peek();
                let got = if t.value.is_empty() {
                    format!("{:?}", t.tt)
                } else {
                    t.value.clone()
                };
                return err(
                    format!("Expected TRANSACTION after START, got '{got}'"),
                    t.pos,
                );
            }
            self.advance(); // TRANSACTION
            let access_mode = self.parse_access_mode()?;
            return Ok(TxControl {
                kind: TxKind::Start,
                access_mode,
            });
        }
        // COMMIT / ROLLBACK — optionally followed by the noise word WORK.
        if self.is_word("work") {
            self.advance();
        }
        Ok(TxControl {
            kind: if kw == "commit" {
                TxKind::Commit
            } else {
                TxKind::Rollback
            },
            access_mode: None,
        })
    }

    fn parse_linear_query(&mut self) -> R<LinearQuery> {
        let mut clauses = Vec::new();
        let mut done = false;
        while !done && !self.at_end() {
            if self.check_kw("return") {
                self.advance();
                clauses.push(Clause::Return(self.parse_projection()?));
                done = true;
            } else if self.check_soft("select") {
                // ISO `<select statement>` — a SQL-shaped terminal query that
                // desugars to MATCH + RETURN (carrying GROUP BY / HAVING / paging).
                clauses.extend(self.parse_select()?);
                done = true;
            } else if self.check_kw("finish") {
                self.advance();
                clauses.push(Clause::Finish);
                done = true;
            } else if self.check_kw("with") {
                clauses.push(Clause::With(self.parse_with_clause()?));
            } else if self.check_kw("let") {
                clauses.push(Clause::Let(self.parse_let_clause()?));
            } else if self.check_kw("filter") {
                clauses.push(Clause::Filter(self.parse_filter_clause()?));
            } else if self.check_kw("for") {
                clauses.push(Clause::For(self.parse_for_clause()?));
            } else if self.check_kw("call") || (self.check_kw("optional") && self.kw_after("call"))
            {
                clauses.push(self.parse_call_clause()?);
            } else if self.check_kw("match") || self.check_kw("optional") {
                clauses.push(Clause::Match(self.parse_match_clause()?));
            } else if self.check_kw("insert") {
                clauses.push(Clause::Insert(self.parse_insert_clause()?));
            } else if self.check_ext("_merge") {
                clauses.push(Clause::Merge(self.parse_merge_clause()?));
            } else if self.check_kw("set") {
                clauses.push(Clause::Set(self.parse_set_clause()?));
            } else if self.check_kw("remove") {
                clauses.push(Clause::Remove(self.parse_remove_clause()?));
            } else if self.check_kw("delete")
                || self.check_kw("detach")
                || self.check_kw("nodetach")
            {
                clauses.push(self.parse_delete_clause()?);
            } else {
                break;
            }
        }
        if clauses.is_empty() {
            let t = self.peek();
            let got = if t.value.is_empty() {
                format!("{:?}", t.tt)
            } else {
                t.value.clone()
            };
            return err(
                format!("Expected a clause (MATCH, INSERT, RETURN, …), got '{got}'"),
                t.pos,
            );
        }
        Ok(LinearQuery { clauses })
    }

    fn parse_query(&mut self) -> R<Query> {
        let query = self.parse_next_chain()?;
        if !self.at_end() {
            let t = self.peek();
            let got = if t.value.is_empty() {
                format!("{:?}", t.tt)
            } else {
                t.value.clone()
            };
            return err(format!("Unexpected trailing input '{got}'"), t.pos);
        }
        Ok(query)
    }

    /// ISO GQL `NEXT` linear-statement composition: `A NEXT [YIELD …] B [NEXT …]`
    /// pipes each statement's RETURN output forward as the next statement's driving
    /// table. Implemented as a rewrite — each pre-NEXT `RETURN` becomes a `WITH`
    /// (plus a second `WITH` for `YIELD`'s column select/rename), concatenated with
    /// the following statement into one linear query, reusing all WITH machinery.
    /// Only single-linear segments are supported (no set operators around NEXT).
    fn parse_next_chain(&mut self) -> R<Query> {
        let head = self.parse_set_op_query()?;
        if !self.check_kw("next") {
            return Ok(head);
        }
        let mut clauses = self.take_linear_for_next(head)?;
        while self.check_kw("next") {
            self.advance();
            // The prior statement's terminal RETURN becomes a WITH piping it forward.
            match clauses.pop() {
                Some(Clause::Return(proj)) => clauses.push(Clause::With(WithClause {
                    projection: proj,
                    where_: None,
                })),
                _ => return err("NEXT must follow a RETURN".to_string(), self.peek().pos),
            }
            // Optional `YIELD col [AS alias], …` → a second WITH selecting/renaming
            // the piped columns (each yielded name resolves against the prior scope).
            if self.check_kw("yield") {
                self.advance();
                let mut items = vec![self.next_yield_item()?];
                while self.check(Tt::Comma) {
                    self.advance();
                    items.push(self.next_yield_item()?);
                }
                clauses.push(Clause::With(WithClause {
                    projection: Projection {
                        star: false,
                        items,
                        distinct: false,
                        group_by: Vec::new(),
                        having: None,
                        order_by: Vec::new(),
                        skip: None,
                        limit: None,
                    },
                    where_: None,
                }));
            }
            let seg = self.parse_set_op_query()?;
            clauses.extend(self.take_linear_for_next(seg)?);
        }
        Ok(Query {
            parts: vec![LinearQuery { clauses }],
            ops: Vec::new(),
        })
    }

    /// A NEXT segment must be a single linear query (no set operators) for the
    /// RETURN→WITH rewrite. Returns its clauses, or errors on a set-op segment.
    fn take_linear_for_next(&mut self, q: Query) -> R<Vec<Clause>> {
        if !q.ops.is_empty() || q.parts.len() != 1 {
            return err(
                "NEXT does not support set operators (UNION/EXCEPT/INTERSECT) in a composed \
                 statement"
                    .to_string(),
                self.peek().pos,
            );
        }
        Ok(q.parts.into_iter().next().unwrap().clauses)
    }

    /// One `YIELD col [AS alias]` for NEXT → a `Var(col)` projection item aliased to
    /// `alias` (or `col`), so the piped column is selected (and optionally renamed).
    fn next_yield_item(&mut self) -> R<ReturnItem> {
        let name = self.bind_name("a YIELD column")?;
        let alias = if self.check_kw("as") {
            self.advance();
            Some(self.bind_name("a YIELD alias")?)
        } else {
            Some(name.clone())
        };
        Ok(ReturnItem {
            expr: Expr::Var(name),
            alias,
        })
    }

    /// Parse one or more linear parts joined by set operators, WITHOUT requiring
    /// end-of-input afterward — so it works both at top level (wrapped by
    /// `parse_query`, which adds the trailing-input check) and inside an inline
    /// `CALL (…) { … }` body (terminated by `}`).
    fn parse_set_op_query(&mut self) -> R<Query> {
        let mut parts = vec![self.parse_linear_query()?];
        let mut ops = Vec::new();
        loop {
            let op = if self.peek().tt == Tt::Keyword {
                match self.peek().value.as_str() {
                    "union" => Some(SetOpKind::Union),
                    "except" => Some(SetOpKind::Except),
                    "intersect" => Some(SetOpKind::Intersect),
                    _ => None,
                }
            } else {
                None
            };
            let Some(op) = op else { break };
            self.advance();
            let all = if self.check_kw("all") {
                self.advance();
                true
            } else {
                if self.check_kw("distinct") {
                    self.advance();
                }
                false
            };
            ops.push(SetOp { op, all });
            parts.push(self.parse_linear_query()?);
        }
        Ok(Query { parts, ops })
    }
}
