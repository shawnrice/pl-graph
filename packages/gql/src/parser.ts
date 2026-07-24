/**
 * Recursive-descent parser: token stream -> `Query` AST.
 *
 * Grammar (informal). This is ISO GQL, not Cypher: `--` is a line comment,
 * undirected edges use `~`, and labels are boolean expressions, not `:A:B`.
 *
 *   query      = match where? return
 *   match      = MATCH pathPattern (',' pathPattern)*
 *   pathPattern= node (rel node)*
 *   node       = '(' var? (labelIntro labelExpr)? ')'
 *   rel        = abbrevEdge | '-'|'<-'|'~'|'<~' relDetail '-'|'->'|'~'|'~>'
 *   abbrevEdge = '->' | '<-' | '<->' | '~' | '~>'
 *   relDetail  = '[' var? (labelIntro labelExpr)? ']'
 *   labelIntro = ':' | IS
 *   labelExpr  = labelAnd ('|' labelAnd)*       // disjunction
 *   labelAnd   = labelNot ('&' labelNot)*       // conjunction
 *   labelNot   = '!' labelNot | labelPrimary    // negation
 *   labelPrimary = '%' | label | '(' labelExpr ')'
 *   where      = WHERE expr
 *   return     = RETURN DISTINCT? item (',' item)* (LIMIT number)?
 *   item       = expr (AS ident)?
 *   expr       = orExpr
 *   orExpr     = andExpr (OR andExpr)*
 *   andExpr    = notExpr (AND notExpr)*
 *   notExpr    = NOT notExpr | comparison
 *   comparison = primary (compareOp primary)?
 *   primary    = ident ('.' ident)? | literal | '(' expr ')'
 *   comment    = '//' line | '--' line | block comment
 */

import { temporalParse } from '@lenke/core';

import type {
  ArithOp,
  Clause,
  CompareOp,
  CountValue,
  DeleteClause,
  Expr,
  InsertClause,
  LabelExpr,
  Dialect,
  LinearQuery,
  MatchClause,
  MergeClause,
  MergeUpdate,
  CallInlineClause,
  CallNamedClause,
  NodePattern,
  PathPattern,
  PathSelector,
  Projection,
  Query,
  YieldItem,
  PropertyConstraint,
  RelPattern,
  RemoveClause,
  RemoveItem,
  SetOp,
  ReturnClause,
  ReturnItem,
  Segment,
  SetClause,
  SetItem,
  SortItem,
  Statement,
  TxControl,
  WithClause,
  FilterClause,
  LetClause,
  ForClause,
} from './ast.js';
import { isTxControl } from './ast.js';
import { GqlSyntaxError, isReserved, type Token, type TokenType, tokenize } from './lexer.js';

// ISO GQL `CAST` target type name → the conversion function it desugars to.
// Integer/float/string/bool/list families are representable; anything else
// (temporal, bytes, record, …) has no home in this value model and returns
// null (a loud CAST error). Mirrors the Rust `cast_target_fn`.
const CAST_INT = new Set([
  'int',
  'integer',
  'int8',
  'int16',
  'int32',
  'int64',
  'int128',
  'int256',
  'uint',
  'uint8',
  'uint16',
  'uint32',
  'uint64',
  'uint128',
  'uint256',
  'bigint',
  'ubigint',
  'smallint',
  'usmallint',
  'signed',
  'unsigned',
]);
const CAST_FLOAT = new Set([
  'float',
  'float32',
  'float64',
  'double',
  'decimal',
  'real',
  'number',
  'numeric',
]);
const CAST_STRING = new Set(['string', 'text', 'varchar', 'char']);
const CAST_BOOL = new Set(['bool', 'boolean']);
const CAST_LIST = new Set(['list', 'array']);
// Temporal CAST targets desugar to the matching temporal constructor function
// (`date()`/`datetime()`/`local_time()`/…). `timestamp` is a DATETIME alias.
const CAST_TEMPORAL = new Map<string, string>([
  ['date', 'date'],
  ['datetime', 'datetime'],
  ['timestamp', 'datetime'],
  ['local_datetime', 'local_datetime'],
  ['local_time', 'local_time'],
  ['zoned_time', 'zoned_time'],
  ['zoned_datetime', 'zoned_datetime'],
  ['duration', 'duration'],
]);

const castTargetFn = (typeName: string): string | null => {
  const t = typeName.toLowerCase();

  if (CAST_INT.has(t)) {
    return 'to_integer';
  }

  if (CAST_FLOAT.has(t)) {
    return 'to_float';
  }

  if (CAST_STRING.has(t)) {
    return 'to_string';
  }

  if (CAST_BOOL.has(t)) {
    return 'to_boolean';
  }

  if (CAST_LIST.has(t)) {
    return 'to_list';
  }

  return CAST_TEMPORAL.get(t) ?? null;
};

/** Map the surrounding-arrow booleans to a relationship direction. */
const directionOf = (leftArrow: boolean, rightArrow: boolean): RelPattern['direction'] => {
  if (rightArrow && !leftArrow) {
    return 'out';
  }

  if (leftArrow && !rightArrow) {
    return 'in';
  }

  return 'both';
};

const COMPARE_OPS: Partial<Record<TokenType, CompareOp>> = {
  eq: '=',
  neq: '<>',
  lt: '<',
  gt: '>',
  lte: '<=',
  gte: '>=',
};

// Temporal typed-literal keywords: `<KW> '<iso>'` (e.g. `DATE '2020-01-01'`).
const TEMPORAL_KW = new Set(['date', 'datetime', 'timestamp', 'duration']);

const ADD_OPS: Partial<Record<TokenType, ArithOp>> = { plus: '+', dash: '-' };
const MUL_OPS: Partial<Record<TokenType, ArithOp>> = { star: '*', slash: '/', percent: '%' };

// A bare reserved word in a binding position is rejected per ISO. The message
// names backticks explicitly and echoes the user's ORIGINAL casing in both the
// name and the suggested delimited form — `keyword` tokens lowercase `value`,
// so `raw` carries the exact spelling (`` `Order` ``, never `` `order` ``).
const reservedError = (tok: Token, what: string): never => {
  const original = tok.raw ?? tok.value;

  throw new GqlSyntaxError(
    `\`${original}\` is a reserved word and can't be used bare as ${what}; ` +
      `quote it as a delimited identifier with backticks: \`${original}\``,
    tok.pos,
  );
};

// eslint-disable-next-line max-statements -- recursive-descent parser: the body is a suite of stateful closures over the token cursor; splitting them would only thread that state through parameters
export const parse = (
  src: string,
  opts?: { dialect?: Dialect; maxOperatorChain?: number },
): Statement => {
  const tokens = tokenize(src);
  let pos = 0;
  // `lenke` (default) recognizes sigil extensions like `_MERGE`; `iso-strict`
  // treats them as ordinary identifiers, so an extension clause is a syntax
  // error — the ISO surface stays provably self-contained.
  const dialect: Dialect = opts?.dialect ?? 'lenke';

  const peek = (): Token => tokens[pos];
  const atEnd = (): boolean => peek().type === 'eof';

  const advance = (): Token => tokens[pos++];

  const check = (type: TokenType): boolean => peek().type === type;

  const checkKeyword = (kw: string): boolean => peek().type === 'keyword' && peek().value === kw;

  // A sigil extension keyword (`_MERGE`, `_ON_CREATE`, …) lexes as a plain
  // identifier, so it's matched contextually here — case-insensitively, never
  // when backtick-delimited, and only under the `lenke` dialect (so it can never
  // shrink the ISO identifier namespace). `name` is the upper-cased sigil.
  const checkExtIdent = (name: string): boolean =>
    dialect === 'lenke' &&
    peek().type === 'ident' &&
    !peek().delimited &&
    peek().value.toUpperCase() === name;

  const expect = (type: TokenType, what: string): Token => {
    if (!check(type)) {
      throw new GqlSyntaxError(
        `Expected ${what}, got '${peek().value || peek().type}'`,
        peek().pos,
      );
    }

    return advance();
  };

  const expectKeyword = (kw: string): Token => {
    if (!checkKeyword(kw)) {
      throw new GqlSyntaxError(
        `Expected '${kw.toUpperCase()}', got '${peek().value || peek().type}'`,
        peek().pos,
      );
    }

    return advance();
  };

  // Recursion-depth guard. Recursive descent over deeply nested input
  // (`((((…))))`, `NOT NOT NOT …`, `!!!…`, nested lists / subqueries) would
  // otherwise overflow the JS stack with an uncaught `RangeError`. Wrapping the
  // recursive entry points in `descend` converts that into a clean
  // `GqlSyntaxError` past a fixed bound, well below any real stack limit.
  const MAX_DEPTH = 500;
  let depth = 0;

  const descend = <T>(body: () => T): T => {
    depth += 1;

    if (depth > MAX_DEPTH) {
      throw new GqlSyntaxError('Query nested too deeply', peek().pos);
    }

    try {
      return body();
    } finally {
      depth -= 1;
    }
  };

  // Operator-chain sanity ceiling. The associative operator nodes are n-ary (a
  // flat array, see `ast.ts`), so a long chain like `true AND true AND … (500k)`
  // is not a chain-deep tree and every walk (eval, analysis) is a loop — no stack
  // overflow regardless of chain length. This is therefore a pure anti-resource-
  // abuse guard (each operand is an allocation + an eval step), not crash-safety.
  // Defaults to 10k; configurable per graph (`new Graph({ maxOperatorChain })`,
  // read by `query()`), which mirrors the native engine's per-graph setting.
  const MAX_CHAIN = opts?.maxOperatorChain ?? 10_000;
  const chainLimit = (count: number): void => {
    if (count > MAX_CHAIN) {
      throw new GqlSyntaxError('Operator chain too long', peek().pos);
    }
  };

  // Consume a number token already known to be present and require it to be a
  // non-negative integer — for SKIP/LIMIT/OFFSET and quantifier bounds, where a
  // float, hex, NaN, or out-of-range value is never valid.
  const readCount = (what: string): number => {
    const t = advance();
    const n = t.num ?? Number.NaN;

    if (!Number.isInteger(n) || n < 0) {
      throw new GqlSyntaxError(`${what} must be a non-negative integer, got '${t.value}'`, t.pos);
    }

    return n;
  };

  const expectCount = (what: string): number => {
    if (!check('number')) {
      throw new GqlSyntaxError(
        `Expected ${what}, got '${peek().value || peek().type}'`,
        peek().pos,
      );
    }

    return readCount(what);
  };

  // A `LIMIT` / `OFFSET` bound: an integer literal, or — when `allowParam` — a
  // dynamic `$param` (ISO `nonNegativeIntegerSpecification`, opengql:2268). The
  // param's bound value is validated to be a non-negative integer at execution.
  // `SKIP` passes `allowParam=false`, so `SKIP $x` stays rejected (Cypher-only).
  const expectCountValue = (what: string, allowParam: boolean): CountValue => {
    if (allowParam && check('param')) {
      return { param: advance().value };
    }

    return expectCount(what);
  };

  // The single, consistent reserved-word rejection used in every binding
  // position. `what` names the role (a label name, a variable, …). The message
  // Consume an identifier in a *binding* position (variable, label, property
  // key, alias). A bare reserved word is rejected per ISO; a delimited
  // identifier (backtick) may be any word. Both token classes that can't be a
  // bare name here are caught up front so the rejection is uniform: a structural
  // `keyword` token (`Order`, `Count`, `Match`, `Set`, …) — which would
  // otherwise fail `expect('ident')` with a generic, casing-losing message — and
  // a reserved-but-not-structural `ident` (`Group`, `Product`).
  const bindName = (what: string): string => {
    const tok = peek();

    if (
      tok.type === 'keyword' ||
      (tok.type === 'ident' && !tok.delimited && isReserved(tok.value))
    ) {
      reservedError(tok, what);
    }

    return expect('ident', what).value;
  };

  // --- patterns --------------------------------------------------------------

  // Pattern property map `{ k: expr, ... }` and inline `WHERE pred` — the ISO
  // element-pattern predicate, shared by node and edge patterns.
  const parsePropertyMap = (): PropertyConstraint[] => {
    expect('lbrace', "'{'");
    const props: PropertyConstraint[] = [];

    if (!check('rbrace')) {
      do {
        const key = bindName('a property name');
        expect('colon', "':'");
        props.push({ key, value: parseExpr() });
      } while (check('comma') && (advance(), true));
    }

    expect('rbrace', "'}'");

    return props;
  };

  const parsePredicate = (): { properties?: PropertyConstraint[]; where?: Expr } => {
    const properties = check('lbrace') ? parsePropertyMap() : undefined;
    const where = checkKeyword('where') ? (advance(), parseExpr()) : undefined;

    return { properties, where };
  };

  const parseNode = (): NodePattern => {
    expect('lparen', "'('");
    let variable: string | undefined;

    if (check('ident')) {
      variable = bindName('a variable');
    }

    // ISO label expression, introduced by `:` or `IS`.
    let label: LabelExpr | undefined;

    if (check('colon') || checkKeyword('is')) {
      advance();
      label = parseLabelExpr();
    }

    const { properties, where } = parsePredicate();
    expect('rparen', "')'");

    return { variable, label, properties, where };
  };

  // Label expressions, lowest-to-highest precedence: `|` < `&` < `!` < primary.
  const parseLabelExpr = (): LabelExpr => descend(parseLabelOr);

  const parseLabelOr = (): LabelExpr => {
    let left = parseLabelAnd();

    while (check('pipe')) {
      advance();
      left = { kind: 'or', left, right: parseLabelAnd() };
    }

    return left;
  };

  const parseLabelAnd = (): LabelExpr => {
    let left = parseLabelNot();

    while (check('amp')) {
      advance();
      left = { kind: 'and', left, right: parseLabelNot() };
    }

    return left;
  };

  const parseLabelNot = (): LabelExpr =>
    descend((): LabelExpr => {
      if (check('bang')) {
        advance();

        return { kind: 'not', expr: parseLabelNot() };
      }

      return parseLabelPrimary();
    });

  const parseLabelPrimary = (): LabelExpr => {
    if (check('percent')) {
      advance();

      return { kind: 'wildcard' };
    }

    if (check('lparen')) {
      advance();
      const inner = parseLabelExpr();
      expect('rparen', "')' to close a label expression");

      return inner;
    }

    return { kind: 'label', name: bindName('a label name') };
  };

  const parseRelDetail = (): {
    variable?: string;
    label?: LabelExpr;
    properties?: PropertyConstraint[];
    where?: Expr;
  } => {
    // Caller has confirmed the next token is '['.
    expect('lbracket', "'['");
    let variable: string | undefined;

    if (check('ident')) {
      variable = bindName('a variable');
    }

    // ISO label expression over edge types, introduced by `:` or `IS`.
    let label: LabelExpr | undefined;

    if (check('colon') || checkKeyword('is')) {
      advance();
      label = parseLabelExpr();
    }

    const { properties, where } = parsePredicate();
    expect('rbracket', "']'");

    return { variable, label, properties, where };
  };

  // ISO abbreviated edges that stand alone (no bracket can follow them).
  const ABBREVIATED: Partial<Record<TokenType, RelPattern['direction']>> = {
    rarrow: 'out', // ->
    lrarrow: 'both', // <->
    tilder: 'both', // ~>
  };

  const parseRel = (): RelPattern => {
    // Pure abbreviated forms first.
    const abbrev = ABBREVIATED[peek().type];

    if (abbrev) {
      advance();

      return { direction: abbrev };
    }

    // Left marker: `<-` points in; `-`/`~`/`<~` are neutral so far. Each of
    // these can either stand alone (abbreviated) or open a bracketed edge.
    let leftArrow = false;

    if (check('larrow')) {
      advance();
      leftArrow = true;
    } else if (check('dash') || check('tilde') || check('ltilde')) {
      advance();
    } else {
      throw new GqlSyntaxError(
        `Expected a relationship (e.g. -[:T]->, <-[:T]-, ~[:T]~, ->), got '${peek().value || peek().type}'`,
        peek().pos,
      );
    }

    // No bracket → abbreviated edge: `<-` is incoming, the rest are undirected.
    if (!check('lbracket')) {
      return { direction: leftArrow ? 'in' : 'both' };
    }

    const detail = parseRelDetail();

    // Closing marker: `->` points out; `-`/`~`/`~>` are neutral.
    let rightArrow = false;

    if (check('rarrow')) {
      advance();
      rightArrow = true;
    } else if (check('dash') || check('tilde') || check('tilder')) {
      advance();
    } else {
      throw new GqlSyntaxError(
        `Expected ']->', ']-' or ']~' to close a relationship, got '${peek().value || peek().type}'`,
        peek().pos,
      );
    }

    return {
      variable: detail.variable,
      label: detail.label,
      direction: directionOf(leftArrow, rightArrow),
      properties: detail.properties,
      where: detail.where,
    };
  };

  const startsRelationship = (): boolean =>
    check('dash') ||
    check('larrow') ||
    check('rarrow') ||
    check('lrarrow') ||
    check('tilde') ||
    check('ltilde') ||
    check('tilder');

  // Variable-length quantifier following an edge: `*`, `+`, `{n}`, `{n,m}`,
  // `{n,}`, `{,m}`.
  const parseQuantifier = (): RelPattern['quantifier'] => {
    if (check('star')) {
      advance();

      return { min: 0, max: null };
    }

    if (check('plus')) {
      advance();

      return { min: 1, max: null };
    }

    if (check('lbrace')) {
      const open = advance();
      const min = check('number') ? readCount('a quantifier bound') : 0;
      let max: number | null = min;

      if (check('comma')) {
        advance();
        max = check('number') ? readCount('a quantifier bound') : null;
      }

      expect('rbrace', "'}' to close a quantifier");

      if (max !== null && max < min) {
        throw new GqlSyntaxError(
          `Quantifier upper bound ${max} is less than lower bound ${min}`,
          open.pos,
        );
      }

      return { min, max };
    }

    return undefined;
  };

  // An optional ISO path selector prefixing a pattern. Only `ANY SHORTEST` is
  // supported today; the other ISO forms are rejected with a pointed message.
  const parsePathSelector = (): PathSelector => {
    const pos0 = peek().pos;

    if (checkKeyword('any')) {
      advance();

      if (checkKeyword('shortest')) {
        advance();

        return 'anyShortest';
      }

      throw new GqlSyntaxError('expected SHORTEST after ANY (bare ANY is not yet supported)', pos0);
    }

    if (checkKeyword('all')) {
      advance();

      if (checkKeyword('shortest')) {
        advance();

        return 'allShortest';
      }

      throw new GqlSyntaxError('the bare ALL path selector is not yet supported', pos0);
    }

    if (checkKeyword('shortest')) {
      throw new GqlSyntaxError('SHORTEST must be written as `ANY SHORTEST`', pos0);
    }

    return 'walk';
  };

  const parsePathPattern = (): PathPattern =>
    descend((): PathPattern => {
      // Optional path variable `p = …`: at the start of a pattern an identifier
      // followed by `=` can only be a path-variable binding (a node opens with `(`).
      const selPos = peek().pos;
      let pathVar: string | undefined;

      if (check('ident') && tokens[pos + 1]?.type === 'eq') {
        pathVar = advance().value;
        advance(); // '='
      }

      const selector = parsePathSelector();

      if (pathVar !== undefined && selector === 'walk') {
        throw new GqlSyntaxError(
          'a named path variable currently requires a path selector (e.g. `p = ANY SHORTEST …`)',
          selPos,
        );
      }

      const start = parseNode();
      const segments: Segment[] = [];

      while (startsRelationship()) {
        const rel = parseRel();
        const quantifier = parseQuantifier();
        const node = parseNode();
        segments.push({ rel: quantifier ? { ...rel, quantifier } : rel, node });
      }

      // A selector matches exactly one variable-length segment with a `*`/`+`
      // (min ≤ 1) quantifier — the canonical shortest shape `(a)-[]->*(b)`.
      if (selector !== 'walk') {
        const q = segments.length === 1 ? segments[0].rel.quantifier : undefined;

        if (!q || q.min > 1) {
          throw new GqlSyntaxError(
            'ANY SHORTEST currently supports a single variable-length segment with a `*` or `+` (min ≤ 1) quantifier, e.g. `(a)-[]->*(b)`',
            selPos,
          );
        }
      }

      return {
        start,
        segments,
        ...(pathVar !== undefined ? { pathVar } : {}),
        selector,
      };
    });

  // Is the token after the current one the keyword `kw`? (Tells `OPTIONAL CALL`
  // from `OPTIONAL MATCH` without consuming.)
  const kwAfter = (kw: string): boolean => {
    const t = tokens[pos + 1];

    return t?.type === 'keyword' && t.value === kw;
  };

  const parseYieldItem = (): YieldItem => {
    const name = bindName('a YIELD column name');
    const alias = checkKeyword('as') ? (advance(), bindName('a YIELD alias')) : undefined;

    return { name, ...(alias !== undefined ? { alias } : {}) };
  };

  const SET_OPS: Partial<Record<string, SetOp['op']>> = {
    union: 'union',
    except: 'except',
    intersect: 'intersect',
  };

  // Parse one or more linear parts joined by set operators, WITHOUT requiring
  // end-of-input afterward — so it serves both the top level (wrapped with the
  // trailing-input check) and an inline `CALL (…) { … }` body (closed by `}`).
  const parseSetOpQuery = (): Query => {
    const parts: LinearQuery[] = [parseLinearQuery()];
    const ops: SetOp[] = [];

    while (peek().type === 'keyword' && SET_OPS[peek().value]) {
      const op = SET_OPS[advance().value]!;
      const all = checkKeyword('all') ? (advance(), true) : false;

      if (!all && checkKeyword('distinct')) {
        advance();
      }

      ops.push({ op, all });
      parts.push(parseLinearQuery());
    }

    return { parts, ops };
  };

  // `[OPTIONAL] CALL (scope) { … }` — an inline subquery.
  const parseInlineCall = (optional: boolean): CallInlineClause => {
    const scope: string[] = [];

    if (check('lparen')) {
      advance();

      if (!check('rparen')) {
        scope.push(bindName('a scoped variable'));

        while (check('comma')) {
          advance();
          scope.push(bindName('a scoped variable'));
        }
      }

      expect('rparen', "')' to close the variable scope");
    }

    expect('lbrace', "'{' to open an inline subquery");
    const body = parseSetOpQuery();
    expect('rbrace', "'}' to close an inline subquery");

    return { kind: 'callInline', optional, scope, body };
  };

  // `[OPTIONAL] CALL …`: the named-procedure form `CALL name(config) [YIELD …]`,
  // or the inline-subquery form `CALL (scope) { … }` / `CALL { … }`.
  const parseCallClause = (): CallNamedClause | CallInlineClause => {
    const optional = checkKeyword('optional') ? (advance(), true) : false;
    expectKeyword('call');

    if (check('lbrace') || check('lparen')) {
      return parseInlineCall(optional);
    }

    let name = bindName('a procedure name');

    while (check('dot')) {
      advance();
      name += `.${bindName('a procedure name segment')}`;
    }

    expect('lparen', "'(' after a procedure name");
    const config = check('lbrace') ? parsePropertyMap() : [];
    expect('rparen', "')' to close procedure arguments");

    let yields: YieldItem[] | undefined;

    if (checkKeyword('yield')) {
      advance();
      yields = [parseYieldItem()];

      while (check('comma')) {
        advance();
        yields.push(parseYieldItem());
      }
    }

    return { kind: 'callNamed', optional, name, config, ...(yields ? { yields } : {}) };
  };

  const parseMatchClause = (): MatchClause => {
    const optional = checkKeyword('optional') ? (advance(), true) : false;
    expectKeyword('match');
    const patterns: PathPattern[] = [parsePathPattern()];

    while (check('comma')) {
      advance();
      patterns.push(parsePathPattern());
    }

    const where = checkKeyword('where') ? (advance(), parseExpr()) : undefined;

    return { kind: 'match', optional, patterns, where };
  };

  // --- expressions -----------------------------------------------------------

  // Precedence, loosest to tightest (ISO §24): OR/XOR, AND, NOT, IS/IN
  // predicates, comparison, `||`, +/-, *///%, unary, primary.
  const parseExpr = (): Expr => descend(parseOrXor);

  // ISO/IEC 39075 `<boolean value expression>`: OR and XOR share one (loosest)
  // precedence level and are left-associative, so `a OR b XOR c` parses as
  // `(a OR b) XOR c`. AND (`<boolean term>`) binds tighter, NOT tighter still.
  // NB: a deliberate divergence from Cypher, which gives XOR higher precedence
  // than OR — we follow the ISO grammar, not Cypher.
  const parseOrXor = (): Expr => {
    let left = parseAnd();
    let chain = 0;

    // Flatten a maximal run of the SAME operator into one n-ary node; nest on an
    // operator switch (both left-associative at one precedence level), so
    // `a OR b OR c XOR d` is `xor([or([a,b,c]), d])`.
    while (checkKeyword('or') || checkKeyword('xor')) {
      chainLimit((chain += 1));
      const isOr = advance().value === 'or';
      const items: Expr[] = [left, parseAnd()];

      while ((isOr && checkKeyword('or')) || (!isOr && checkKeyword('xor'))) {
        chainLimit((chain += 1));
        advance();
        items.push(parseAnd());
      }

      left = { kind: isOr ? 'or' : 'xor', items };
    }

    return left;
  };

  const parseAnd = (): Expr => {
    const first = parseNot();

    if (!checkKeyword('and')) {
      return first;
    }

    const items: Expr[] = [first];
    let chain = 0;

    while (checkKeyword('and')) {
      chainLimit((chain += 1));
      advance();
      items.push(parseNot());
    }

    return { kind: 'and', items };
  };

  const parseNot = (): Expr =>
    descend((): Expr => {
      if (checkKeyword('not')) {
        advance();

        return { kind: 'not', expr: parseNot() };
      }

      return parsePostfixPredicate();
    });

  // Postfix predicates: `x IS [NOT] NULL`, `x IS [NOT] TRUE|FALSE|UNKNOWN`,
  // `x [NOT] IN list`.
  const parsePostfixPredicate = (): Expr => {
    const e = parseComparison();

    // ISO `<labeled predicate>` COLON form: `n:Label` on a bare element variable
    // reference (opengql:2078 — `elementVariableReference COLON labelExpression`).
    // Desugars to the same `isLabeled` node as `IS LABELED`, reusing the pattern
    // parser's label-expression grammar (so `n:A|B`, `n:A&B`, `n:!A` all work).
    if (check('colon') && e.kind === 'var') {
      advance();

      return { kind: 'isLabeled', expr: e, label: parseLabelExpr(), negated: false };
    }

    if (checkKeyword('is')) {
      advance();
      const negated = checkKeyword('not') ? (advance(), true) : false;

      if (checkKeyword('null')) {
        advance();

        return { kind: 'isNull', expr: e, negated };
      }

      // ISO `<boolean test>`: IS [NOT] TRUE | FALSE | UNKNOWN.
      if (checkKeyword('true')) {
        advance();

        return { kind: 'isTruth', expr: e, truth: true, negated };
      }

      if (checkKeyword('false')) {
        advance();

        return { kind: 'isTruth', expr: e, truth: false, negated };
      }

      if (checkKeyword('unknown')) {
        advance();

        return { kind: 'isTruth', expr: e, truth: null, negated };
      }

      // ISO `<labeled predicate>`: IS [NOT] LABELED <label expression>. LABELED
      // is a non-reserved keyword, so it arrives as an identifier.
      if (check('ident') && peek().value.toLowerCase() === 'labeled') {
        advance();

        return { kind: 'isLabeled', expr: e, label: parseLabelExpr(), negated };
      }

      throw new GqlSyntaxError(
        'Expected NULL, TRUE, FALSE, UNKNOWN, or LABELED after IS',
        peek().pos,
      );
    }

    if (checkKeyword('in')) {
      advance();

      return { kind: 'in', expr: e, list: parseUnary(), negated: false };
    }

    if (checkKeyword('not')) {
      advance();
      expectKeyword('in');

      return { kind: 'in', expr: e, list: parseUnary(), negated: true };
    }

    // ISO string-matching predicates. `contains`/`starts`/`ends` are non-reserved
    // words (plain idents after a complete operand); they desugar to the
    // equivalent BOOL functions. `STARTS`/`ENDS` require a following `WITH`.
    if (check('ident')) {
      const word = peek().value.toLowerCase();

      if (word === 'contains') {
        advance();

        return {
          kind: 'func',
          name: 'contains',
          args: [e, parseConcat()],
          distinct: false,
          star: false,
        };
      }

      if (word === 'starts') {
        advance();
        expectKeyword('with');

        return {
          kind: 'func',
          name: 'starts_with',
          args: [e, parseConcat()],
          distinct: false,
          star: false,
        };
      }

      if (word === 'ends') {
        advance();
        expectKeyword('with');

        return {
          kind: 'func',
          name: 'ends_with',
          args: [e, parseConcat()],
          distinct: false,
          star: false,
        };
      }
    }

    return e;
  };

  const parseComparison = (): Expr => {
    const left = parseConcat();
    const op = COMPARE_OPS[peek().type];

    if (op) {
      advance();

      return { kind: 'compare', op, left, right: parseConcat() };
    }

    return left;
  };

  const parseConcat = (): Expr => {
    const first = parseAdditive();

    if (!check('concat')) {
      return first;
    }

    const items: Expr[] = [first];
    let chain = 0;

    while (check('concat')) {
      chainLimit((chain += 1));
      advance();
      items.push(parseAdditive());
    }

    return { kind: 'concat', items };
  };

  const parseAdditive = (): Expr => {
    const head = parseMultiplicative();
    const tail: (readonly [ArithOp, Expr])[] = [];
    let op = ADD_OPS[peek().type];
    let chain = 0;

    while (op) {
      chainLimit((chain += 1));
      advance();
      tail.push([op, parseMultiplicative()]);
      op = ADD_OPS[peek().type];
    }

    return tail.length === 0 ? head : { kind: 'arith', head, tail };
  };

  const parseMultiplicative = (): Expr => {
    const head = parseUnary();
    const tail: (readonly [ArithOp, Expr])[] = [];
    let op = MUL_OPS[peek().type];
    let chain = 0;

    while (op) {
      chainLimit((chain += 1));
      advance();
      tail.push([op, parseUnary()]);
      op = MUL_OPS[peek().type];
    }

    return tail.length === 0 ? head : { kind: 'arith', head, tail };
  };

  const parseUnary = (): Expr =>
    descend((): Expr => {
      if (check('dash')) {
        advance();

        return { kind: 'neg', expr: parseUnary() };
      }

      if (check('plus')) {
        advance();

        return parseUnary();
      }

      // Postfix subscript `base[index]` and property access `base.key`, chained
      // left to right (`a[0].amount`, `head(rels).x`). A leading `[` is still a
      // list literal, and a bare `variable.key` is `prop` (parsePrimary); this
      // loop extends a primary with further `[…]`/`.key` steps.
      let e = parsePrimary();

      for (;;) {
        if (check('lbracket')) {
          advance();

          const idx = parseExpr();

          expect('rbracket', "']' to close a list subscript");
          e = { kind: 'index', base: e, index: idx };
        } else if (check('dot')) {
          advance();
          e = { kind: 'field', base: e, key: bindName('a property name') };
        } else {
          break;
        }
      }

      return e;
    });

  // The body shared by the braced subqueries `EXISTS { … }` and `COUNT { … }`:
  // `{ pattern, … [WHERE pred] }`, with an optional leading MATCH keyword.
  const parseBracedSubquery = (): { patterns: PathPattern[]; where?: Expr } => {
    expect('lbrace', "'{'");

    if (checkKeyword('match')) {
      advance();
    }

    const patterns: PathPattern[] = [parsePathPattern()];

    while (check('comma')) {
      advance();
      patterns.push(parsePathPattern());
    }

    const where = checkKeyword('where') ? (advance(), parseExpr()) : undefined;
    expect('rbrace', "'}'");

    return { patterns, where };
  };

  // `(*)` | `(DISTINCT? expr, …)` — the argument list of a function/aggregate
  // call, the caller having already consumed the function name.
  const parseCallArgs = (): { args: Expr[]; distinct: boolean; star: boolean } => {
    expect('lparen', "'(' to open a function call");
    let star = false;
    let distinct = false;
    const args: Expr[] = [];

    if (check('star')) {
      advance();
      star = true;
    } else if (!check('rparen')) {
      if (checkKeyword('distinct')) {
        advance();
        distinct = true;
      }

      args.push(parseExpr());

      while (check('comma')) {
        advance();
        args.push(parseExpr());
      }
    }

    expect('rparen', "')' to close a function call");

    return { args, distinct, star };
  };

  // `CAST(value AS type)` — the leading `cast` ident is consumed; the current
  // token is `(`. Desugars to the conversion function for the target type. An
  // unrepresentable target type is a loud syntax error, never a silent null.
  const parseCast = (): Expr => {
    expect('lparen', "'(' after CAST");
    const value = parseExpr();

    if (!checkKeyword('as')) {
      throw new GqlSyntaxError("expected 'AS' in CAST(value AS type)", peek().pos);
    }

    advance();
    const typeTok = peek();

    if (typeTok.type !== 'ident' && typeTok.type !== 'keyword') {
      throw new GqlSyntaxError(
        `expected a type name in CAST, got '${typeTok.value || typeTok.type}'`,
        typeTok.pos,
      );
    }

    advance();

    // Two-word temporal type names: `LOCAL DATETIME` / `LOCAL TIME` /
    // `ZONED TIME` / `ZONED DATETIME`. The compact `LOCAL_DATETIME` keyword forms
    // are single tokens and fall through unchanged.
    let typeName = typeTok.value;
    const lead = typeName.toLowerCase();

    if (lead === 'local' || lead === 'zoned') {
      const next = peek();

      if (next.type === 'ident' || next.type === 'keyword') {
        const word = next.value.toLowerCase();

        if (word === 'datetime' || word === 'time') {
          advance();
          typeName = `${lead}_${word}`;
        }
      }
    }

    const fn = castTargetFn(typeName);

    if (fn === null) {
      throw new GqlSyntaxError(`CAST to unsupported type '${typeName}'`, typeTok.pos);
    }

    expect('rparen', "')' to close CAST");

    return { kind: 'func', name: fn, args: [value], distinct: false, star: false };
  };

  // EXISTS is a reserved word; it only introduces a braced subquery.
  const parseExists = (): Expr => {
    expectKeyword('exists');

    return { kind: 'exists', ...parseBracedSubquery() };
  };

  // COUNT is reserved and overloaded: `COUNT { … }` is the subquery, `COUNT(…)`
  // (incl. `COUNT(*)`) is the aggregate.
  const parseCount = (): Expr => {
    expectKeyword('count');

    if (check('lbrace')) {
      return { kind: 'countSubquery', ...parseBracedSubquery() };
    }

    return { kind: 'func', name: 'count', ...parseCallArgs() };
  };

  // ISO `<case expression>`: `CASE [subject] (WHEN test THEN result)+ [ELSE r] END`.
  // A subject before the first WHEN makes it a simple CASE; otherwise searched.
  const parseCase = (): Expr => {
    expectKeyword('case');
    const subject = checkKeyword('when') ? undefined : parseExpr();
    const whens: { when: Expr; then: Expr }[] = [];

    while (checkKeyword('when')) {
      advance();
      const when = parseExpr();
      expectKeyword('then');
      // `then` is the ISO GQL CASE…WHEN…THEN branch, not a thenable; never awaited.
      // eslint-disable-next-line unicorn/no-thenable
      whens.push({ when, then: parseExpr() });
    }

    if (whens.length === 0) {
      throw new GqlSyntaxError('CASE requires at least one WHEN ... THEN', peek().pos);
    }

    const elseExpr = checkKeyword('else') ? (advance(), parseExpr()) : undefined;
    expectKeyword('end');

    return { kind: 'case', subject, whens, elseExpr };
  };

  /** Is the current token a temporal type keyword directly before a string? */
  const temporalLiteralAhead = (): boolean => {
    const t = peek();

    return (
      t.type === 'ident' &&
      !t.delimited &&
      TEMPORAL_KW.has(t.value.toLowerCase()) &&
      tokens[pos + 1]?.type === 'string'
    );
  };

  const parseTemporalLiteral = (): Expr => {
    const kw = advance();
    const strTok = advance();
    const kind = kw.value.toLowerCase();
    let tag: 'date' | 'datetime' | 'duration';

    if (kind === 'date') {
      tag = 'date';
    } else if (kind === 'duration') {
      tag = 'duration';
    } else {
      tag = 'datetime';
    }

    try {
      return { kind: 'lit', value: temporalParse(tag, strTok.value) };
    } catch (cause) {
      throw new GqlSyntaxError(
        `invalid ${kw.value.toUpperCase()} literal: ${(cause as Error).message}`,
        strTok.pos,
      );
    }
  };

  const parsePrimary = (): Expr => {
    const t = peek();

    if (t.type === 'number') {
      advance();

      return { kind: 'lit', value: t.num };
    }

    if (t.type === 'string') {
      advance();

      return { kind: 'lit', value: t.value };
    }

    // ISO typed temporal literal: `DATE '2020-01-01'` / `DATETIME '…'` /
    // `TIMESTAMP '…'` / `DURATION 'P…'` — a soft-keyword ident before a string.
    // (The `date(…)` constructor-function form falls through to the call path.)
    if (temporalLiteralAhead()) {
      return parseTemporalLiteral();
    }

    // Bare now-functions `current_date` / `current_timestamp` / `local_timestamp`
    // desugar to a reserved `$__now` DATETIME param the host supplies — the engine
    // never reads the clock, which keeps the two engines byte-identical.
    // `current_date` truncates via `date(...)`; the datetime forms wrap in
    // `local_datetime(...)` so the result is DATETIME-kind regardless of what
    // kind `$__now` was supplied as (a DATE `$__now` coerces to midnight rather
    // than leaking a DATE out of `current_timestamp`).
    if (t.type === 'ident' && !t.delimited) {
      const lc = t.value.toLowerCase();

      if (lc === 'current_date' || lc === 'current_timestamp' || lc === 'local_timestamp') {
        advance();

        if (check('lparen')) {
          advance();
          expect('rparen', "')' to close a now-function");
        }

        const now: Expr = { kind: 'param', name: '__now' };
        const fn = lc === 'current_date' ? 'date' : 'local_datetime';

        return { kind: 'func', name: fn, args: [now], distinct: false, star: false };
      }
    }

    if (t.type === 'param') {
      advance();

      return { kind: 'param', name: t.value };
    }

    if (t.type === 'keyword' && (t.value === 'true' || t.value === 'false')) {
      advance();

      return { kind: 'lit', value: t.value === 'true' };
    }

    if (t.type === 'keyword' && t.value === 'null') {
      advance();

      return { kind: 'lit', value: null };
    }

    if (checkKeyword('case')) {
      return parseCase();
    }

    if (checkKeyword('exists')) {
      return parseExists();
    }

    if (checkKeyword('count')) {
      return parseCount();
    }

    if (t.type === 'lparen') {
      advance();
      const inner = parseExpr();
      expect('rparen', "')'");

      return inner;
    }

    if (t.type === 'lbracket') {
      advance();
      const items: Expr[] = [];

      if (!check('rbracket')) {
        do {
          items.push(parseExpr());
        } while (check('comma') && (advance(), true));
      }

      expect('rbracket', "']' to close a list");

      return { kind: 'list', items };
    }

    if (t.type === 'ident') {
      advance();

      // Function call: the name may be a reserved word (e.g. UPPER, SUM, ABS).
      if (check('lparen')) {
        // `CAST(value AS type)` is a keyword-shaped call; desugar it to the
        // matching conversion function (to_integer/…).
        if (!t.delimited && t.value.toLowerCase() === 'cast') {
          return parseCast();
        }

        return { kind: 'func', name: t.value.toLowerCase(), ...parseCallArgs() };
      }

      // A bare reserved word is not a valid variable reference.
      if (!t.delimited && isReserved(t.value)) {
        reservedError(t, 'a variable');
      }

      if (check('dot')) {
        advance();
        const key = bindName('a property name');

        return { kind: 'prop', variable: t.value, key };
      }

      return { kind: 'var', name: t.value };
    }

    throw new GqlSyntaxError(`Unexpected '${t.value || t.type}' in expression`, t.pos);
  };

  // --- return ----------------------------------------------------------------

  const parseReturnItem = (): ReturnItem => {
    const expr = parseExpr();
    let alias: string | undefined;

    if (checkKeyword('as')) {
      advance();
      alias = bindName('an alias name');
    }

    return { expr, alias };
  };

  const parseSortItem = (): SortItem => {
    const expr = parseExpr();
    let descending = false;

    if (checkKeyword('desc') || checkKeyword('descending')) {
      advance();
      descending = true;
    } else if (checkKeyword('asc') || checkKeyword('ascending')) {
      advance();
    }

    // ISO `<null ordering>`: optional NULLS FIRST | NULLS LAST. NULLS is a
    // reserved word; FIRST/LAST are non-reserved, so they arrive as identifiers.
    let nullsFirst: boolean | undefined;

    if (checkKeyword('nulls')) {
      advance();
      const where = check('ident') ? peek().value.toLowerCase() : '';

      if (where === 'first') {
        nullsFirst = true;
        advance();
      } else if (where === 'last') {
        nullsFirst = false;
        advance();
      } else {
        throw new GqlSyntaxError('Expected FIRST or LAST after NULLS', peek().pos);
      }
    }

    return { expr, descending, nullsFirst };
  };

  // The shared projection body of WITH and RETURN.
  const parseProjection = (): Projection => {
    const distinct = checkKeyword('distinct') ? (advance(), true) : false;

    let star = false;
    const items: ReturnItem[] = [];

    if (check('star')) {
      advance();
      star = true;
    } else {
      items.push(parseReturnItem());

      while (check('comma')) {
        advance();
        items.push(parseReturnItem());
      }
    }

    let orderBy: SortItem[] | undefined;

    if (checkKeyword('order')) {
      advance();
      expectKeyword('by');
      orderBy = [parseSortItem()];

      while (check('comma')) {
        advance();
        orderBy.push(parseSortItem());
      }
    }

    let skip: CountValue | undefined;

    if (checkKeyword('skip') || checkKeyword('offset')) {
      // OFFSET is the ISO spelling and accepts a dynamic `$param`; SKIP is the
      // Cypher synonym and stays literal-only.
      const allowParam = checkKeyword('offset');
      advance();
      skip = expectCountValue('a non-negative integer after SKIP/OFFSET', allowParam);
    }

    let limit: CountValue | undefined;

    if (checkKeyword('limit')) {
      advance();
      limit = expectCountValue('a non-negative integer after LIMIT', true);
    }

    return { star, items, distinct, orderBy, skip, limit };
  };

  const parseWithClause = (): WithClause => {
    expectKeyword('with');
    const projection = parseProjection();
    const where = checkKeyword('where') ? (advance(), parseExpr()) : undefined;

    return { kind: 'with', projection, where };
  };

  // `FILTER [WHERE] <condition>` — ISO GQL §14.6 (the WHERE is optional noise).
  const parseFilterClause = (): FilterClause => {
    expectKeyword('filter');

    if (checkKeyword('where')) {
      advance();
    }

    return { kind: 'filter', where: parseExpr() };
  };

  // `LET x = <expr> [, y = <expr>]*` — ISO GQL §14.7. Comma-separated bindings.
  const parseLetClause = (): LetClause => {
    expectKeyword('let');
    const items: { var: string; expr: Expr }[] = [];

    do {
      const v = bindName('a LET variable');
      expect('eq', "'=' after a LET variable");
      items.push({ var: v, expr: parseExpr() });
    } while (check('comma') && (advance(), true));

    return { kind: 'let', items };
  };

  // Is the token after `WITH` an ORDINALITY/OFFSET modifier (vs the start of a
  // new WITH clause)? `ORDINALITY` is a soft keyword — it arrives as an ident.
  const forModifierAhead = (): boolean => {
    const t = tokens[pos + 1];

    return (
      t !== undefined &&
      ((t.type === 'keyword' && t.value === 'offset') ||
        (t.type === 'ident' && t.value.toLowerCase() === 'ordinality'))
    );
  };

  const parseForClause = (): ForClause => {
    expectKeyword('for');
    const alias = bindName('a FOR variable');
    expectKeyword('in');
    // `IN` is consumed as a keyword up front, so it is not mistaken for the `IN`
    // membership operator inside the list expression.
    const list = parseExpr();
    // `WITH ORDINALITY|OFFSET var` is a FOR modifier ONLY when ORDINALITY/OFFSET
    // follows WITH; a bare WITH here begins the next clause and must be left for
    // the clause loop.
    let ordinality: ForClause['ordinality'];

    if (checkKeyword('with') && forModifierAhead()) {
      advance(); // WITH
      const kind: 'ordinality' | 'offset' = checkKeyword('offset') ? 'offset' : 'ordinality';
      advance(); // OFFSET (keyword) or ORDINALITY (soft ident)
      ordinality = { kind, var: bindName('an ORDINALITY/OFFSET variable') };
    }

    return { kind: 'for', alias, list, ordinality };
  };

  const parseReturnClause = (): ReturnClause => {
    expectKeyword('return');

    return { kind: 'return', projection: parseProjection() };
  };

  // --- write clauses ---------------------------------------------------------

  const parseInsertClause = (): InsertClause => {
    expectKeyword('insert');
    const patterns: PathPattern[] = [parsePathPattern()];

    while (check('comma')) {
      advance();
      patterns.push(parsePathPattern());
    }

    return { kind: 'insert', patterns };
  };

  const parseSetItem = (): SetItem => {
    const variable = bindName('a variable');

    if (check('colon') || checkKeyword('is')) {
      advance();

      return { variable, label: bindName('a label name') };
    }

    expect('dot', "'.' or ':'");
    const key = bindName('a property name');
    expect('eq', "'='");

    return { variable, key, value: parseExpr() };
  };

  const parseSetItemList = (): SetItem[] => {
    const items: SetItem[] = [parseSetItem()];

    while (check('comma')) {
      advance();
      items.push(parseSetItem());
    }

    return items;
  };

  const parseSetClause = (): SetClause => {
    expectKeyword('set');

    return { kind: 'set', items: parseSetItemList() };
  };

  // `_MERGE pattern [_ON_CREATE SET …] [_ON_UPDATE SET … [WHERE p] |
  // _ON_UPDATE_NOTHING]` — the lenke keyed-upsert extension (sigil-marked; see
  // docs/design/gql-extensions.md §2). The pattern reuses the standard path
  // grammar; branches may appear in any order, each at most once, and an explicit
  // _ON_UPDATE excludes _ON_UPDATE_NOTHING (one update disposition).
  const parseMergeClause = (): MergeClause => {
    advance(); // _MERGE
    const pattern = parsePathPattern();
    let onCreate: SetItem[] | undefined;
    let onUpdate: MergeUpdate | undefined;

    for (;;) {
      if (checkExtIdent('_ON_CREATE')) {
        advance();

        if (onCreate) {
          throw new GqlSyntaxError('duplicate _ON_CREATE in _MERGE', peek().pos);
        }

        expectKeyword('set');
        onCreate = parseSetItemList();
      } else if (checkExtIdent('_ON_UPDATE_NOTHING')) {
        advance();

        if (onUpdate) {
          throw new GqlSyntaxError('conflicting update disposition in _MERGE', peek().pos);
        }

        onUpdate = { kind: 'nothing' };
      } else if (checkExtIdent('_ON_UPDATE')) {
        advance();

        if (onUpdate) {
          throw new GqlSyntaxError('conflicting update disposition in _MERGE', peek().pos);
        }

        expectKeyword('set');
        const items = parseSetItemList();
        let where: Expr | undefined;

        if (checkKeyword('where')) {
          advance();
          where = parseExpr();
        }

        onUpdate = { kind: 'set', items, where };
      } else {
        break;
      }
    }

    return { kind: 'merge', pattern, onCreate, onUpdate };
  };

  const parseRemoveItem = (): RemoveItem => {
    const variable = bindName('a variable');

    if (check('colon') || checkKeyword('is')) {
      advance();

      return { variable, label: bindName('a label name') };
    }

    expect('dot', "'.' or ':'");

    return { variable, key: bindName('a property name') };
  };

  const parseRemoveClause = (): RemoveClause => {
    expectKeyword('remove');
    const items: RemoveItem[] = [parseRemoveItem()];

    while (check('comma')) {
      advance();
      items.push(parseRemoveItem());
    }

    return { kind: 'remove', items };
  };

  const parseDeleteClause = (): DeleteClause => {
    const detach = checkKeyword('detach') ? (advance(), true) : false;

    if (!detach && checkKeyword('nodetach')) {
      advance();
    }

    expectKeyword('delete');
    const targets: Expr[] = [parseExpr()];

    while (check('comma')) {
      advance();
      targets.push(parseExpr());
    }

    return { kind: 'delete', detach, targets };
  };

  // A linear query: a sequence of clauses, optionally ending in RETURN or
  // FINISH (a write-only query needs neither).
  const parseLinearQuery = (): LinearQuery => {
    const clauses: Clause[] = [];
    let done = false;

    while (!done && !atEnd()) {
      if (checkKeyword('return')) {
        clauses.push(parseReturnClause());
        done = true;
      } else if (checkKeyword('finish')) {
        advance();
        clauses.push({ kind: 'finish' });
        done = true;
      } else if (checkKeyword('with')) {
        clauses.push(parseWithClause());
      } else if (checkKeyword('let')) {
        clauses.push(parseLetClause());
      } else if (checkKeyword('filter')) {
        clauses.push(parseFilterClause());
      } else if (checkKeyword('for')) {
        clauses.push(parseForClause());
      } else if (checkKeyword('call') || (checkKeyword('optional') && kwAfter('call'))) {
        clauses.push(parseCallClause());
      } else if (checkKeyword('match') || checkKeyword('optional')) {
        clauses.push(parseMatchClause());
      } else if (checkKeyword('insert')) {
        clauses.push(parseInsertClause());
      } else if (checkExtIdent('_MERGE')) {
        clauses.push(parseMergeClause());
      } else if (checkKeyword('set')) {
        clauses.push(parseSetClause());
      } else if (checkKeyword('remove')) {
        clauses.push(parseRemoveClause());
      } else if (checkKeyword('delete') || checkKeyword('detach') || checkKeyword('nodetach')) {
        clauses.push(parseDeleteClause());
      } else {
        break;
      }
    }

    if (clauses.length === 0) {
      throw new GqlSyntaxError(
        `Expected a clause (MATCH, INSERT, RETURN, …), got '${peek().value || peek().type}'`,
        peek().pos,
      );
    }

    return { clauses };
  };

  // --- transaction-control statements (ISO/IEC 39075) ------------------------

  // The transaction keywords (START/TRANSACTION/COMMIT/ROLLBACK/WORK/READ/ONLY/
  // WRITE) are recognized *contextually* — matched here on the identifier text at
  // statement start rather than promoted to lexer keywords or global reserved
  // words. That leaves them fully usable as ordinary identifiers (variables,
  // labels, aliases) everywhere else, so `READ`/`WRITE`/`ONLY`/`WORK`/
  // `TRANSACTION` never shrink the namespace. (`START`/`COMMIT`/`ROLLBACK` remain
  // ISO reserved words as they already were — quote them to use as identifiers.)
  const isWord = (word: string): boolean =>
    peek().type === 'ident' && !peek().delimited && peek().value.toLowerCase() === word;

  // A statement is transaction control iff its first token is a bare `START`,
  // `COMMIT`, or `ROLLBACK` identifier. A linear query can never begin with one of
  // these, so there is no ambiguity — and a delimited `` `start` `` stays an
  // identifier.
  const startsTxControl = (): boolean => isWord('start') || isWord('commit') || isWord('rollback');

  // `READ ONLY | READ WRITE` — the single optional access mode after
  // `START TRANSACTION`. ISO also allows a comma-separated mode list; v1 supports
  // one mode. A second mode / trailing comma is left to the top-level trailing-
  // input check (a syntax error), and `READ` not followed by `ONLY`/`WRITE` is a
  // syntax error here.
  const parseAccessMode = (): TxControl['accessMode'] | undefined => {
    if (!isWord('read')) {
      return undefined;
    }

    advance(); // READ

    if (isWord('only')) {
      advance();

      return 'read only';
    }

    if (isWord('write')) {
      advance();

      return 'read write';
    }

    throw new GqlSyntaxError('Expected ONLY or WRITE after READ', peek().pos);
  };

  const parseTxControl = (): TxControl => {
    const kw = advance().value.toLowerCase(); // start | commit | rollback

    if (kw === 'start') {
      if (!isWord('transaction')) {
        throw new GqlSyntaxError(
          `Expected TRANSACTION after START, got '${peek().value || peek().type}'`,
          peek().pos,
        );
      }

      advance(); // TRANSACTION

      return { kind: 'start', accessMode: parseAccessMode() };
    }

    // COMMIT / ROLLBACK — optionally followed by the noise word WORK.
    if (isWord('work')) {
      advance();
    }

    return { kind: kw === 'commit' ? 'commit' : 'rollback' };
  };

  if (startsTxControl()) {
    const tx = parseTxControl();

    if (!atEnd()) {
      throw new GqlSyntaxError(
        `Unexpected trailing input '${peek().value || peek().type}'`,
        peek().pos,
      );
    }

    return tx;
  }

  // --- top level: linear queries joined by set operators ---------------------

  // A NEXT segment must be a single linear query (no set operators) for the
  // RETURN→WITH rewrite below. Returns a mutable copy of its clauses.
  const takeLinearForNext = (q: Query): Clause[] => {
    if (q.ops.length > 0 || q.parts.length !== 1) {
      throw new GqlSyntaxError(
        'NEXT does not support set operators (UNION/EXCEPT/INTERSECT) in a composed statement',
        peek().pos,
      );
    }

    return [...q.parts[0].clauses];
  };

  // One `YIELD col [AS alias]` → a `Var(col)` projection item aliased to `alias`
  // (or `col`), so the piped column is selected (and optionally renamed).
  const nextYieldItem = (): ReturnItem => {
    const name = bindName('a YIELD column');
    const alias = checkKeyword('as') ? (advance(), bindName('a YIELD alias')) : name;

    return { expr: { kind: 'var', name }, alias };
  };

  // ISO GQL `NEXT` linear-statement composition: `A NEXT [YIELD …] B [NEXT …]`
  // pipes each statement's RETURN output forward as the next's driving table.
  // Rewrite: each pre-NEXT RETURN becomes a WITH (plus a second WITH for YIELD),
  // concatenated into one linear query — reusing all WITH machinery.
  const parseNextChain = (): Query => {
    const head = parseSetOpQuery();

    if (!checkKeyword('next')) {
      return head;
    }

    const clauses = takeLinearForNext(head);

    while (checkKeyword('next')) {
      advance();

      const last = clauses.pop();

      if (last?.kind !== 'return') {
        throw new GqlSyntaxError('NEXT must follow a RETURN', peek().pos);
      }

      clauses.push({ kind: 'with', projection: last.projection });

      if (checkKeyword('yield')) {
        advance();
        const items = [nextYieldItem()];

        while (check('comma')) {
          advance();
          items.push(nextYieldItem());
        }

        clauses.push({ kind: 'with', projection: { star: false, items, distinct: false } });
      }

      clauses.push(...takeLinearForNext(parseSetOpQuery()));
    }

    return { parts: [{ clauses }], ops: [] };
  };

  const query = parseNextChain();

  if (!atEnd()) {
    throw new GqlSyntaxError(
      `Unexpected trailing input '${peek().value || peek().type}'`,
      peek().pos,
    );
  }

  return query;
};

/**
 * Parse a bare boolean predicate — a WHERE-clause expression — into its `Expr`
 * AST. This is the compiler surface a declarative VALIDATOR constraint needs
 * (`createValidator(g, 'User', 'u', 'u.age >= 0')`). It runs the *same* ISO
 * expression grammar as a real `WHERE` by parsing a minimal wrapper query and
 * lifting its predicate, so the validator path can never drift from `WHERE`.
 * Throws {@link GqlSyntaxError} (code `E_SYNTAX`) on an unparseable predicate,
 * or a predicate that smuggles in extra clauses (e.g. a trailing `RETURN`).
 */
export const parsePredicate = (src: string): Expr => {
  const parsed = parse(`MATCH (_v) WHERE ${src}`);

  // The MATCH wrapper always yields a linear query, never a transaction-control
  // command — narrow so the `.parts` access below is well-typed.
  if (isTxControl(parsed)) {
    throw new GqlSyntaxError('a validator predicate must be a single boolean expression', 0);
  }

  // Exactly one linear query, exactly one clause (the MATCH we wrapped it in) —
  // anything more means the predicate carried extra clauses/set-operators.
  const clause =
    parsed.parts.length === 1 && parsed.parts[0].clauses.length === 1
      ? parsed.parts[0].clauses[0]
      : undefined;

  if (clause?.kind !== 'match' || clause.where === undefined) {
    throw new GqlSyntaxError('a validator predicate must be a single boolean expression', 0);
  }

  return clause.where;
};
