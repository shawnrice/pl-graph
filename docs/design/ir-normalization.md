# Predicate normalization before index seeding

Status: **not built, deliberately deferred.** This records why it is the right
shape and what has to exist before it is safe to attempt.

## The problem it solves

Four index-seeding gaps have been found and fixed, and all four were the same
bug:

| spelling                                                           | cost vs. the seeking form |
| ------------------------------------------------------------------ | ------------------------- |
| a clause `WHERE u.k = $x` before a traversal, vs. inline `{k: $x}` | 60x                       |
| `u.k IN [$a, $b]`                                                  | 220x                      |
| `$x = u.k` (constant first)                                        | 107x                      |
| `u.k = $a OR u.k = $b`                                             | 220x                      |
| `5 <= u.n AND 9 >= u.n` (grouped path)                             | 197x                      |

Every one returned the **correct answer**, so no correctness test could catch
them, and every one was found by hand or by the equivalence test that now exists.

They share a cause. `prop_index_hint` pattern-matches on the _surface shape_ of
the compiled expression: it looks for a `Compare` whose left operand is a `Prop`,
then separately for an `And` of those, then separately for an `In`, then
separately for an `Or`. Each new spelling needs its own arm, arms drift out of
sync — the fix for the single-comparison operand order did not fix the grouped
path, which is separate code — and nobody can enumerate the spellings that are
still missing.

## What it is NOT

It is not a new IR. The engine already lowers the AST to `CExpr` in `plan.rs`,
and that is what the planner reads. Adding another layer would not help.

## What it is

A **normalization pass over the existing `CExpr`**, run once at plan time, that
rewrites equivalent predicates into one canonical form before the recognizer sees
them:

```
$x = u.k                  ->  u.k = $x          (constant to the right)
5 <= u.n                  ->  u.n >= 5          (operator flipped with operands)
u.k = $a OR u.k = $b      ->  u.k IN [$a, $b]   (same key, all equality)
(a OR b) OR c             ->  a OR b OR c       (flatten)
u.k IN [$a]               ->  u.k = $a          (singleton)
```

The recognizer then handles canonical forms only. It gets **smaller** while
covering **more** spellings, which is the opposite of the trajectory it is on
now.

## Cost

Effectively free where it matters. Normalization runs once per plan, and prepared
plans are cached, so the cost lands on parse rather than execution. The current
approach pays its cost per _execution_ — every arm the recognizer tries against
every predicate, on every query.

## Why it is deferred

A rewrite pass that changes what the planner sees is exactly the kind of change
that silently returns wrong rows: fold a disjunction wrongly and matches vanish;
flip an operator wrongly and you get the complement. Both failure modes are
invisible to a rate check and to most correctness tests.

The safety net is `equivalent_spellings_cost_the_same` in
`gql/index_seed_tests.rs`, which asserts that groups of equivalent queries return
identical rows and run within a factor of each other. That test is what makes the
refactor checkable rather than scary — it already caught one gap inside the fix
for another.

**Before attempting this, broaden those groups.** Seven today, all predicate
forms. It wants path shapes, quantifier spellings, and the negations
(`NOT IN`, `<>`, `IS NOT NULL`) whose whole point is that they must NOT seed —
a normalization that helpfully "simplifies" one of those into a seekable form is
the most likely way this goes wrong.

## Related

`starts_with` still scans, and that one is genuinely different: a missing
prefix-range seek is a feature, not an unrecognized spelling. Normalization would
not help it.
