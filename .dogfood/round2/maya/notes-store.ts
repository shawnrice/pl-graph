// Typed store wrapper over the wasm backend — the "browser" code path exercised
// headlessly. Loads the .wasm bytes, builds a RustGraph + reactive Store, and
// exposes typed live queries + mutations for the note app.

import { readFile } from 'node:fs/promises';

import {
  createStore,
  graphFromNdjson,
  type Store,
} from '@lenke/native';
import { createWasmBackend } from '@lenke/native/wasm';

import { BACKLINKS, TAG_COUNTS, seedNdjson, type Backlink, type TagCount } from './data.ts';

const WASM_PATH =
  '/home/shawn/projects/pl-graph/crates/lenke-core/target/wasm32-unknown-unknown/release/lenke_core.wasm';

export async function createNotesStore(): Promise<Store> {
  // In a real browser this would be `createWasmBackend(fetch('/lenke_core.wasm'))`.
  const backend = await createWasmBackend(await readFile(WASM_PATH));
  const graph = graphFromNdjson(backend, seedNdjson());
  return createStore(graph);
}

export function backlinksOf(store: Store, id: string) {
  return store.liveQuery<Backlink>(BACKLINKS, { deps: ['Note', 'LINKS_TO'], params: { id } });
}

export function tagCounts(store: Store) {
  return store.liveQuery<TagCount>(TAG_COUNTS, { deps: ['Note', 'Tag', 'TAGGED'] });
}

// A mutation: create a note and link it to `to`. GQL DML through store.mutate,
// which only notifies subscribers if the graph version actually moved.
export function addLinkingNote(
  store: Store,
  note: { id: string; title: string },
  to: string,
): void {
  store.mutate((g) => {
    g.query('INSERT (n:Note {id: $id, title: $title, body: $body})', {
      id: note.id,
      title: note.title,
      body: `${note.title} body.`,
    });
    g.query('MATCH (n:Note {id: $id}), (m:Note {id: $to}) INSERT (n)-[:LINKS_TO]->(m)', {
      id: note.id,
      to,
    });
  });
}

// A mutation: tag an existing note.
export function tagNote(store: Store, noteId: string, tag: string): void {
  store.mutate((g) => {
    // Tag vertex may not exist yet — MERGE-style via OPTIONAL then INSERT if missing.
    const existing = g.query<{ n: unknown }>('MATCH (t:Tag {name: $name}) RETURN t.name AS n', {
      name: tag,
    });
    if (existing.length === 0) {
      g.query('INSERT (:Tag {name: $name})', { name: tag });
    }
    g.query('MATCH (n:Note {id: $id}), (t:Tag {name: $name}) INSERT (n)-[:TAGGED]->(t)', {
      id: noteId,
      name: tag,
    });
  });
}
