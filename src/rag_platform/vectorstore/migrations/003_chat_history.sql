-- Chat history for the Streamlit UI.
-- Run once in Supabase: SQL Editor -> New query -> paste -> Run.
--
-- Access model: the Streamlit server talks to these tables with the key in
-- its secrets. Row Level Security is ON with no policies, so the public
-- anon key (which can be exposed in browsers) cannot read or write chats.
-- Only the service-role key works, and it bypasses RLS.
-- => Put your SERVICE ROLE key in Streamlit secrets as SUPABASE_KEY
--    (it stays on Streamlit's server and is never sent to visitors).

create table if not exists public.conversations (
    id          uuid primary key default gen_random_uuid(),
    visitor_id  text not null,
    title       text not null default 'New chat',
    mode        text not null default 'Knowledge Base',
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

create index if not exists conversations_visitor_updated_idx
    on public.conversations (visitor_id, updated_at desc);

create table if not exists public.messages (
    id               uuid primary key default gen_random_uuid(),
    conversation_id  uuid not null references public.conversations(id) on delete cascade,
    role             text not null check (role in ('user', 'assistant')),
    content          text not null,
    sources          jsonb not null default '[]'::jsonb,
    meta             jsonb not null default '{}'::jsonb,
    created_at       timestamptz not null default now()
);

create index if not exists messages_conversation_created_idx
    on public.messages (conversation_id, created_at);

alter table public.conversations enable row level security;
alter table public.messages enable row level security;

-- Verify (should return two rows):
-- select table_name from information_schema.tables
-- where table_schema = 'public' and table_name in ('conversations', 'messages');
