-- Anonymous, append-only event storage for assessment schema 1.0.
-- Run in the Supabase SQL editor as the project owner.

create table if not exists public.assessment_events (
  id uuid primary key,
  received_at timestamptz not null default now(),
  run_id uuid not null,
  participant_id uuid not null,
  phase text not null check (phase in ('pre', 'post')),
  event_type text not null check (event_type in ('started', 'completed')),
  venue text not null check (venue in ('NYU', 'UMN', 'TAU', 'ONLINE')),
  assessment_version text not null check (char_length(assessment_version) between 1 and 32),
  workshop_git_sha text not null check (workshop_git_sha = 'development' or workshop_git_sha ~ '^[0-9a-f]{40}$'),
  consent_version text not null check (char_length(consent_version) between 1 and 32),
  payload jsonb not null check (jsonb_typeof(payload) = 'object'),
  constraint assessment_event_payload_size check (pg_column_size(payload) <= 65536),
  constraint completed_event_has_answers check (
    event_type = 'started' or jsonb_typeof(payload #> '{answers,knowledge}') = 'array'
  )
);

create index if not exists assessment_events_run_id_idx on public.assessment_events (run_id);
create index if not exists assessment_events_received_at_idx on public.assessment_events (received_at);
create index if not exists assessment_events_version_idx on public.assessment_events (assessment_version, phase);

alter table public.assessment_events enable row level security;
alter table public.assessment_events force row level security;

revoke all on table public.assessment_events from public;
revoke all on table public.assessment_events from anon;
revoke all on table public.assessment_events from authenticated;
grant insert on table public.assessment_events to anon;

drop policy if exists "anonymous assessment insert only" on public.assessment_events;
create policy "anonymous assessment insert only"
on public.assessment_events
for insert
to anon
with check (
  payload ? 'assessment_schema_version'
  and payload ? 'question_bank_hash'
  and not (payload ?| array[
    'name', 'email', 'phone', 'username', 'department', 'lab', 'age', 'gender',
    'career_stage', 'geolocation', 'latitude', 'longitude', 'ip', 'user_agent',
    'screen', 'referrer', 'timezone', 'advertising_id', 'analytics_id'
  ])
);

comment on table public.assessment_events is
  'Anonymous workshop assessment events. Research exports use completed events; provider request logs are not study data.';
comment on column public.assessment_events.received_at is
  'Canonical UTC-capable server timestamp assigned by PostgreSQL.';
