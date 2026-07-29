# Friendfolio command handling

This document describes how Telegram commands and button callbacks travel through the bot and what each action reads or writes.

## Request path

1. Telegram sends a `POST` request to the `telegram_webhook` Firebase HTTPS function in `functions/main.py`.
2. The function compares Telegram's `X-Telegram-Bot-Api-Secret-Token` header with `TELEGRAM_WEBHOOK_SECRET`. Invalid requests receive HTTP `401`.
3. JSON updates are passed to `BotHandlers.handle_update()` in `functions/friendfolio/handlers.py`.
4. Text messages go to `_handle_message()`; inline-button presses go to `_handle_callback()`.
5. `/whoami` is available to everyone. Every other command requires an ID in `TELEGRAM_ALLOWED_USER_IDS` and a private Telegram chat.
6. Data operations are delegated to `FirestoreRegistry` in `functions/friendfolio/store.py`. Each user's data is isolated under `users/{telegram-user-id}`.
7. The handler sends or edits a Telegram message with the result. The webhook returns HTTP `200`, including when an internal error is logged, so Telegram does not retry the same update forever.

Bot commands may include the bot username, such as `/help@FriendfolioBot`. The username suffix is removed before routing, and command matching is case-insensitive.

## Command routing

| Command | Handler | Processing and result |
| --- | --- | --- |
| `/start` | Inline response | Sends the same HTML command list as `/help`. No database access. |
| `/help` | Inline response | Sends the command list. No database access. |
| `/whoami` | Inline response | Returns the sender's numeric Telegram user ID. This deliberately runs before the allowlist check so a new user can discover their ID. |
| `/friend <name>` | `_create_entity(..., "friend", ...)` | Validates a 1–120 character name, creates or updates `friends/{stable-name-hash}`, and reports whether it was newly created. |
| `/project <name>` | `_create_entity(..., "project", ...)` | Same flow as `/friend`, but writes to `projects/{stable-name-hash}`. |
| `/add [-debug] <note>` | `_add()` | Classifies and stages a proposal. With `-debug`, also shows raw DeepSeek traces and automatically traces later steering revisions. |
| `/friends` | `_list_entities(..., "friend")` | Reads up to 100 friends, sorted by name, and shows note counts and saved birthdays. |
| `/projects` | `_list_entities(..., "project")` | Reads up to 100 projects, sorted by name, and shows note counts. |
| `/show friend <name>` | `_show()` | Resolves the stable friend ID and displays the 25 most recent active notes plus the birthday, if present. |
| `/show project <name>` | `_show()` | Resolves the stable project ID and displays its 25 most recent active notes. |
| `/inbox` | `_inbox()` | Shows up to 20 active notes whose `target_type` is `uncategorized`, oldest first, with eight-character display IDs. |
| `/reclassify [-debug] <ID> [context]` | `_reclassify()` | Retries an inbox note, optionally enabling traces for classification and later steering. |
| `/followups` | `_followups()` | Lists up to 30 notes with `follow_up_status: pending`, ordered by follow-up time. Items without a date appear after dated items. |
| `/next` | `_next()` | Filters pending follow-ups to project targets, returning up to 30. |
| `/done <ID>` | `_done()` | Resolves a unique note-ID prefix and changes its follow-up status from `pending` to `done`, recording `completed_at`. |
| `/birthdays` | `_birthdays()` | Reads friends with `birthday_mm_dd`, sorts them by month/day and name, and displays the saved dates. |
| `/search <words>` | `_search()` | Uses DeepSeek to plan the search, ranks matching notes locally, then asks DeepSeek to synthesize a grounded answer from those matches. Falls back to raw results if answer generation fails. |
| `/confidence [0-100]` | `_confidence()` | Shows the current per-user threshold, or stores a new percentage. Proposals below it are sent to the uncategorized inbox. |

Unknown commands receive `Unknown command. Use /help.` Commands with missing or invalid arguments receive a usage message and do not write anything.

Debug mode belongs to the pending proposal: `-debug` must be the first `/add` or `/reclassify`
argument. It emits raw prompts, HTTP responses, and captured errors without including the API
authorization header. The proposal is still staged normally. Every later steering message
automatically emits revision traces and updates the staged proposal and Telegram preview.

## `/add` classification and approval

`/add` is the only normal command that sends text to DeepSeek.

1. The handler loads existing friend and project names from Firestore.
2. DeepSeek selects which existing entities the new note clearly concerns. This first pass receives
   names only and does not classify or rewrite the note.
3. The handler validates those names against Firestore and loads each selected friend or project's
   current note.
4. `DeepSeekClassifier.classify()` receives the new input and selected current-note context, then
   proposes `append`, `merge`, `replace`, or `delete` operations rather than rewriting the note.
   History remains stored locally and is not sent as context.
   A clearly named person or project that is not already in Firestore becomes a proposed new entity
   note; being absent from the existing-name list alone does not send it to the uncategorized inbox.
5. A second DeepSeek request audits the operations for missing details, broad paraphrasing,
   unsupported additions, unsafe deletion, and lost information. It returns the minimally corrected
   operation proposal.
6. Python validates every operation and applies it deterministically. `merge`, `replace`, and
   `delete` must identify exactly one existing line or multi-line block. The resulting full notes
   are validated as a `NoteProposal`; one input can affect several entities.
7. A deterministic proposal token is calculated from the Telegram user ID and update ID. This prevents a retried Telegram update from creating duplicate proposals.
8. The proposal is stored in `pending_actions/{token}` with `status: pending` and an expiry time, normally 24 hours.
9. The bot shows the proposed target, category, changed content lines, dates, confidence, and reason.
   Removed lines use `-` and added lines use `+`. Each changed block retains one surrounding context
   line so a value added to an empty section still displays its section heading; unrelated unchanged
   lines are omitted. **View full proposed note** sends the complete latest staged content in
   Telegram-safe chunks. The registry is unchanged until **Approve** is pressed.
   Every delta, including after steering a proposal, compares the latest proposed note against the
   current Firestore note. It therefore shows the final effect approval would have and omits
   intermediate proposal mistakes.

Ordinary information uses the `note` category. New friend notes use this standard profile template:

```text
Current events:

Upcoming events:

Hobbies/interests:

Siblings:

Birthday:

Likes:

Dislikes:

Relationship with family:
```

Within a section, Python normally stores facts as bullets. `append` creates a new bullet. `merge`
groups a related fact into one existing topic bullet while requiring the complete replacement to
preserve both old and new details. `replace` is reserved for corrections or state transitions.
`delete` requires an explicit removal or false-information instruction and removes only one
uniquely matched line or multi-line block. Every operation carries a verbatim `source_quote`.

Project and uncategorized notes remain concise and do not use the friend-profile sections. On
friend or project approval, the original `/add` text is appended to that entity's history with a
server timestamp, while DeepSeek's merged content is stored as the one mutable current note.
The friend profile sections are formatting inside the note's `content` string, not separate JSON
fields. If DeepSeek omits an expected section, the bot does not rebuild or otherwise change the
content. It places a warning before the normal proposal so the user can review, cancel, and retry
the same `/add` command when necessary. Section detection is case-insensitive.

Only explicitly scheduled reminders use the separate `follow_up` category. Follow-ups do not
replace the friend or project current note. Each follow-up requires a specific `follow_up_at`
date/time, which drives reminder notifications. `/followups` shows all pending follow-ups, while
`/next` retains a project-only follow-up view.

If DeepSeek fails, the bot creates a proposed uncategorized inbox note instead. This fallback still requires approval.
The confidence threshold defaults to 65%. DeepSeek is instructed to use it, and the bot also
enforces it locally by changing lower-confidence proposal items into uncategorized inbox items.
DeepSeek also converts relative time wording such as "today" or "tomorrow" to calendar dates.
Ages are stored with an as-of date; an age alone is never used to infer an exact birth date.

### Approve button

Callback data uses `proposal:approve:{token}`. `approve_pending()` runs one Firestore transaction that:

- verifies the proposal exists, is still pending, and has not expired;
- creates or updates any referenced friend/project;
- appends the original input to each affected friend or project's timestamped history;
- updates the one current note for each affected friend or project;
- creates scheduled follow-ups as separate notes so they can be completed independently;
- archives any older duplicate active notes for that entity and corrects its note count;
- stores follow-up timestamps in UTC and marks `follow_up` items as pending;
- stores a friend's birthday when one was proposed;
- archives the source inbox note during reclassification; and
- changes the proposal status to `approved`.

Pressing Approve again is safe: an already-decided proposal is reported but not written twice.
After a successful approval, the callback message identifies each note as saved or updated and
shows the target name and resulting content instead of only displaying an internal note ID.

### Cancel button

Callback data uses `proposal:cancel:{token}`. `cancel_pending()` changes a still-pending proposal to `cancelled`. No friend, project, or note is created. Expired and previously decided proposals are left unchanged.

## Follow-up reminder buttons

Daily reminder messages can contain these callbacks:

| Button | Callback data | Effect |
| --- | --- | --- |
| **Done** | `followup:done:{full-note-id}` | Changes `follow_up_status` from `pending` to `done` and records completion time. |
| **Snooze 1 day** | `followup:snooze:{full-note-id}` | Moves `follow_up_at` to 24 hours after the button is processed. |

Callback senders are checked against the same allowlist as commands. Telegram's original reminder message is edited to show the outcome.

## Scheduled notifications

The `daily_reminders` Firebase function runs every day at 09:00 in `Asia/Singapore`. For every allowed user it:

1. sends birthday reminders at the configured offsets, defaulting to 7 days, 1 day, and the birthday itself;
2. sends reminders for dated follow-ups that are due, including Done and Snooze buttons; and
3. sends one daily summary when the uncategorized inbox is not empty.

Every sent notification gets a deterministic entry in `notification_log`. The function checks this collection before sending, which prevents duplicate notifications if the scheduled function is retried on the same day.

## Firestore collections

```text
users/{telegram-user-id}
  friends/{stable-name-hash}
  projects/{stable-name-hash}
  notes/{deterministic-note-id}
  pending_actions/{proposal-token}
  notification_log/{notification-id}
```

Friend and project documents in `notes` use `record_type: note`, `history`, or `follow_up`. Each
entity has one mutable current note, any number of timestamped history entries, and any number of
independently completable follow-ups. Older friend records using `record_type: summary` remain
readable and are converted to `note` when next updated.

- Friend and project IDs are hashes of normalized names, so capitalization and repeated spaces do not create separate entities.
- Each friend or project has at most one active current note, while its history entries accumulate.
  Later approvals update the current note in place.
  Uncategorized inbox notes are not deduplicated.
- Telegram displays the first eight characters of note IDs. Commands accepting an ID require that prefix to match exactly one active note.
- Archived notes are omitted from normal lists and searches.
- Search uses DeepSeek to produce a structured plan, ranks notes locally, and sends only the
  matching notes back to DeepSeek for a grounded natural-language answer. Planning failures use a
  token-based local plan; answer-generation failures show the raw matching notes.
- Firestore client rules deny direct access; the deployed Firebase Admin SDK performs these operations.

## Local test environment

The project's Python virtual environment is stored in `functions/venv`. From the repository root,
activate it and run the unit tests with:

```bash
source functions/venv/bin/activate
python -m unittest discover -v
python -m compileall -q functions scripts tests
```

## Relevant source files

| File | Responsibility |
| --- | --- |
| `functions/main.py` | Firebase webhook, webhook-secret verification, dependency setup, and daily schedule. |
| `functions/friendfolio/handlers.py` | Command routing, validation, Telegram messages, button callbacks, and reminder orchestration. |
| `functions/friendfolio/deepseek.py` | DeepSeek request, JSON-mode prompt, retries, and proposal validation. |
| `functions/friendfolio/store.py` | Firestore paths, queries, transactions, note IDs, completion, snoozing, birthdays, and notification deduplication. |
| `functions/friendfolio/models.py` | Allowed proposal fields, categories, target types, and validation rules. |
| `functions/friendfolio/telegram_api.py` | Telegram Bot API calls for sending, editing, and answering callbacks. |
