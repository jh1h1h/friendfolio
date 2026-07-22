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
| `/add <note>` | `_add()` | Validates a maximum of 5,000 characters, asks DeepSeek to classify the note, stages the proposal in `pending_actions`, and displays **Approve** and **Cancel** buttons. It does not create registry notes until approval. |
| `/friends` | `_list_entities(..., "friend")` | Reads up to 100 friends, sorted by name, and shows note counts and saved birthdays. |
| `/projects` | `_list_entities(..., "project")` | Reads up to 100 projects, sorted by name, and shows note counts. |
| `/show friend <name>` | `_show()` | Resolves the stable friend ID and displays the 25 most recent active notes plus the birthday, if present. |
| `/show project <name>` | `_show()` | Resolves the stable project ID and displays its 25 most recent active notes. |
| `/inbox` | `_inbox()` | Shows up to 20 active notes whose `target_type` is `uncategorized`, oldest first, with eight-character display IDs. |
| `/reclassify <ID> [context]` | `_reclassify()` | Resolves a unique note-ID prefix, confirms that it is an active inbox note, asks DeepSeek to classify it again, then stages another approval proposal. Approval archives the old inbox note and creates the replacement note or notes atomically. |
| `/followups` | `_followups()` | Lists up to 30 notes with `follow_up_status: pending`, ordered by follow-up time. Items without a date appear after dated items. |
| `/next` | `_next()` | Filters pending notes to project notes categorized as `next_action`, returning up to 30. |
| `/done <ID>` | `_done()` | Resolves a unique note-ID prefix and changes its follow-up status from `pending` to `done`, recording `completed_at`. |
| `/birthdays` | `_birthdays()` | Reads friends with `birthday_mm_dd`, sorts them by month/day and name, and displays the saved dates. |
| `/search <words>` | `_search()` | Performs a case-insensitive substring scan of active note content, sorts newest first, and returns up to 20 matches. |

Unknown commands receive `Unknown command. Use /help.` Commands with missing or invalid arguments receive a usage message and do not write anything.

## `/add` classification and approval

`/add` is the only normal command that sends text to DeepSeek.

1. The handler loads existing friend and project names from Firestore.
2. `DeepSeekClassifier.classify()` sends the note, entity names, local date/time, and expected JSON schema to DeepSeek.
3. The returned JSON is validated as a `NoteProposal`. One input can become multiple proposed notes when it refers to several friends or projects.
4. A deterministic proposal token is calculated from the Telegram user ID and update ID. This prevents a retried Telegram update from creating duplicate proposals.
5. The proposal is stored in `pending_actions/{token}` with `status: pending` and an expiry time, normally 24 hours.
6. The bot shows the proposed target, category, content, dates, confidence, and reason. The registry is unchanged until **Approve** is pressed.

If DeepSeek fails, the bot creates a proposed uncategorized inbox note instead. This fallback still requires approval.

### Approve button

Callback data uses `proposal:approve:{token}`. `approve_pending()` runs one Firestore transaction that:

- verifies the proposal exists, is still pending, and has not expired;
- creates or updates any referenced friend/project and increments its note count;
- saves each proposed note under a deterministic note ID;
- stores follow-up timestamps in UTC and marks dated follow-ups or `next_action` notes as pending;
- stores a friend's birthday when one was proposed;
- archives the source inbox note during reclassification; and
- changes the proposal status to `approved`.

Pressing Approve again is safe: an already-decided proposal is reported but not written twice.

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

- Friend and project IDs are hashes of normalized names, so capitalization and repeated spaces do not create separate entities.
- Telegram displays the first eight characters of note IDs. Commands accepting an ID require that prefix to match exactly one active note.
- Archived notes are omitted from normal lists and searches.
- Firestore client rules deny direct access; the deployed Firebase Admin SDK performs these operations.

## Relevant source files

| File | Responsibility |
| --- | --- |
| `functions/main.py` | Firebase webhook, webhook-secret verification, dependency setup, and daily schedule. |
| `functions/friendfolio/handlers.py` | Command routing, validation, Telegram messages, button callbacks, and reminder orchestration. |
| `functions/friendfolio/deepseek.py` | DeepSeek request, JSON-mode prompt, retries, and proposal validation. |
| `functions/friendfolio/store.py` | Firestore paths, queries, transactions, note IDs, completion, snoozing, birthdays, and notification deduplication. |
| `functions/friendfolio/models.py` | Allowed proposal fields, categories, target types, and validation rules. |
| `functions/friendfolio/telegram_api.py` | Telegram Bot API calls for sending, editing, and answering callbacks. |
