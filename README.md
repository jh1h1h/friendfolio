# Friendfolio Bot — Firebase + DeepSeek

A private Telegram bot for keeping lightweight notes about friends and personal/work projects.
This version is serverless: Telegram calls a Firebase HTTPS function, Firestore stores the registry,
a scheduled function sends reminders, and DeepSeek V4 Flash classifies `/add` messages.

No OpenAI or ChatGPT API is used by this version.

## Architecture

```text
Telegram -> HTTPS webhook -> Firebase Function -> DeepSeek API
                                  |
                                  +-> Firestore

Cloud Scheduler -> daily_reminders Function -> Telegram
```

Firebase Hosting is not needed; it hosts websites, whereas Telegram calls the function URL directly.

## Features

- Each friend and project has an append-only timestamped history and one mutable current note.
- Friend notes use a structured profile template; project notes remain concise and free-form.
- Durable time references: relative dates and ages are anchored to explicit calendar dates.
- Project notes and scheduled project follow-ups.
- Two-stage DeepSeek `/add`: resolve related entities, load their saved context, then propose a
  consolidated update with explicit **Approve**/**Cancel** buttons.
- Atomic, idempotent Firestore approval: retrying a webhook cannot duplicate approved notes.
- DeepSeek-assisted semantic `/search` over notes and entities.
- Uncategorized inbox with AI reclassification.
- Daily birthday, overdue follow-up and uncategorized-inbox notifications.
- Telegram user allowlist and private-chat restriction.
- Webhook secret validation.
- Firestore client access denied by default; only the Firebase Admin SDK can access the registry.
- Firebase emulator configuration and unit tests.

## Before deploying: cost, accounts and privacy

Cloud Functions deployment requires Firebase's **Blaze** billing plan. Firestore and Functions both
have no-cost usage allowances, and a single-person bot will usually be tiny, but attaching billing
means charges are possible. Configure Google Cloud budget alerts. Budget alerts notify you; they do
not automatically cap spending. If you are not the billing-account owner, ask the account owner to
configure Blaze and the budget rather than attaching payment details without permission.

DeepSeek is much cheaper than the OpenAI model previously configured, but it is still a paid API
requiring a DeepSeek API key and balance. This project defaults to `deepseek-v4-flash`; DeepSeek says
the older `deepseek-chat` name will be retired on 24 July 2026.

There is an important privacy trade-off. Your `/add` text and existing friend/project names are sent
to DeepSeek. DeepSeek's February 2026 privacy policy says inputs may be collected and used to improve
or train its technology, and that personal data is processed and stored in the People's Republic of
China. Do not put secrets, health information, confidential work information, or sensitive facts
about friends into this bot unless that data handling is acceptable to everyone involved. DeepSeek's
terms also contain requirements for users under 18; review them with a parent or guardian where
applicable.

Sources:

- [Firebase pricing plans](https://firebase.google.com/docs/projects/billing/firebase-pricing-plans)
- [Firestore no-cost quotas](https://firebase.google.com/docs/firestore/quotas)
- [DeepSeek models and current pricing](https://api-docs.deepseek.com/quick_start/pricing/)
- [DeepSeek JSON output](https://api-docs.deepseek.com/guides/json_mode/)
- [DeepSeek privacy policy](https://cdn.deepseek.com/policies/en-US/deepseek-privacy-policy.html)

## Prerequisites

- Python 3.12
- Node.js 20 or newer
- A Telegram bot token from `@BotFather`
- A Firebase project on the Blaze plan
- A Firestore database
- A DeepSeek API key
- Firebase CLI: `npm install -g firebase-tools`

## 1. Create and select the Firebase project

1. Create a project in the Firebase console.
2. Upgrade it to Blaze and create conservative budget alerts.
3. Create a Firestore database in Native mode. Singapore is the natural region for this project's
   default `asia-southeast1` Functions deployment.
4. Log in and select the project:

   ```bash
   firebase login
   cp .firebaserc.example .firebaserc
   # Replace the project ID inside .firebaserc
   firebase use your-project-id
   ```

## 2. Configure non-secret settings

```bash
cp functions/.env.example functions/.env
```

Edit `functions/.env`:

```dotenv
TELEGRAM_ALLOWED_USER_IDS=0
APP_TIMEZONE=Asia/Singapore
DEEPSEEK_MODEL=deepseek-v4-flash
BIRTHDAY_REMINDER_DAYS=7,1,0
PENDING_EXPIRY_HOURS=24
```

If you do not know your Telegram numeric ID, initially leave it as `0`. After deployment, `/whoami`
works for unauthorised users; update the value and redeploy.

## 3. Store secrets

Generate a webhook secret locally:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Then store all three values in Google Secret Manager through the Firebase CLI:

```bash
firebase functions:secrets:set TELEGRAM_BOT_TOKEN
firebase functions:secrets:set TELEGRAM_WEBHOOK_SECRET
firebase functions:secrets:set DEEPSEEK_API_KEY
```

Use the same webhook-secret value again in step 5. Do not put these values in `functions/.env`.

## 4. Deploy Firestore rules and Functions

```bash
firebase deploy --only firestore,functions
```

This deploys:

- `telegram_webhook` in Singapore with a 60-second timeout and maximum three instances.
- `daily_reminders`, scheduled for 09:00 Asia/Singapore.
- Firestore rules that deny all direct client access.

## 5. Register the Telegram webhook

The helper keeps tokens out of command-line arguments, but they are temporarily placed in the
current shell environment:

```bash
python -m venv functions/venv
source functions/venv/bin/activate
pip install httpx

export TELEGRAM_BOT_TOKEN='your-token'
export TELEGRAM_WEBHOOK_SECRET='the-same-random-value-from-step-3'
python scripts/set_webhook.py --project-id your-project-id

unset TELEGRAM_BOT_TOKEN TELEGRAM_WEBHOOK_SECRET
```

Message the bot `/whoami`. If `TELEGRAM_ALLOWED_USER_IDS` was `0`, update `functions/.env` with the
returned ID and deploy Functions again:

```bash
firebase deploy --only functions
```

## Commands

| Command | Purpose |
| --- | --- |
| `/add [-debug] <note>` | Classify a note, optionally enabling traces for it and later revisions |
| `/friend <name>` | Explicitly create a friend |
| `/project <name>` | Explicitly create a project |
| `/friends`, `/projects` | List entities and note counts |
| `/show friend <name>` | Show one friend's recent notes |
| `/show project <name>` | Show one project's recent notes |
| `/next` | Show pending project follow-ups |
| `/followups` | Show all pending follow-ups |
| `/done <ID>` | Complete an item using its displayed eight-character ID |
| `/inbox` | Show uncategorized notes |
| `/reclassify [-debug] <ID> [context]` | Retry an inbox note, optionally enabling proposal traces |
| `/birthdays` | Show birthdays |
| `/search <words>` | Search notes, then have DeepSeek synthesize a grounded answer |
| `/confidence [0-100]` | View or set the per-user classification confidence threshold |
| `/whoami` | Show your Telegram user ID |

After `/add`, plain text revises the pending proposal. If the proposal began with `/add -debug`,
all later steering automatically emits raw revision traces while continuing to update the proposal.

## Firestore structure

All personal data is isolated beneath the Telegram user ID:

```text
users/{telegram-user-id}
  friends/{stable-name-hash}
  projects/{stable-name-hash}
  notes/{deterministic-note-id}  # current notes, history entries, follow-ups
  pending_actions/{proposal-token}
  notification_log/{notification-id}
```

Approving a proposal writes its entities, notes and approval status in one Firestore transaction.
For a friend or project, approval appends the original `/add` input as a timestamped history record
and updates that entity's current note. Friend notes use the structured profile template, while
project notes stay free-form. Scheduled follow-ups and uncategorized inbox entries remain separate.
The approval confirmation says whether each note was saved or updated and displays its new content.
The proposal itself is staged before approval so the buttons remain usable across cold starts, but
no friend/project/note registry record is created until approval.

## Local checks

Install dependencies into a virtual environment:

```bash
python -m venv functions/venv
source functions/venv/bin/activate
pip install -r functions/requirements.txt
python -m unittest discover -v
python -m compileall -q functions scripts tests
```

For emulator testing, copy the local templates and start Firestore and Functions:

```bash
cp functions/.env.example functions/.env.local
cp functions/.secret.local.example functions/.secret.local
firebase emulators:start --only functions,firestore
```

The Telegram service cannot reach localhost directly. Use a secure development tunnel only for
temporary webhook testing, and replace the webhook with the deployed function URL afterwards.

## Operational notes

- Check `firebase functions:log` if Telegram receives no response.
- DeepSeek JSON mode guarantees valid JSON, but not this application's exact schema; Pydantic
  validates the response and the bot stages an uncategorized fallback if classification fails.
- Search and reminder scans intentionally favor simplicity for a small personal registry. If the
  registry grows to thousands of notes, add indexed Firestore queries instead of scanning.
- Scheduled Firebase functions can run more than once. Notification IDs and proposal/note IDs are
  deterministic, limiting duplicate effects.
