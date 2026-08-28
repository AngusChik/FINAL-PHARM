# Development-first release workflow

The pharmacy uses two persistent Git worktrees from the same repository:

- **Development** is checked out on `development`, has its own development
  database/configuration, and is where changes are committed and tested. This
  branch remains local until its exact commit has passed the production release.
- **Production** is checked out on `main`, has its own `.env`, virtual
  environment, runtime state, logs, Caddy state, and production database.

Never point development at the production database. A second worktree separates
tracked code, but ignored files such as `.env`, `env`, `.runtime`, `logs`, and
`caddy_data` must also be independently provisioned in each root.

The production role is fail-closed. Only the production worktree has
`.runtime\production-role.json`; production settings, the production controller,
and the shortcut installer reject a development checkout even if it still has
an old `.env` file. Candidate release checks use an explicit, temporary role-root
override and do not authorize development for normal production use.

## One-time cutover

The first transition is deliberately two-phase because the current `main`
worktree may not contain the new startup installer until the first release.

1. Commit the complete workflow change and finish the isolated development-data
   setup. The setup controller refuses a dirty checkout or an unsafe development
   database.
2. While the original checkout is still the known production checkout, create a
   verified backup and stop its existing Waitress and Caddy processes. Confirm
   ports 8000 and 443 are closed. Do this before switching that checkout to
   `development` or copying Caddy/runtime data; the new production worktree must
   never inherit untracked live processes from the old root.
3. From the clean checkout, create the local development branch and provision
   the sibling production worktree without requesting shortcut installation:

   ```powershell
   setup-development-workflow.bat -CreateDevelopmentBranch
   ```

4. Run `check`, then perform the first confirmed `publish`. This promotes the
   exact tested development commit into the production worktree and then to
   GitHub `main` only after production health passes.
5. After that first release, rerun setup to install the staff shortcut from the
   now-current production worktree and migrate existing pharmacy scheduled-task
   actions away from development. Add `-EnableAutoStart` only when automatic
   sign-in/recovery startup is wanted:

   ```powershell
   setup-development-workflow.bat -InstallStartup -EnableAutoStart
   ```

6. Verify the `Pharmacy Production Startup`, `Pharmacy Scheduled Jobs`, and
   `Pharmacy Supplier Ordering` task actions all reference the production root,
   not the checkout that is now development. The migration leaves tasks that
   were never installed absent; install missing automation from
   `scripts\install-automation-task.ps1` in the production worktree when needed.

The normal staff shortcut is `Pharmacy`. It starts production through a hidden
launcher, so users do not work in a command-prompt window. `Pharmacy Admin
Control` intentionally opens the production controls for an administrator.
An explicit administrator stop is durable: scheduled `ensure` runs respect it,
while clicking the staff shortcut is an explicit request to start again.

## Shared workflow configuration

Development controls and the release engine discover production through this
gitignored file:

```text
.runtime\development-workflow.json
```

Schema version 1:

```json
{
  "schema_version": 1,
  "development_branch": "development",
  "production_branch": "main",
  "remote": "origin",
  "expected_origin_url": "https://github.com/AngusChik/FINAL-PHARM.git",
  "production_worktree": "C:\\Pharmacy\\FINAL-PHARM-PRODUCTION"
}
```

Command-line parameters override the configuration. If the file is absent, the
release engine looks for a sibling folder named `FINAL-PHARM-PRODUCTION` (or the
path in `PHARMACY_PRODUCTION_WORKTREE`).

Provisioning the production root is intentionally separate from publishing.
Before recording it in this file, verify that it is a registered worktree of
the same repository on `main`, has a clean checkout matching `origin/main`, and
has its own production `.env`, Python environment, Caddy files/certificates,
backup location, runtime directory, and logs. After the first release, shortcut
and task actions must also be verified against this production path. Never copy
`.env.development` into production; it is development-only configuration.

## Release commands

Run these from the development worktree:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\publish-release.ps1 -Action check
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\publish-release.ps1 -Action publish
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\publish-release.ps1 -Action status
```

`check` is non-deploying. It verifies clean committed branches, refreshes
`origin/main`, confirms local development contains main, confirms production
exactly matches `origin/main`, then runs the
development check, missing-migration detection, the candidate code's deployment
check against the production environment, a read-only deployment check using
the current production worktree and environment, the complete Django suite,
and the PowerShell contract tests. The production controller checks the
candidate again after the guarded fast-forward and before service health can
pass.

`publish` repeats the checks and requires typing the displayed commit-specific
confirmation. Automation that has already obtained explicit operator approval
may pass `-ConfirmRelease`. `-DryRun` skips mutating commands and prints a
non-deploying rehearsal; because it skips `git fetch`, its remote comparisons
use the currently cached `origin/main` ref.

After confirmation, publishing:

1. Creates an annotated local release tag, JSON manifest, and verified Git
   bundle under `.runtime\release-engine\releases`.
2. Stops production, then creates and records a verified final backup of the
   unchanged production database.
3. Fast-forwards the clean production `main`
   worktree to the exact development commit.
4. Starts through the existing production controller, which performs deployment
   checks, migrations, static collection, and another pre-start backup.
5. Requires Waitress, Caddy, Django/database health, and HTTPS health.
6. Only after production is healthy, atomically pushes `main` and the release
   tag to `origin`.

The publisher and production controller share the exclusive
`.runtime\production-release.lock`. The publisher holds this operating-system
lock continuously from the final production snapshot through stop, backup,
code promotion, startup, health verification, durable synchronization intent,
and the first atomic Git synchronization attempt. Desktop shortcuts, sign-in
`ensure` tasks, and other state-changing production commands therefore cannot
restart or alter production in the middle of a release. Publisher child calls
receive a per-release GUID recorded in
`.runtime\production-release.owner.json`; the controller accepts that token
only while the matching OS lock is actively held, and still serializes each
child through the normal production-control lock. The owner metadata is removed
and the OS lock is released at the end of the transaction, including failure
paths.

Before the first stop, the publisher also writes
`.runtime\production-recovery-required.json`. Scheduled startup remains blocked
while that durable journal exists. It is cleared automatically only after the
candidate release is healthy or the previous release has been fully restored
and proven healthy. If automatic rollback cannot safely complete, leave
production stopped and use `Pharmacy Admin Control` to inspect the recovery
record. Clearing it is an interactive administrator action that requires the
tracked processes and ports to be stopped and the exact release ID to be typed.

If deployment fails before health is proven, the engine restores the recorded
final database backup, resets code to the previous clean production commit, and
restarts the previous release. The manifest records recovery success or failure
with whether database restoration was required, a completion time, and recovery
notes. A rollback is marked `healthy` only after the previous release passes the
full production health probe. If code or database restoration is incomplete,
production is left stopped rather than starting a mixed state.

After production health passes, synchronization intent is written to
`.runtime\release-engine\sync-pending.json` while the production release lock is
still held and before the recovery journal is cleared. The initial atomic push
then runs under the same lock. This ordering also makes a process or machine
failure in the health-to-push window resumable.

If the final push fails, healthy production remains on the released commit and
the pending record remains. `status` reports the release and error. Running
`publish` again verifies the exact live commit, tag, and health under the lock,
then retries only the atomic Git synchronization; it does not redeploy or rerun
migrations. Pending state must match the configured production path,
`origin/main`, release ID, full commit, tag, and contained manifest; a corrupt or
redirected record fails before any command or mutation.
