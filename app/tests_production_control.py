from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase


class ProductionControlSourceTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = (
            Path(settings.BASE_DIR) / "scripts" / "production.ps1"
        ).read_text(encoding="utf-8")

    def test_taskkill_stderr_cannot_abort_restart_before_verification(self):
        stop_start = self.source.index("function Stop-TrackedProcessTree")
        stop_end = self.source.index("function Wait-TcpPortClosed", stop_start)
        stop_source = self.source[stop_start:stop_end]

        self.assertIn('$ErrorActionPreference = "Continue"', stop_source)
        self.assertIn("& taskkill.exe /PID $ProcessId /T /F 2>&1", stop_source)
        self.assertIn("Stop-Process -Id $treeProcessId", stop_source)
        self.assertIn("Get-Process -Id $ProcessId", stop_source)
        self.assertIn("Could not stop tracked production process", stop_source)

    def test_stop_verifies_ports_before_discarding_process_state(self):
        stop_start = self.source.index("function Stop-Production")
        stop_end = self.source.index("function Show-Status", stop_start)
        stop_source = self.source[stop_start:stop_end]

        port_8000 = stop_source.index("Wait-TcpPortClosed 8000")
        port_443 = stop_source.index("Wait-TcpPortClosed 443")
        remove_state = stop_source.index("Remove-Item -LiteralPath $pidFile")
        self.assertLess(port_8000, remove_state)
        self.assertLess(port_443, remove_state)

    def test_pid_reuse_is_fenced_by_identity_and_start_time(self):
        self.assertIn('$process.ProcessName -notin $allowedNames', self.source)
        self.assertIn('waitress_started_at = $waitressProcess.StartTime', self.source)
        self.assertIn('caddy_started_at = $caddyProcess.StartTime', self.source)
        self.assertIn('[Math]::Abs(($actualStart - $expectedStart).TotalSeconds)', self.source)

    def test_access_denied_shutdown_elevates_only_exact_taskkill(self):
        self.assertIn('function Invoke-ElevatedProductionStop', self.source)
        self.assertIn('-Verb RunAs', self.source)
        self.assertIn('$detail -match "Access is denied"', self.source)
        self.assertIn('@("/PID", "$ProcessId", "/T", "/F")', self.source)
        self.assertIn("Invoke-ElevatedProductionStop $ProcessId", self.source)
        self.assertNotIn("ElevatedRetry", self.source)

    def test_control_lock_is_exclusive_and_always_released(self):
        lock_start = self.source.index("function Invoke-WithProductionControlLock")
        lock_end = self.source.index("function Test-ReleaseGateAuthorization", lock_start)
        lock_source = self.source[lock_start:lock_end]

        self.assertIn("[IO.FileShare]::None", lock_source)
        self.assertIn("Another production control operation is already in progress", lock_source)
        self.assertIn("finally", lock_source)
        self.assertIn("$lockStream.Dispose()", lock_source)

        dispatch_start = self.source.index("$configuration = Read-DotEnv")
        dispatch_source = self.source[dispatch_start:]
        self.assertIn('"ensure" {', dispatch_source)
        self.assertIn("Invoke-WithProductionMutationLocks", dispatch_source)
        self.assertIn("-ReleaseGateTimeoutMilliseconds 1200000", dispatch_source)
        self.assertIn("-ControlTimeoutMilliseconds 600000", dispatch_source)
        self.assertNotIn("Invoke-WithProductionControlLock", dispatch_source)
        stop_start = dispatch_source.index('"stop" {')
        stop_end = dispatch_source.index('"status" {', stop_start)
        self.assertIn(
            "Invoke-WithProductionMutationLocks",
            dispatch_source[stop_start:stop_end],
        )

    def test_release_gate_requires_matching_guid_and_active_os_lock(self):
        self.assertIn('[string]$ReleaseToken = ""', self.source)
        self.assertIn('"production-release.lock"', self.source)
        self.assertIn('"production-release.owner.json"', self.source)

        auth_start = self.source.index("function Test-ReleaseGateAuthorization")
        auth_end = self.source.index("function Invoke-WithProductionReleaseGate", auth_start)
        auth_source = self.source[auth_start:auth_end]

        self.assertGreaterEqual(auth_source.count("[Guid]::TryParse"), 2)
        self.assertIn('$owner.PSObject.Properties["release_token"]', auth_source)
        self.assertIn("if ($ownerGuid -ne $tokenGuid) { return $false }", auth_source)
        self.assertIn("[IO.FileMode]::Open", auth_source)
        self.assertIn("[IO.FileShare]::None", auth_source)
        self.assertIn("$nativeError = $_.Exception.HResult -band 0xFFFF", auth_source)
        self.assertIn("return @(32, 33) -contains $nativeError", auth_source)
        self.assertIn("$probeStream.Dispose()", auth_source)

    def test_release_token_bypasses_only_outer_gate(self):
        gate_start = self.source.index("function Invoke-WithProductionReleaseGate")
        gate_end = self.source.index("function Invoke-WithProductionMutationLocks", gate_start)
        gate_source = self.source[gate_start:gate_end]

        token_check = gate_source.index("Test-ReleaseGateAuthorization $ReleaseToken")
        operation = gate_source.index("& $Operation", token_check)
        release_lock = gate_source.index("[IO.File]::Open", operation)
        self.assertLess(token_check, operation)
        self.assertLess(operation, release_lock)
        self.assertIn("invalid or its release lock is not active", gate_source)

        mutation_start = gate_end
        mutation_end = self.source.index("function Test-IsAdministrator", mutation_start)
        mutation_source = self.source[mutation_start:mutation_end]
        outer = mutation_source.index("Invoke-WithProductionReleaseGate")
        inner = mutation_source.index("Invoke-WithProductionControlLock")
        self.assertLess(outer, inner)
        self.assertIn("$operationToRun = $Operation", mutation_source)
        self.assertIn("& $operationToRun", mutation_source)
        self.assertIn("Assert-ProductionRole", mutation_source)

    def test_production_role_marker_fences_every_mutating_control(self):
        self.assertIn('"production-role.json"', self.source)
        role_start = self.source.index("function Assert-ProductionRole")
        role_end = self.source.index("function Invoke-WithProductionMutationLocks", role_start)
        role_source = self.source[role_start:role_end]

        for required_property in (
            "schema_version",
            "role",
            "worktree",
            "branch",
            "remote",
            "created_at",
        ):
            self.assertIn(f'"{required_property}"', role_source)
        self.assertIn('$schemaVersion -ne 1', role_source)
        self.assertIn('[string]$roleMarker.role -cne "production"', role_source)
        self.assertIn('[string]$roleMarker.branch -cne "main"', role_source)
        self.assertIn('[string]$roleMarker.remote -cne "origin"', role_source)
        self.assertIn("[IO.Path]::IsPathRooted", role_source)
        self.assertIn("[StringComparer]::OrdinalIgnoreCase.Equals", role_source)
        self.assertIn("symbolic-ref --quiet --short HEAD", role_source)
        self.assertIn('$actualBranch -cne "main"', role_source)

        mutation_start = role_end
        mutation_end = self.source.index("function Test-IsAdministrator", mutation_start)
        mutation_source = self.source[mutation_start:mutation_end]
        role_check = mutation_source.index("Assert-ProductionRole")
        operation = mutation_source.index("& $operationToRun")
        self.assertLess(role_check, operation)

    def test_all_admin_menu_mutations_use_both_production_locks(self):
        menu_start = self.source.index("function Show-ProductionMenu")
        menu_end = self.source.index("$configuration = Read-DotEnv", menu_start)
        menu_source = self.source[menu_start:menu_end]

        self.assertEqual(menu_source.count("Invoke-WithProductionMutationLocks"), 5)
        self.assertNotIn("Invoke-WithProductionControlLock", menu_source)

    def test_all_normal_mutating_actions_use_both_production_locks(self):
        dispatch_start = self.source.index("$configuration = Read-DotEnv")
        dispatch_source = self.source[dispatch_start:]
        next_action = {
            "ensure": "start",
            "start": "stop",
            "stop": "status",
            "update": "restart",
            "restart": "logs",
        }
        for action, following in next_action.items():
            with self.subTest(action=action):
                start = dispatch_source.index(f'"{action}" {{')
                end = dispatch_source.index(f'"{following}" {{', start)
                action_source = dispatch_source[start:end]
                self.assertIn("Invoke-WithProductionMutationLocks", action_source)

        backup_start = dispatch_source.index('"backup" {')
        backup_source = dispatch_source[backup_start:]
        self.assertIn("Invoke-WithProductionMutationLocks", backup_source)
        clear_start = dispatch_source.index('"clear-recovery-block" {')
        clear_source = dispatch_source[clear_start:]
        self.assertIn("Invoke-WithProductionMutationLocks", clear_source)

    def test_recovery_marker_blocks_every_start_path_but_not_diagnostics(self):
        self.assertIn('"production-recovery-required.json"', self.source)
        block_start = self.source.index("function Get-ProductionRecoveryBlock")
        block_end = self.source.index("function Clear-ProductionRecoveryBlock", block_start)
        block_source = self.source[block_start:block_end]
        self.assertIn('Properties.Name -contains "release_id"', block_source)
        self.assertIn("invalid recovery marker", block_source)
        recovery_guard = block_source.index("function Assert-ProductionRecoveryCleared")
        guarded_source = block_source[recovery_guard:]
        token_authorization = guarded_source.index(
            "$ReleaseToken -and (Test-ReleaseGateAuthorization $ReleaseToken)"
        )
        marker_read = guarded_source.index("$block = Get-ProductionRecoveryBlock")
        self.assertLess(token_authorization, marker_read)

        start_start = self.source.index("function Start-Production")
        start_end = self.source.index("function Ensure-Production", start_start)
        start_source = self.source[start_start:start_end]
        self.assertLess(
            start_source.index("Assert-ProductionRecoveryCleared"),
            start_source.index("Get-ProductionHealth"),
        )

        ensure_start = start_end
        ensure_end = self.source.index("function Show-ProductionMenu", ensure_start)
        ensure_source = self.source[ensure_start:ensure_end]
        self.assertLess(
            ensure_source.index("Assert-ProductionRecoveryCleared"),
            ensure_source.index("Get-ProductionHealth"),
        )

        dispatch_start = self.source.index("$configuration = Read-DotEnv")
        dispatch_source = self.source[dispatch_start:]
        actions = {"start": "stop", "update": "restart", "restart": "logs"}
        for action, following in actions.items():
            with self.subTest(action=action):
                start = dispatch_source.index(f'"{action}" {{')
                end = dispatch_source.index(f'"{following}" {{', start)
                action_source = dispatch_source[start:end]
                self.assertIn("Assert-ProductionRecoveryCleared", action_source)

        unaffected = {"stop": "status", "status": "update", "logs": "open"}
        for action, following in unaffected.items():
            with self.subTest(action=action):
                start = dispatch_source.index(f'"{action}" {{')
                end = dispatch_source.index(f'"{following}" {{', start)
                action_source = dispatch_source[start:end]
                self.assertNotIn("Assert-ProductionRecoveryCleared", action_source)
        backup_start = dispatch_source.index('"backup" {')
        backup_end = dispatch_source.index('"clear-recovery-block" {', backup_start)
        self.assertNotIn(
            "Assert-ProductionRecoveryCleared",
            dispatch_source[backup_start:backup_end],
        )

    def test_recovery_block_clear_is_interactive_stopped_and_audited(self):
        clear_start = self.source.index("function Clear-ProductionRecoveryBlock")
        clear_end = self.source.index("function Show-Status", clear_start)
        clear_source = self.source[clear_start:clear_end]

        noninteractive = clear_source.index("if ($NonInteractive)")
        prompt = clear_source.index("Read-Host")
        self.assertLess(noninteractive, prompt)
        self.assertIn('Test-TrackedProcess $state "waitress_pid"', clear_source)
        self.assertIn('Test-TrackedProcess $state "caddy_pid"', clear_source)
        self.assertIn('Test-TcpPort "127.0.0.1" 8000', clear_source)
        self.assertIn('Test-TcpPort "127.0.0.1" 443', clear_source)
        self.assertIn("$confirmation -cne $block.ReleaseId", clear_source)
        reread = clear_source.index("$currentBlock = Get-ProductionRecoveryBlock")
        remove = clear_source.index("Remove-Item -LiteralPath $recoveryRequiredFile")
        self.assertLess(reread, remove)
        self.assertIn("Write-ProductionControlLog", clear_source)
        self.assertIn('"clear-recovery-block"', self.source)

    def test_operator_stop_is_persistent_except_for_release_engine(self):
        self.assertIn('"production-operator-stopped.json"', self.source)
        marker_start = self.source.index("function Set-ProductionOperatorStopped")
        marker_end = self.source.index("function Stop-Production", marker_start)
        marker_source = self.source[marker_start:marker_end]
        self.assertIn("if ($ReleaseToken) { return }", marker_source)
        self.assertIn("Set-Content", marker_source)
        self.assertIn("Remove-Item -LiteralPath $operatorStoppedFile", marker_source)

        ensure_start = self.source.index("function Ensure-Production")
        ensure_end = self.source.index("function Show-ProductionMenu", ensure_start)
        ensure_source = self.source[ensure_start:ensure_end]
        marker_check = ensure_source.index("Test-ProductionOperatorStopped")
        scheduled_return = ensure_source.index(
            "Ensure honored the operator-stopped marker", marker_check
        )
        health = ensure_source.index("Get-ProductionHealth")
        self.assertLess(marker_check, scheduled_return)
        self.assertLess(scheduled_return, health)
        self.assertIn("$operatorStopped -and -not $UserRequested", ensure_source)
        self.assertIn("Clear-ProductionOperatorStopped", ensure_source)

        dispatch_start = self.source.index("$configuration = Read-DotEnv")
        dispatch_source = self.source[dispatch_start:]
        stop_start = dispatch_source.index('"stop" {')
        stop_end = dispatch_source.index('"status" {', stop_start)
        stop_source = dispatch_source[stop_start:stop_end]
        self.assertIn("Set-ProductionOperatorStopped", stop_source)
        mutation_lock = stop_source.index("Invoke-WithProductionMutationLocks")
        set_marker = stop_source.index("Set-ProductionOperatorStopped")
        self.assertLess(mutation_lock, set_marker)

    def test_ensure_opens_healthy_site_and_repairs_incomplete_pair(self):
        ensure_start = self.source.index("function Ensure-Production")
        ensure_end = self.source.index("function Show-ProductionMenu", ensure_start)
        ensure_source = self.source[ensure_start:ensure_end]

        self.assertIn("$health = Get-ProductionHealth", ensure_source)
        self.assertIn("if ($health.IsHealthy)", ensure_source)
        self.assertIn("Open-ProductionSite $config", ensure_source)
        self.assertIn("if ($health.AnyTracked)", ensure_source)
        validation = ensure_source.index("Assert-ProductionConfiguration")
        stop = ensure_source.index("Stop-Production")
        start = ensure_source.index("Start-Production")
        self.assertLess(validation, stop)
        self.assertLess(stop, start)

    def test_start_rejects_partial_or_unhealthy_tracked_state(self):
        start = self.source.index("function Start-Production")
        end = self.source.index("function Ensure-Production", start)
        start_source = self.source[start:end]

        self.assertIn("if ($health.IsHealthy)", start_source)
        self.assertIn("if ($health.AnyTracked)", start_source)
        self.assertIn("partially running or unhealthy", start_source)

    def test_noninteractive_control_never_requests_credentials_or_uac(self):
        self.assertIn("[switch]$NonInteractive", self.source)

        login_start = self.source.index("function Ensure-DatabaseLogin")
        login_end = self.source.index("function Invoke-DatabaseBackup", login_start)
        login_source = self.source[login_start:login_end]
        self.assertIn("$AllowCredentialPrompt", login_source)
        self.assertIn("DB_PASSWORD is missing from .env", login_source)
        self.assertIn("The saved PostgreSQL password was rejected", login_source)
        missing_guard = login_source.index("if (-not $AllowCredentialPrompt)")
        first_prompt = login_source.index("Read-DatabasePassword $databaseUser")
        self.assertLess(missing_guard, first_prompt)
        rejected_error = login_source.index("The saved PostgreSQL password was rejected")
        retry_prompt = login_source.rindex("Read-DatabasePassword $databaseUser")
        self.assertLess(rejected_error, retry_prompt)

        stop_start = self.source.index("function Stop-TrackedProcessTree")
        stop_end = self.source.index("function Wait-TcpPortClosed", stop_start)
        stop_source = self.source[stop_start:stop_end]
        noninteractive = stop_source.index("if ($NonInteractive)")
        elevate = stop_source.index("Invoke-ElevatedProductionStop")
        self.assertLess(noninteractive, elevate)


class ProductionShortcutSourceTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        scripts = Path(settings.BASE_DIR) / "scripts"
        cls.hidden_runner = (scripts / "start-production-hidden.vbs").read_text(
            encoding="utf-8"
        )
        cls.installer = (scripts / "install-production-startup.ps1").read_text(
            encoding="utf-8"
        )

    def test_hidden_runner_uses_ensure_without_a_console(self):
        self.assertIn("-Action ensure -NonInteractive", self.hidden_runner)
        self.assertIn("shell.Run(command, 0, True)", self.hidden_runner)
        self.assertIn('Case "--no-browser"', self.hidden_runner)
        self.assertIn('Case "--user-requested"', self.hidden_runner)
        self.assertIn('command = command & " -UserRequested"', self.hidden_runner)
        self.assertIn('= "--probe"', self.hidden_runner)
        self.assertIn("Open Pharmacy Admin Control", self.hidden_runner)
        self.assertNotIn("production.bat", self.hidden_runner)

    def test_installer_creates_user_and_admin_shortcuts(self):
        self.assertIn('"Pharmacy.lnk"', self.installer)
        self.assertIn('"Pharmacy Admin Control.lnk"', self.installer)
        self.assertIn('`" --user-requested"', self.installer)
        self.assertIn('$legacyAdminShortcut', self.installer)
        self.assertIn('Remove-Item -LiteralPath $legacyAdminShortcut -Force', self.installer)
        self.assertIn('"start-production-hidden.vbs"', self.installer)
        self.assertIn('"production.bat"', self.installer)
        self.assertIn('[Environment]::GetFolderPath("Desktop")', self.installer)
        self.assertIn('[Environment]::GetFolderPath("Programs")', self.installer)

    def test_installer_requires_the_exact_main_production_role(self):
        self.assertIn('".runtime\\production-role.json"', self.installer)
        self.assertIn("function Assert-ProductionRole", self.installer)
        self.assertIn('$schemaVersion -ne 1', self.installer)
        self.assertIn('[string]$roleMarker.role -cne "production"', self.installer)
        self.assertIn('[string]$roleMarker.branch -cne "main"', self.installer)
        self.assertIn("[IO.Path]::IsPathRooted", self.installer)
        self.assertIn("[StringComparer]::OrdinalIgnoreCase.Equals", self.installer)
        self.assertIn("symbolic-ref --quiet --short HEAD", self.installer)
        role_check = self.installer.index("Assert-ProductionRole\n\n# Parse")
        shortcuts = self.installer.index("Install-ProductionShortcuts", role_check)
        self.assertLess(role_check, shortcuts)

    def test_optional_recovery_task_is_delayed_periodic_and_non_system(self):
        self.assertIn('$taskName = "Pharmacy Production Startup"', self.installer)
        self.assertIn("$logonTrigger = New-ScheduledTaskTrigger -AtLogOn", self.installer)
        self.assertIn('$logonTrigger.Delay = "PT30S"', self.installer)
        self.assertIn("-AtLogOn -User $RunAsUser", self.installer)
        self.assertIn("$recoveryTrigger = New-ScheduledTaskTrigger", self.installer)
        self.assertIn("-RepetitionInterval (New-TimeSpan -Minutes 5)", self.installer)
        self.assertIn("$triggers = @($logonTrigger, $recoveryTrigger)", self.installer)
        self.assertIn("-Trigger $triggers", self.installer)
        self.assertIn("-LogonType Interactive -RunLevel Limited", self.installer)
        self.assertIn("-UserId $RunAsUser", self.installer)
        self.assertIn("SYSTEM|LOCAL SERVICE|NETWORK SERVICE", self.installer)
        self.assertIn("-MultipleInstances IgnoreNew", self.installer)
        self.assertIn("--no-browser --quiet", self.installer)
        scheduled_action = self.installer.index("$action = New-ScheduledTaskAction")
        scheduled_end = self.installer.index("$logonTrigger", scheduled_action)
        self.assertNotIn(
            "--user-requested",
            self.installer[scheduled_action:scheduled_end],
        )
        self.assertIn('$verifiedLogonTrigger.Delay -ne "PT30S"', self.installer)
        self.assertIn(
            '$verifiedRecoveryTrigger.Repetition.Interval -ne "PT5M"',
            self.installer,
        )
        self.assertNotIn("Pharmacy Supplier Ordering", self.installer)
