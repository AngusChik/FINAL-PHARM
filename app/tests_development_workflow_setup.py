from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase


class DevelopmentWorkflowSetupSourceTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = (
            Path(settings.BASE_DIR)
            / "scripts"
            / "setup-development-workflow.ps1"
        ).read_text(encoding="utf-8")
        cls.task_migrator_source = (
            Path(settings.BASE_DIR)
            / "scripts"
            / "migrate-production-task-paths.ps1"
        ).read_text(encoding="utf-8")

    def test_production_worktree_cannot_be_inside_development(self):
        self.assertIn("function Assert-SafeProductionPath", self.source)
        self.assertIn(
            "$ProductionWorktree -ieq $projectRoot",
            self.source,
        )
        self.assertIn(
            "$productionWithSeparator.StartsWith($developmentWithSeparator",
            self.source,
        )
        self.assertIn(
            "Production worktree must be a direct sibling of development",
            self.source,
        )

    def test_setup_requires_a_clean_development_branch_and_known_remote(self):
        self.assertIn('git @Arguments', self.source)
        self.assertIn('$ErrorActionPreference = "Continue"', self.source)
        self.assertIn('$exitCode = $LASTEXITCODE', self.source)
        self.assertNotIn(
            ')\n        "Development folder is not a Git worktree")[-1]',
            self.source,
        )
        self.assertIn('$inside = @(\n        Invoke-Git', self.source)
        self.assertIn('status", "--porcelain"', self.source)
        self.assertIn("Commit or discard development changes", self.source)
        self.assertIn(
            'https://github.com/AngusChik/FINAL-PHARM.git',
            self.source,
        )
        self.assertIn(
            "Custom branch or remote names are not supported",
            self.source,
        )

    def test_first_cutover_requires_legacy_production_to_be_stopped(self):
        legacy_check = self.source.rindex("Assert-LegacyProductionStopped")
        branch_setup = self.source.rindex("Assert-DevelopmentRepository")
        self.assertLess(legacy_check, branch_setup)
        self.assertIn('foreach ($port in @(8000, 443))', self.source)
        self.assertIn('".runtime\\production.json"', self.source)
        self.assertIn(
            "production.bat stop from this checkout before the first cutover",
            self.source,
        )

    def test_production_runtime_is_separate_and_never_copies_development_env(self):
        self.assertIn('Copy-RuntimeFileIfMissing ".env" -Required', self.source)
        self.assertIn('"caddy_data"', self.source)
        self.assertIn('".mckesson_profile"', self.source)
        self.assertIn('".kohlfrisch_profile"', self.source)
        self.assertNotIn(
            'Copy-RuntimeFileIfMissing ".env.development"',
            self.source,
        )
        self.assertIn(
            'Join-Path $ProductionWorktree "env\\Scripts\\python.exe"',
            self.source,
        )

    def test_configuration_is_written_only_after_production_validation(self):
        role_marker = self.source.rindex("Write-ProductionRoleMarker")
        validation = self.source.rindex("Test-ProductionWorktree")
        startup_install = self.source.rindex(
            "if ($InstallStartup) { Install-ProductionStartupExperience }"
        )
        write_config = self.source.rindex("Write-WorkflowConfiguration")
        self.assertLess(role_marker, validation)
        self.assertLess(validation, startup_install)
        self.assertLess(startup_install, write_config)
        self.assertIn("$roleMarkerCreated", self.source)
        self.assertIn("Remove-Item -LiteralPath $roleMarkerPath -Force", self.source)
        self.assertIn("Assert-ExistingProductionRoleMarker", self.source)
        self.assertIn("development-workflow.json", self.source)
        self.assertIn("production_worktree = $ProductionWorktree", self.source)

    def test_existing_production_must_be_clean_and_exactly_at_origin_main(self):
        self.assertIn(
            "git -C $ProductionWorktree status --porcelain",
            self.source,
        )
        self.assertIn(
            "Production worktree must be clean before setup continues",
            self.source,
        )
        self.assertIn(
            'rev-parse "$Remote/$ProductionBranch"',
            self.source,
        )
        self.assertIn(
            "Production must exactly match $Remote/$ProductionBranch",
            self.source,
        )

    def test_runtime_files_and_directories_are_staged_before_activation(self):
        file_copy_start = self.source.index("function Copy-RuntimeFileIfMissing")
        directory_copy_start = self.source.index(
            "function Copy-RuntimeDirectoryIfMissing"
        )
        production_copy_start = self.source.index("function Copy-ProductionRuntime")
        file_copy = self.source[file_copy_start:directory_copy_start]
        directory_copy = self.source[directory_copy_start:production_copy_start]

        self.assertIn(
            '"$destination.setup-$([Guid]::NewGuid().ToString(\'N\'))"',
            file_copy,
        )
        self.assertLess(
            file_copy.index("Copy-Item -LiteralPath $source -Destination $temporary"),
            file_copy.index(
                "Move-Item -LiteralPath $temporary -Destination $destination"
            ),
        )
        self.assertIn("Remove-Item -LiteralPath $temporary -Force", file_copy)

        self.assertIn(
            '"$destination.setup-$([Guid]::NewGuid().ToString(\'N\'))"',
            directory_copy,
        )
        self.assertLess(
            directory_copy.index(
                "Copy-Item -LiteralPath $resolvedSource -Destination $temporary"
            ),
            directory_copy.index(
                "Move-Item -LiteralPath $temporary -Destination $destination"
            ),
        )
        self.assertIn(
            "Remove-Item -LiteralPath $temporary -Recurse -Force",
            directory_copy,
        )
        self.assertIn("function Write-JsonAtomic", self.source)

    def test_development_upstream_removal_is_checked(self):
        self.assertIn(
            'foreach ($settingName in @("remote", "merge", "pushRemote"))',
            self.source,
        )
        self.assertIn('$configKey = "branch.$DevelopmentBranch.$settingName"', self.source)
        self.assertIn(
            'Invoke-Git @("config", "--unset-all", $configKey)',
            self.source,
        )
        self.assertIn("$remainingUpstream", self.source)
        self.assertIn('$upstreamExitCode = $LASTEXITCODE', self.source)
        self.assertIn(
            '$upstreamExitCode -eq 0 -or ($remainingUpstream -join "").Trim()',
            self.source,
        )
        self.assertIn(
            "Development branch must remain local-only and have no upstream",
            self.source,
        )

    def test_existing_development_branch_must_match_reviewed_commit(self):
        self.assertIn("$reviewedStartCommit", self.source)
        self.assertIn("$existingDevelopmentCommit", self.source)
        self.assertIn(
            "already exists at a different",
            self.source,
        )
        self.assertLess(
            self.source.index("$existingDevelopmentCommit -ne $reviewedStartCommit"),
            self.source.index('Invoke-Git @("switch", $DevelopmentBranch)'),
        )

    def test_installer_can_add_shortcuts_and_autostart_from_production(self):
        self.assertIn("function Install-ProductionStartupExperience", self.source)
        self.assertIn("install-production-startup.ps1", self.source)
        self.assertIn('if ($EnableAutoStart)', self.source)
        self.assertIn(
            'if ($EnableAutoStart -and -not $InstallStartup)',
            self.source,
        )
        self.assertIn("migrate-production-task-paths.ps1", self.source)
        self.assertIn('-DevelopmentWorktree $projectRoot', self.source)

    def test_task_migrator_rebinds_existing_actions_and_rejects_development(self):
        self.assertIn("Assert-ProductionRole", self.task_migrator_source)
        self.assertIn(
            'Name = "Pharmacy Scheduled Jobs"',
            self.task_migrator_source,
        )
        self.assertIn(
            'Name = "Pharmacy Supplier Ordering"',
            self.task_migrator_source,
        )
        self.assertIn(
            'Name = "Pharmacy Production Startup"',
            self.task_migrator_source,
        )
        self.assertIn(
            "start-production-hidden.vbs",
            self.task_migrator_source,
        )
        self.assertIn(
            "--no-browser --quiet",
            self.task_migrator_source,
        )
        self.assertIn(
            "Set-ScheduledTask -TaskName $mapping.Name -Action $action",
            self.task_migrator_source,
        )
        self.assertIn("$principalFingerprint", self.task_migrator_source)
        self.assertIn("$triggerFingerprint", self.task_migrator_source)
        self.assertIn("$changedTasks.Add", self.task_migrator_source)
        self.assertIn(
            "Set-ScheduledTask -TaskName $changed.Name -Action $changed.Action",
            self.task_migrator_source,
        )
        self.assertIn(
            "still executes from development",
            self.task_migrator_source,
        )
        self.assertLess(
            self.task_migrator_source.index(
                'Name = "Pharmacy Production Startup"'
            ),
            self.task_migrator_source.index(
                "still executes from development"
            ),
        )
